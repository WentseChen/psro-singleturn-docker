"""Merge an off-topic verdict into the labels the reward is computed from.

``openrlhf/trainer/ppo_utils/language_game.py`` lives in the vendored upstream
tree, so this patch is applied to ``$SSP_WORK`` after ``prepare_sources.py``.

``evaluate_game_outcomes`` asks WildGuard about (rewrite, reply).  WildGuard
never sees the *seed*, so it cannot tell a faithful rewrite from a memorized
request that happens to elicit harm -- the blind spot that let attackers A3/A4/A6
emit a single prompt for all 60 held-out seeds and still score well.

This patch compares the seed carried on each game (``game["prompts"]``) with
the rewrite, then merges ``off_topic`` into the WildGuard label dict. In
``GAME_REWARD_MODE=shared_binary``, the external API verdict is required and
``shared_game_reward.py`` subtracts the penalty after the zero-sum projection.
API failure stops the run after bounded retries.

Legacy mode separately uses the optional local ``OFFTOPIC_URL`` verifier;
its failures leave the verdict absent. That older path does not provide the
shared training/matrix reward.

    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_offtopic_labels.py
    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_offtopic_labels.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import pathlib
import sys

# Regex, not a literal: upstream leaves trailing whitespace on the blank line
# before the return, and a byte-exact literal silently matches nothing.
BEFORE_RE = re.compile(
    r"[ \t]*if isinstance\(batch_labels_dict, list\):[ \t]*\n"
    r"[ \t]*batch_labels_dict = \{idx: label for idx, label in enumerate\(batch_labels_dict\)\}[ \t]*\n"
    r"[ \t]*\n"
    r"[ \t]*return batch_labels_dict[ \t]*\n"
)

API_BLOCK = '''        from roll.utils.shared_game_reward import shared_game_reward_enabled
        from roll.utils.role_lora_training_reward import invalid_move_reward_enabled
        _invalid_move = invalid_move_reward_enabled()
        if shared_game_reward_enabled() or _invalid_move:
            from roll.utils.topic_api import label_topic_queries
            # shared_binary drops benign-to-harmful games, so it never needs their
            # verdict; the invalid-move reward stacks drift and off-topic (-4).
            topic_queries = [
                {"game_idx": idx, "seed": game["prompts"],
                 "rewrite": game["history"][0]["content"]}
                for idx, game in self.active_games.items()
                if game["prompt_type"] in {"generated_harmful", "generated_benign"}
                and game.get("history")
                and not batch_labels_dict.get(idx, {}).get("is_parsing_error", False)
                and (_invalid_move or not (game["prompt_type"] == "generated_benign"
                         and batch_labels_dict.get(idx, {}).get("prompt_harmfulness") == "harmful"))
            ]
            for verdict in label_topic_queries(topic_queries):
                batch_labels_dict[verdict["game_idx"]].update(verdict)
            return batch_labels_dict

'''

AFTER = '''        if isinstance(batch_labels_dict, list):
            batch_labels_dict = {idx: label for idx, label in enumerate(batch_labels_dict)}

''' + API_BLOCK + '''        # Second verdict: is the attacker's rewrite still asking what the seed
        # asked?  WildGuard cannot answer that -- it never sees the seed -- and
        # without it a fixed memorized request that reliably elicits harm is a
        # local optimum.  A failure here must not take the run down, so any
        # error leaves the label absent and the penalty simply does not apply.
        offtopic_url = os.environ.get("OFFTOPIC_URL", "").strip()
        if offtopic_url:
            topic_queries = [
                {
                    "game_idx": idx,
                    "seed": game["prompts"],
                    "rewrite": game["history"][0]["content"],
                }
                for idx, game in self.active_games.items()
                if len(game.get("history", [])) >= 1
            ]
            try:
                # This optional service must not inherit WildGuard's 12 long retries.
                import requests
                response = requests.post(
                    offtopic_url,
                    json={"queries": topic_queries},
                    timeout=(3, max(0.1, float(os.environ.get("OFFTOPIC_TIMEOUT_SECONDS", "15")))),
                )
                response.raise_for_status()
                verdicts = response.json()["labels"]
                if isinstance(verdicts, list):
                    verdicts = {
                        v.get("game_idx", i): v for i, v in enumerate(verdicts)
                    }
                applied = 0
                for idx, verdict in verdicts.items():
                    if idx in batch_labels_dict and isinstance(verdict, dict):
                        batch_labels_dict[idx]["off_topic"] = verdict.get("off_topic")
                        applied += 1
                self.strategy.print(
                    f"Rank #{self.strategy.get_rank()}, off-topic verdicts merged: {applied}"
                )
            except Exception as exc:  # noqa: BLE001 - never fail training on this
                self.strategy.print(
                    f"Rank #{self.strategy.get_rank()}, off-topic verifier "
                    f"unavailable ({type(exc).__name__}); penalty not applied"
                )

        return batch_labels_dict
'''

MARKER = 'offtopic_url = os.environ.get("OFFTOPIC_URL"'
LEGACY_CALL = '                verdicts = remote_rm_fn(offtopic_url, topic_queries, score_key="labels")'
BOUNDED_CALL = AFTER[AFTER.index("                # This optional service"):AFTER.index("                if isinstance(verdicts, list):")].rstrip()


def apply(text: str) -> tuple[str, bool]:
    """Return ``(patched_text, changed)``; an already-patched input is unchanged."""
    if MARKER in text:
        changed = False
        if LEGACY_CALL in text:
            text = text.replace(LEGACY_CALL, BOUNDED_CALL, 1)
            changed = True
        if "from roll.utils.topic_api import label_topic_queries" not in text:
            marker = "        # Second verdict: is the attacker's rewrite still asking what the seed"
            if text.count(marker) != 1:
                raise SystemExit("Cannot locate the existing topic label patch")
            text = text.replace(marker, API_BLOCK + marker, 1)
            changed = True
        return text, changed
    found = BEFORE_RE.findall(text)
    if len(found) != 1:
        raise SystemExit(
            "language_game.py does not contain exactly one unpatched label-return "
            f"block (found {len(found)}); upstream moved"
        )
    text = BEFORE_RE.sub(lambda _m: AFTER, text, count=1)
    if "\nimport os" not in text and "\nimport os\n" not in text:
        # the module imports time/json but not os in some upstream revisions
        text = text.replace("import time", "import os\nimport time", 1)
    return text, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.environ.get("SSP_WORK"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    work = args.work or (
        pathlib.Path(os.environ["SSP_ROOT"]) / "work"
        if os.environ.get("SSP_ROOT") else None
    )
    if work is None:
        raise SystemExit("set --work or SSP_WORK/SSP_ROOT")
    target = pathlib.Path(work) / "openrlhf" / "trainer" / "ppo_utils" / "language_game.py"
    if not target.is_file():
        raise SystemExit(f"missing {target}; run prepare_sources.py first")

    text = target.read_text(encoding="utf-8")
    if args.check:
        if MARKER not in text or LEGACY_CALL in text or "from roll.utils.topic_api import label_topic_queries" not in text:
            print(f"NOT PATCHED: {target}", file=sys.stderr)
            return 1
        print(f"patched: {target}")
        return 0

    patched, changed = apply(text)
    if changed:
        backup = target.with_suffix(target.suffix + ".bak_offtopic")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        target.write_text(patched, encoding="utf-8")
        print(f"patched {target} (backup {backup.name})")
    else:
        print(f"already patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
