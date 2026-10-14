"""Render a frozen *trained* defender with its CoT scaffold during attacker training.

``red_team/utils.py`` lives in the vendored upstream tree, not in this repo, so
this patch is applied to ``$SSP_WORK`` after ``prepare_sources.py`` has
materialized it.  It supersedes ``patch_defender_prefill.py``.

Two independent decisions were both keyed on ``no_defender_turn``:

* the defender's instruction wrapper -- ``DEFENDER_INSTRUCTION_COT_PROMPT``
  versus a bare ``"{user_query}"`` (upstream);
* the assistant prefill ``"Let's think step by step.\\n<think> "``
  (added by the superseded patch).

``no_defender_turn`` means "the defender's turns are not optimizer items"; it is
also consumed by ``language_game.py`` and ``replay_buffer.py`` for exactly that.
Overloading it to mean "the defender cannot read a CoT template" was only ever
true at generation 1, where the frozen opponent is the untrained base model.
From A2 onward every opponent with non-negligible mixture weight is a *trained*
defender that was trained with the template and is rendered *with* it in every
payoff cell (``direct_base_defender = raw_defender_path is None`` in
``modal_upstream_v2_payoff.py``) and in every benchmark (``chat_template:
llama3_cot``).  Attacker training was therefore optimizing against a rendering
nothing else uses.

This patch routes both decisions through one predicate:

* ``direct_chat_no_cot``            -> no wrapper, no prefill (unchanged);
* ``no_defender_turn`` and the pool contains an adapter-less opponent
                                    -> no wrapper, no prefill (base model: the
                                       prefill jailbreaks it, measured 0.972 vs
                                       a true 0.328-0.438);
* ``no_defender_turn`` and every weighted opponent has an adapter
                                    -> wrapper AND prefill, matching payoff;
* otherwise (defender training)     -> wrapper AND prefill (unchanged).

``role_phase.sh`` decides the middle two by inspecting the opponent pool and
setting ``defender_cot_when_frozen``.  The attacker keeps its own prefill on
every path (``_render_attacker_prompt``), so this is defender-only.

    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_defender_cot.py
    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_defender_cot.py --check

Idempotent: re-running is a no-op, and ``--check`` exits non-zero if the patch
is absent.
"""

from __future__ import annotations

import argparse
import os
import re
import pathlib
import sys

HELPER = '''
def defender_uses_cot(custom_configs) -> bool:
    """Whether the defender's turn gets the CoT wrapper and the assistant prefill.

    Applied to the wrapper in ``convert_game_history_to_query`` and to the
    prefill in ``convert_game_history_to_messages`` so the two can never drift
    apart -- they drifted once already, and prefilling an unwrapped base
    defender is a jailbreak rather than a scaffold.
    """
    if not custom_configs:
        return True
    if custom_configs.get("direct_chat_no_cot", False):
        return False
    if custom_configs.get("no_defender_turn", False):
        # Frozen opponent.  Only a trained defender knows the template; the
        # base model does not, and role_phase.sh clears this flag whenever an
        # adapter-less opponent carries weight in the mixture.
        return bool(custom_configs.get("defender_cot_when_frozen", False))
    return True

'''

# --- prefill site -----------------------------------------------------------
PREFILL_OLD_UNPATCHED = '''    if custom_configs and custom_configs.get("direct_chat_no_cot", False):
        pass
    else:
        chat_message += ASSISTANT_THINKING_PREFIX
'''

PREFILL_OLD_PREVIOUS_PATCH_MARKER = 'plain_defender = player_role != "attacker"'

PREFILL_NEW = '''    if player_role != "attacker" and not defender_uses_cot(custom_configs):
        pass
    elif custom_configs and custom_configs.get("direct_chat_no_cot", False):
        pass
    else:
        chat_message += ASSISTANT_THINKING_PREFIX
'''

# --- wrapper site -----------------------------------------------------------
# Matched by regex: upstream leaves trailing whitespace on the COT_PROMPT line,
# which a byte-exact literal silently fails to match.
WRAPPER_RE = re.compile(
    r"[ \t]*if custom_configs and \(custom_configs\.get\(\"direct_chat_no_cot\".*?\n"
    r"(?:[ \t]*#[^\n]*\n)?"
    r"[ \t]*role_instruction_prompt = \"\{user_query\}\"[ \t]*\n"
    r"[ \t]*else:[ \t]*\n"
    r"[ \t]*role_instruction_prompt = DEFENDER_INSTRUCTION_COT_PROMPT[ \t]*\n",
    re.DOTALL,
)

WRAPPER_NEW = '''        if not defender_uses_cot(custom_configs):
            # No wrapper only for a genuinely plain chat model: direct-chat mode,
            # or a frozen opponent pool that still carries an adapter-less
            # (base) defender.  A trained frozen defender keeps the template.
            role_instruction_prompt = "{user_query}"
        else:
            role_instruction_prompt = DEFENDER_INSTRUCTION_COT_PROMPT
'''

MARKER = "def defender_uses_cot("


def apply(text: str) -> tuple[str, bool]:
    """Return ``(patched_text, changed)``; an already-patched input is returned as-is."""
    if MARKER in text:
        return text, False

    # Insert the helper just above the first renderer that uses it.
    anchor = "def convert_game_history_to_messages("
    if text.count(anchor) != 1:
        raise SystemExit(
            f"red_team/utils.py has {text.count(anchor)} definitions of "
            f"{anchor!r}; upstream moved"
        )
    text = text.replace(anchor, HELPER.lstrip("\n") + "\n" + anchor, 1)

    # Prefill: accept either the pristine upstream block or the superseded patch.
    if text.count(PREFILL_OLD_UNPATCHED) == 1:
        text = text.replace(PREFILL_OLD_UNPATCHED, PREFILL_NEW, 1)
    elif PREFILL_OLD_PREVIOUS_PATCH_MARKER in text:
        start = text.index("    plain_defender = player_role")
        end = text.index("chat_message += ASSISTANT_THINKING_PREFIX", start)
        end = text.index("\n", end) + 1
        text = text[:start] + PREFILL_NEW + text[end:]
    else:
        raise SystemExit(
            "red_team/utils.py prefill block matches neither the unpatched "
            "upstream form nor the superseded patch; upstream moved"
        )

    found = WRAPPER_RE.findall(text)
    if len(found) != 1:
        raise SystemExit(
            "red_team/utils.py does not contain exactly one unpatched defender "
            f"wrapper block (found {len(found)}); upstream moved"
        )
    text = WRAPPER_RE.sub(lambda _m: WRAPPER_NEW, text, count=1)
    return text, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=os.environ.get("SSP_WORK"))
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero unless the patch is already applied")
    args = parser.parse_args()

    work = args.work or (
        pathlib.Path(os.environ["SSP_ROOT"]) / "work"
        if os.environ.get("SSP_ROOT") else None
    )
    if work is None:
        raise SystemExit("set --work or SSP_WORK/SSP_ROOT")
    target = pathlib.Path(work) / "red_team" / "utils.py"
    if not target.is_file():
        raise SystemExit(f"missing {target}; run prepare_sources.py first")

    text = target.read_text(encoding="utf-8")
    if args.check:
        if MARKER not in text:
            print(f"NOT PATCHED: {target}", file=sys.stderr)
            return 1
        print(f"patched: {target}")
        return 0

    patched, changed = apply(text)
    if changed:
        backup = target.with_suffix(target.suffix + ".bak_cot_gate")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        target.write_text(patched, encoding="utf-8")
        print(f"patched {target} (backup {backup.name})")
    else:
        print(f"already patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
