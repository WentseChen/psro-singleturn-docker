"""Stop prefilling a frozen defender's turn with the CoT scaffold.

``red_team/utils.py`` lives in the vendored upstream tree, not in this repo, so
this patch is applied to ``$SSP_WORK`` after ``prepare_sources.py`` has
materialized it.

Upstream appends ``ASSISTANT_THINKING_PREFIX`` ("Let's think step by step.\\n
<think> ") to every rendered turn.  That prefix is a chain-of-thought scaffold
for a model trained to continue it, and the trainable role must keep it.  A
*frozen* defender is different: when ``no_defender_turn`` is set the defender is
already rendered with the bare ``"{user_query}"`` wrapper precisely because a
non-trainable defender "does not know how to use the CoT template", and
prefilling its reply then jailbreaks it.

Measured on ``cold_psro_v6c_s300_race_20260908`` A1: training reported
``successful_harmful_request_rate`` rising 0.555 -> 0.972, while all fifteen
retained checkpoints scored a flat 0.328-0.438 on held-out prompts against the
same base defender rendered *without* the prefill (mean 0.367, SE 0.060 at
n=64).  Held-out selection had no signal to choose on and picked step 20 out of
the noise.  The payoff cells and every external benchmark send no prefill to a
base defender (``direct_base_defender`` in ``modal_upstream_v2_payoff.py``), so
training was optimizing against an opponent nothing else uses.

The attacker keeps its prefill in both paths -- ``_render_attacker_prompt``
prefills too -- so this is deliberately defender-only, and it fires exactly
where the instruction wrapper is already dropped so the two stay in lockstep.

    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_defender_prefill.py
    SSP_ROOT=/path/to/workspace python scripts/slurm/patch_defender_prefill.py --check

Idempotent: re-running is a no-op, and ``--check`` exits non-zero if the patch
is absent.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

BEFORE = '''    if custom_configs and custom_configs.get("direct_chat_no_cot", False):
        pass
    else:
        chat_message += ASSISTANT_THINKING_PREFIX
'''

AFTER = '''    # The assistant prefill is a CoT scaffold for a model that was trained to
    # continue it.  A defender rendered with the bare "{user_query}" wrapper is
    # a plain chat model that never learned the template, and prefilling its
    # turn with "Let's think step by step.\\n<think> " jailbreaks it: attacker
    # training against a prefilled base defender reached 0.972 harmful-request
    # success while the same 15 checkpoints measured a flat 0.328-0.438 against
    # the unprefilled defender that every payoff cell and external benchmark
    # actually uses.  Drop the prefill exactly where the wrapper is dropped, so
    # the two stay in lockstep.  The attacker keeps its prefill in both paths
    # (see _render_attacker_prompt in modal_upstream_v2_payoff.py).
    plain_defender = player_role != "attacker" and bool(
        custom_configs
        and (
            custom_configs.get("direct_chat_no_cot", False)
            or custom_configs.get("no_defender_turn", False)
        )
    )
    if (custom_configs and custom_configs.get("direct_chat_no_cot", False)) or plain_defender:
        pass
    else:
        chat_message += ASSISTANT_THINKING_PREFIX
'''

MARKER = "plain_defender = player_role != \"attacker\""


def apply(text: str) -> tuple[str, bool]:
    """Return ``(patched_text, changed)``; a patched input is returned as-is."""
    if MARKER in text:
        return text, False
    if text.count(BEFORE) != 1:
        raise SystemExit(
            "red_team/utils.py does not contain exactly one unpatched "
            f"prefill block (found {text.count(BEFORE)}); upstream moved"
        )
    return text.replace(BEFORE, AFTER, 1), True


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
        target.write_text(patched, encoding="utf-8")
        print(f"patched {target}")
    else:
        print(f"already patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
