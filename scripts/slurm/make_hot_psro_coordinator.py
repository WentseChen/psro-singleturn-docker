#!/usr/bin/env python3
"""Derive the hot-start PSRO coordinator from the cold one.

Hot start differs from cold start in exactly one place in the algorithm: an
oracle continues the previous same-role adapter instead of opening a fresh one
on the base model.  That change lives in the local trainer
(scripts/slurm/cold_psro_local.py), not in the coordinator -- the schedule, the
payoff cells, the meta-game solve and the frozen-mixture opponent rule are
identical.

What the coordinator does own is the run *contract*, which every call
re-derives and compares against the stored one.  Running hot under the cold
coordinator would stamp ``init_mode: "cold"`` onto a warm-started run and name
its W&B runs ``role_lora_cold_psro__*``.  So this script emits a derived
coordinator whose only differences are those labels, and records the exact
substitutions plus both file hashes in a sidecar.  ``_psro_implementation_hashes``
hashes ``__file__``, so the generated file -- not the original -- is what the
contract pins.

    python scripts/slurm/make_hot_psro_coordinator.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "modal_role_lora_zero_sum_psro.py"
DESTINATION = REPO_ROOT / "modal_role_lora_zero_sum_psro_hot.py"

SUBSTITUTIONS = [
    (
        '''        "init_mode": "cold",
        "cold_definition": (
            "every learned A_i and D_i starts from the same base with a new "
            f"rank-{lora_rank} adapter and fresh optimizer"
        ),''',
        '''        "init_mode": "hot",
        "hot_definition": (
            "every learned A_i and D_i continues the previous same-role "
            f"rank-{lora_rank} adapter with a fresh optimizer; A1 and D1 open "
            "a new adapter on the base model"
        ),''',
    ),
    (
        '            wandb_identity=f"role_lora_cold_psro__{suffix}__{target}",',
        '            wandb_identity=f"role_lora_hot_psro__{suffix}__{target}",',
    ),
]

# A comment block, not a docstring: the source starts with `from __future__`,
# which must be the first statement in the file.
HEADER = """\
# GENERATED -- do not edit.
#
# Derived from modal_role_lora_zero_sum_psro.py by
# scripts/slurm/make_hot_psro_coordinator.py. The only differences are the
# contract's init-mode labels and the W&B run-name prefix; see
# modal_role_lora_zero_sum_psro_hot.provenance.json for the exact
# substitutions. Regenerate rather than editing this file.
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    original = SOURCE.read_text(encoding="utf-8")
    derived = original
    for old, new in SUBSTITUTIONS:
        count = derived.count(old)
        if count != 1:
            raise SystemExit(
                f"expected exactly one occurrence, found {count}:\n{old}"
            )
        derived = derived.replace(old, new, 1)
    derived = HEADER + derived

    DESTINATION.write_text(derived, encoding="utf-8")
    diff = list(
        difflib.unified_diff(
            original.splitlines(), derived.splitlines(),
            fromfile=SOURCE.name, tofile=DESTINATION.name, lineterm="", n=1,
        )
    )
    provenance = {
        "schema_version": "hot-psro-coordinator-derivation-v1",
        "source": SOURCE.name,
        "source_sha256": _sha256(original),
        "derived": DESTINATION.name,
        "derived_sha256": _sha256(derived),
        "substitutions": [{"from": old, "to": new} for old, new in SUBSTITUTIONS],
        "note": (
            "train_cold_start_iteration_one is a legacy one-shot entrypoint "
            "that advance_cold_psro never calls; its contract literal is left "
            "untouched."
        ),
        "diff": diff,
    }
    DESTINATION.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {DESTINATION}")
    print("\n".join(diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
