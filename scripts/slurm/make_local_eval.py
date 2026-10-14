#!/usr/bin/env python3
"""Derive a Slurm-runnable copy of the official defender evaluation.

The PSRO contract names this workflow verbatim ("current selfredteam-official-eval
defender workflow for every D1-D5"), so the evaluation itself must not be
reimplemented: it merges the defender adapter into a full checkpoint, runs the
released ``safety-eval-fork`` over the eight defender tasks, and writes
comparison.json against the published baseline column values.

Only the things Modal owns are parameterised, and each becomes an environment
variable that defaults to the original literal:

* the upstream checkout that supplies ``eval/benchmarks/safety-eval-fork``
* the ``/output`` volume path the reports are written under
* the two-GPU worker pool, which on Modal came from ``gpu="A100-40GB:2"``
* the Hugging Face token check, which on Modal read a mandatory secret; here it
  is only required by ``wildguardtest``, the one task whose dataset is gated
* the task list, so a sanity check can run a single task

    python scripts/slurm/make_local_eval.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "modal_selfredteam_official_eval.py"
DESTINATION = REPO_ROOT / "modal_selfredteam_official_eval_local.py"

SUBSTITUTIONS = [
    (
        'UPSTREAM_REMOTE = "/selfplay-redteaming"',
        'UPSTREAM_REMOTE = os.environ.get(\n'
        '    "SSP_EVAL_UPSTREAM", "/selfplay-redteaming"\n'
        ')',
    ),
    (
        '    output_root = Path("/output/upstream_selfredteam_role_full_eval") / output_slug',
        '    output_root = Path(\n'
        '        os.environ.get(\n'
        '            "SSP_EVAL_OUTPUT_ROOT",\n'
        '            "/output/upstream_selfredteam_role_full_eval",\n'
        '        )\n'
        '    ) / output_slug',
    ),
    (
        '    for gpu_id in range(2):',
        '    for gpu_id in range(int(os.environ.get("SSP_EVAL_GPUS", "2"))):',
    ),
    (
        '    with ThreadPoolExecutor(max_workers=2) as executor:',
        '    with ThreadPoolExecutor(\n'
        '        max_workers=int(os.environ.get("SSP_EVAL_GPUS", "2"))\n'
        '    ) as executor:',
    ),
    (
        '''    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("roll-secrets does not contain a Hugging Face token")
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fork_root = Path(UPSTREAM_REMOTE) / "eval/benchmarks/safety-eval-fork"''',
        '''    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
    elif any("wildguardtest" in task for task in FULL_DEFENDER_TASKS):
        # Only wildguardtest downloads a gated dataset (allenai/wildguardmix);
        # the other seven tasks read files shipped inside the fork.
        raise RuntimeError(
            "wildguardtest needs HF_TOKEN for the gated allenai/wildguardmix"
        )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fork_root = Path(UPSTREAM_REMOTE) / "eval/benchmarks/safety-eval-fork"''',
    ),
    (
        'FULL_DEFENDER_TASKS = CORE_TASKS + BENIGN_BOUNDARY_TASKS',
        'FULL_DEFENDER_TASKS = tuple(\n'
        '    os.environ["SSP_EVAL_TASKS"].split(",")\n'
        ') if os.environ.get("SSP_EVAL_TASKS") else CORE_TASKS + BENIGN_BOUNDARY_TASKS',
    ),
]

HEADER = """\
# GENERATED -- do not edit.
#
# Derived from modal_selfredteam_official_eval.py by
# scripts/slurm/make_local_eval.py. Every difference is an environment variable
# whose default is the original literal; see
# modal_selfredteam_official_eval_local.provenance.json for the exact
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
            raise SystemExit(f"expected 1 occurrence, found {count}:\n{old[:120]}")
        derived = derived.replace(old, new, 1)
    derived = HEADER + derived
    DESTINATION.write_text(derived, encoding="utf-8")
    diff = list(
        difflib.unified_diff(
            original.splitlines(), derived.splitlines(),
            fromfile=SOURCE.name, tofile=DESTINATION.name, lineterm="", n=1,
        )
    )
    DESTINATION.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "local-official-eval-derivation-v1",
                "source": SOURCE.name,
                "source_sha256": _sha256(original),
                "derived_sha256": _sha256(derived),
                "substitutions": [{"from": o, "to": n} for o, n in SUBSTITUTIONS],
                "diff": diff,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {DESTINATION}")
    print(f"{len(diff)} diff lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
