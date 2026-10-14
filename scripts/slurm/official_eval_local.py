#!/usr/bin/env python3
"""Run the official defender evaluation for one PSRO run, without Modal.

The coordinator's ``kind == "evaluation"`` action calls
``selfredteam-official-eval.evaluate_full_checkpoint_vs_base`` once per learned
defender, with output_slug ``{init_mode}_psro_{suffix}_{defender}``.  This
reproduces exactly that loop over D1..D<generations> of a finished run, reading
the defender checkpoints out of the run's own state.json, and records each
result back into state.json under ``evaluations`` so a resumed job skips what
is already done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402


def log(message: str) -> None:
    print(f"[eval {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--eval-output-root", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--label-prefix", default="cold_psro")
    parser.add_argument("--defenders", default="")
    parser.add_argument("--coordinator", default=common.COLD_COORDINATOR)
    args = parser.parse_args()

    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_CACHE"] = str(args.hf_home / "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["SSP_EVAL_UPSTREAM"] = str(args.upstream)
    os.environ["SSP_EVAL_OUTPUT_ROOT"] = str(args.eval_output_root)
    os.environ["SSP_EVAL_GPUS"] = str(args.gpus)
    if args.tasks:
        os.environ["SSP_EVAL_TASKS"] = args.tasks

    common._stub_modal()
    if str(common.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(common.REPO_ROOT))
    import modal_selfredteam_official_eval_local as official

    evaluate = official.evaluate_full_checkpoint_vs_base
    evaluate = getattr(evaluate, "get_raw_f", lambda: evaluate)()

    run_root = args.psro_root / args.suffix
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    generations = int(state["contract"]["generations"])
    if args.defenders:
        defenders = args.defenders.split(",")
    else:
        defenders = [f"D{index}" for index in range(1, generations + 1)]

    log(f"run {args.suffix}: evaluating {defenders}")
    log(f"tasks: {official.FULL_DEFENDER_TASKS}")
    for defender in defenders:
        record = state["population"].get(defender)
        if not record or not record.get("path"):
            raise SystemExit(f"{defender} is not in the population yet")
        if defender in state.get("evaluations", {}):
            log(f"{defender} already evaluated; skipping")
            continue
        log(f"evaluating {defender}: {record['path']}")
        result = evaluate(
            trained_checkpoint=record["path"],
            output_slug=f"{args.label_prefix}_{args.suffix}_{defender}",
            trained_label=f"{args.label_prefix}_{defender}",
            evaluate_base=False,
        )
        # Re-read: a concurrent driver may have advanced the same file.
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("evaluations", {})[defender] = result
        temporary = state_path.with_name(".state.json.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(state_path)
        log(f"{defender} done -> {result}")
    log("all defender evaluations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
