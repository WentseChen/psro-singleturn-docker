#!/usr/bin/env python3
"""Run the official defender evaluation on explicit checkpoint paths.

official_eval_local.py can only evaluate defenders registered in a PSRO run's
state.json population.  Follow-up runs produced by run_single_oracle.py live
outside any state.json, so this takes <label>=<checkpoint> pairs directly and
uses the same evaluator, template and task list.
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
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--eval-output-root", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="where to write the collected result index")
    parser.add_argument("targets", nargs="+",
                        help="LABEL=/path/to/checkpoint pairs")
    args = parser.parse_args()

    pairs = []
    for target in args.targets:
        if "=" not in target:
            raise SystemExit(f"expected LABEL=/path, got {target!r}")
        label, _, path = target.partition("=")
        checkpoint = Path(path)
        if not checkpoint.is_dir():
            raise SystemExit(f"missing checkpoint for {label}: {checkpoint}")
        pairs.append((label, checkpoint))

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

    log(f"evaluating {[label for label, _ in pairs]}")
    log(f"tasks: {official.FULL_DEFENDER_TASKS}")
    results = {}
    for label, checkpoint in pairs:
        log(f"evaluating {label}: {checkpoint}")
        results[label] = evaluate(
            trained_checkpoint=str(checkpoint),
            output_slug=label,
            trained_label=label,
            evaluate_base=False,
        )
        log(f"{label} done -> {results[label]}")
        if args.results_json:
            args.results_json.parent.mkdir(parents=True, exist_ok=True)
            args.results_json.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log("all checkpoint evaluations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
