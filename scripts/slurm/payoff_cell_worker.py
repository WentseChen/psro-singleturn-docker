#!/usr/bin/env python3
"""Evaluate one cold-PSRO payoff cell in its own process, on one GPU.

The Modal coordinator runs a cell per container.  The payoff generator builds a
single-GPU vLLM engine at ``gpu_memory_utilization=0.90``, so one cell per card
is the same shape here, and cells are independent -- nothing is shared but the
judge.  The driver runs three of these against GPUs 1-3 while the judge holds
GPU 0, then re-enters the coordinator, which finds the completed raw cell and
returns it without touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", default=common.COLD_COORDINATOR)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--raw-payoff-root", required=True, type=Path)
    parser.add_argument("--upstream-source", required=True, type=Path)
    parser.add_argument("--judge-url", required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--attacker-label", required=True)
    parser.add_argument("--defender-label", required=True)
    parser.add_argument("--attacker-adapter", required=True)
    parser.add_argument("--defender-adapter", required=True)
    parser.add_argument("--attacker-sha256", required=True)
    parser.add_argument("--defender-sha256", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--max-candidate-multiplier", type=int, required=True)
    parser.add_argument("--candidate-wave-pairs", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, required=True)
    parser.add_argument("--judge-batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument(
        "--adaptive-stratum-sampling", action="store_true",
        help="size each stratum by its observed retention instead of 1:1",
    )
    args = parser.parse_args()

    psro, payoff = common.load_modules(args.coordinator)
    common.retarget(
        psro,
        payoff,
        psro_root=args.psro_root,
        raw_payoff_root=args.raw_payoff_root,
        upstream_source=args.upstream_source,
        judge_url=args.judge_url,
    )

    # Upstream copies each adapter to a fixed /tmp name before handing it to
    # PEFT; concurrent cells would overwrite each other's copy mid-load.
    original = payoff._prepare_peft_compatible_adapter

    def _unique(source: str, destination_name: str = "attacker_lora_init_compatible") -> str:
        return original(source, destination_name=f"{destination_name}_{os.getpid()}")

    payoff._prepare_peft_compatible_adapter = _unique

    summary = psro.evaluate_zero_sum_psro_cell(
        psro_run_suffix=args.run_suffix,
        attacker_label=args.attacker_label,
        defender_label=args.defender_label,
        attacker_adapter=args.attacker_adapter,
        defender_adapter=args.defender_adapter,
        attacker_sha256=args.attacker_sha256,
        defender_sha256=args.defender_sha256,
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_candidate_multiplier=args.max_candidate_multiplier,
        candidate_wave_pairs=args.candidate_wave_pairs,
        generation_batch_size=args.generation_batch_size,
        judge_batch_size=args.judge_batch_size,
        max_new_tokens=args.max_new_tokens,
        adaptive_stratum_sampling=args.adaptive_stratum_sampling,
    )
    print(
        f"CELL_DONE {args.attacker_label}__{args.defender_label} "
        f"value={summary['value']:.6f}",
        flush=True,
    )
    print(json.dumps({"value": summary["value"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
