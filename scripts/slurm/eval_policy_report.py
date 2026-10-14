"""Report a benchmark metric for a checkpoint or for a stage's Nash mixture.

A mixed strategy's expected metric is the mixture-weighted mean of its
components' metrics, so a mixture usually costs no new GPU time: evaluate each
support member once -- most already are -- and combine.

    # what the generation-8 defender mixture scores on SR12 and SORRY-Bench
    python scripts/slurm/eval_policy_report.py \\
        --suffix cold_psro_v6d_s300_race_20260909 --policy nash:defender

    # a single checkpoint, same report shape
    python scripts/slurm/eval_policy_report.py --suffix ... --policy D7

    # what would I have to evaluate to answer this?
    python scripts/slurm/eval_policy_report.py --suffix ... --policy nash:attacker --plan

``--plan`` prints the components and exits without requiring their metrics, so
you can queue exactly the missing evaluations and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from roll.utils.eval_policy import (  # noqa: E402
    PolicySpecError,
    combine_metric,
    resolve,
)

PSRO_ROOT = pathlib.Path(
    os.environ.get(
        "PSRO_ROOT",
        "/zfsauton/scratch/wentsec/coldpsro_slurm/psro/role_lora_zero_sum_psro",
    )
)
HARD_BENCH = pathlib.Path(
    os.environ.get("HARD_BENCH", "/zfsauton/scratch/wentsec/coldpsro_slurm/hard_bench")
)
STD_EVAL = pathlib.Path(
    os.environ.get(
        "STD_EVAL",
        "/zfsauton/scratch/wentsec/coldpsro_slurm/psro"
        "/upstream_selfredteam_role_full_eval",
    )
)

STANDARD_METRICS = [
    "WG adv ASR", "WG vanilla ASR", "WJB harmful ASR", "DAN ASR",
    "HarmBench adv ASR", "OR-Bench RTA", "XSTest RTA",
    "WJB benign comply", "XSTest benign comply", "OR-Bench benign comply",
]


def _arm_name(prefix: str, label: str) -> str:
    return f"{prefix}_{label}"


def sr12(arm: str) -> dict[str, float] | None:
    path = HARD_BENCH / "out_jb" / f"{arm}.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    scores = {
        key: value["score"]
        for key, value in raw.items()
        if key != "_summary" and isinstance(value, dict) and "score" in value
    }
    wrapped = [v for k, v in scores.items() if k != "none"]
    return {
        "SR12 mean": statistics.mean(wrapped),
        "SR12 worst": max(wrapped),
        "SR12 no-wrapper": scores.get("none"),
    }


def sorry_bench(arm: str) -> dict[str, float] | None:
    path = HARD_BENCH / "out_sb" / f"{arm}.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    labels = [float(r["label"]) for r in rows if r.get("label") not in (None, "")]
    return {"SORRY-Bench": sum(labels) / len(labels)} if labels else None


def standard(suffix: str, label: str, prefix: str) -> dict[str, float] | None:
    """Locate a standard-suite result for one component.

    Output directories were created by several launchers over the life of the
    run and do not share one naming convention -- ``std_v6d_D2``,
    ``cold_psro_<suffix>_D6``, ``v6d_eval_only_D5s240_D5s240``.  Match on the
    directory *ending* in the label instead of guessing a prefix, preferring a
    name that also mentions this run's suffix so two runs cannot collide.
    """
    if not STD_EVAL.is_dir():
        return None
    candidates = [
        d for d in STD_EVAL.iterdir()
        if d.is_dir() and (d.name == label or d.name.endswith(f"_{label}"))
        and (d / "comparison.json").is_file()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: (suffix not in d.name, prefix not in d.name, d.name))
    measured = list(
        json.loads((candidates[0] / "comparison.json").read_text())["measured"].values()
    )[0]
    return {k: measured[k] for k in STANDARD_METRICS if k in measured}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        help="a population label (D7), an adapter path, or nash:<role>[@gen]",
    )
    parser.add_argument(
        "--arm-prefix",
        default="v6d",
        help="prefix used for this run's arms in hard_bench/adapters_eval.json",
    )
    parser.add_argument("--plan", action="store_true",
                        help="print the components and exit")
    args = parser.parse_args()

    state_path = PSRO_ROOT / args.suffix / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"no run at {state_path}")
    state = json.loads(state_path.read_text())

    try:
        resolved = resolve(args.policy, state)
    except PolicySpecError as exc:
        raise SystemExit(f"cannot resolve {args.policy!r}: {exc}")

    components = resolved["components"]
    header = args.policy
    if resolved["kind"] == "nash":
        generation = resolved["generation"]
        header += f"  (generation {'latest' if generation is None else generation})"
    print(f"policy: {header}")
    for component in components:
        source = component["path"] or "BASE (no adapter)"
        print(
            f"  {component['label']:<6} p={component['probability']:.4f}  "
            f"{pathlib.Path(source).name if component['path'] else source}"
        )

    if args.plan:
        print("\nevaluate these arms, then re-run without --plan:")
        for component in components:
            print(f"  {_arm_name(args.arm_prefix, component['label'])}")
        return 0

    collected: dict[str, dict[str, float]] = {}
    for component in components:
        arm = _arm_name(args.arm_prefix, component["label"])
        merged: dict[str, float] = {}
        for source in (
            sr12(arm),
            sorry_bench(arm),
            standard(args.suffix, component["label"], args.arm_prefix.rstrip("_")),
        ):
            if source:
                merged.update({k: v for k, v in source.items() if v is not None})
        collected[component["label"]] = merged

    metrics = sorted({k for m in collected.values() for k in m})
    if not metrics:
        raise SystemExit(
            "no component metrics found; run --plan and evaluate those arms first"
        )

    print(f"\n{'metric':<24}" + "".join(f"{c['label']:>10}" for c in components)
          + f"{'MIXTURE':>12}")
    for metric in metrics:
        per_component = {
            label: values[metric]
            for label, values in collected.items()
            if metric in values
        }
        cells = "".join(
            f"{per_component.get(c['label'], float('nan')):>10.4f}" for c in components
        )
        try:
            mixed = combine_metric(per_component, components)
            print(f"{metric:<24}{cells}{mixed:>12.4f}")
        except PolicySpecError:
            print(f"{metric:<24}{cells}{'incomplete':>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
