#!/usr/bin/env python3
"""Merge one finished PSRO evaluation into the README's defender results table.

Reads the per-defender ``comparison.json`` files the official evaluator wrote,
emits a standalone summary next to the run's own artifacts, and appends rows to
the existing table in README.md using its established row format
(``SafeSelfPlay D<i>, step <n> (<variant>)`` / ``Our evaluation``, three
decimals, ``-`` for a column the run did not measure).

The README edit is strictly additive and idempotent: rows are matched on their
exact label, so re-running replaces this run's own rows in place and never
touches a historical row. Cold and hot runs use different variant tags and
therefore never collide.

    python scripts/slurm/merge_eval_results.py \\
        --psro-root ... --suffix ... --eval-output-root ... \\
        --label-prefix cold_psro --variant "cold PSRO 20260827"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
TABLE_HEADER_PREFIX = "| Model | Source | WG adv ASR"
COLUMNS = (
    "WG adv ASR",
    "WG vanilla ASR",
    "WJB harmful ASR",
    "DAN ASR",
    "HarmBench adv ASR",
    "OR-Bench RTA",
    "XSTest RTA",
    "WJB benign comply",
    "XSTest benign comply",
    "OR-Bench benign comply",
)


def _format(value: Any) -> str:
    if value is None or isinstance(value, str):
        return "-"
    return f"{float(value):.3f}"


def _row(label: str, metrics: dict[str, Any]) -> str:
    cells = " | ".join(_format(metrics.get(column)) for column in COLUMNS)
    return f"| {label} | Our evaluation | {cells} |"


def collect(
    eval_root: Path, suffix: str, label_prefix: str, defenders: list[str]
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for defender in defenders:
        path = eval_root / f"{label_prefix}_{suffix}_{defender}" / "comparison.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        measured = payload.get("measured") or {}
        # The evaluator keys `measured` by trained_label, and evaluate_base=False
        # means there is exactly one entry.
        key = f"{label_prefix}_{defender}"
        metrics = measured.get(key) or next(iter(measured.values()), {})
        found[defender] = {"metrics": metrics, "comparison": str(path)}
    return found


def write_summary(
    run_root: Path, suffix: str, variant: str,
    results: dict[str, dict[str, Any]], steps: int,
) -> Path:
    summary_md = run_root / "results_summary.md"
    lines = [
        f"# Defender evaluation: {variant}",
        "",
        f"Run suffix: `{suffix}`  ",
        f"Run root: `{run_root}`  ",
        "Evaluator: released `safety-eval-fork`, `llama3_cot` template, "
        "`evaluate_base=False` (baseline columns come from the previous "
        "released-evaluator run).",
        "",
        "| Model | Source | " + " | ".join(f"{c} {'↓' if c.endswith('ASR') else '↑'}" for c in COLUMNS) + " |",
        "|---|---|" + "---:|" * len(COLUMNS),
    ]
    for defender in sorted(results, key=lambda d: int(d[1:])):
        label = f"SafeSelfPlay {defender}, step {steps} ({variant})"
        lines.append(_row(label, results[defender]["metrics"]))
    lines += ["", "## Result manifests", ""]
    for defender in sorted(results, key=lambda d: int(d[1:])):
        lines.append(f"- `{defender}`: `{results[defender]['comparison']}`")
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    (run_root / "results_summary.json").write_text(
        json.dumps(
            {
                "run_suffix": suffix,
                "variant": variant,
                "steps_per_role": steps,
                "columns": list(COLUMNS),
                "results": results,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return summary_md


def update_readme(
    rows: list[tuple[str, str]], summary_md: Path, variant: str, dry_run: bool
) -> None:
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        header = next(
            index for index, line in enumerate(lines)
            if line.startswith(TABLE_HEADER_PREFIX)
        )
    except StopIteration:
        raise SystemExit("defender results table not found in README.md")

    end = header + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1

    body = lines[header + 2 : end]
    existing = {line.split("|")[1].strip(): index for index, line in enumerate(body)}
    for label, row in rows:
        if label in existing:
            body[existing[label]] = row      # refresh this run's own row
        else:
            body.append(row)                 # never displace a historical row

    note = (
        f"The `{variant}` rows come from the sequential Slurm PSRO chain; their "
        f"per-defender result manifests and a standalone copy of this block are "
        f"recorded in `{summary_md}`."
    )
    tail = lines[end:]
    if not any(note.split(";")[0] in line for line in tail):
        tail = ["", note] + tail

    updated = "\n".join(lines[: header + 2] + body + tail) + "\n"
    if dry_run:
        print("--- rows ---")
        for _, row in rows:
            print(row)
        print("--- note ---")
        print(note)
        return
    README.write_text(updated, encoding="utf-8")
    print(f"README.md updated with {len(rows)} row(s) for {variant}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--eval-output-root", required=True, type=Path)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = args.psro_root / args.suffix
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    steps = int(state["contract"]["steps_per_role"])
    generations = int(state["contract"]["generations"])
    defenders = [f"D{index}" for index in range(1, generations + 1)]

    results = collect(
        args.eval_output_root, args.suffix, args.label_prefix, defenders
    )
    if not results:
        raise SystemExit(
            f"no comparison.json found under {args.eval_output_root} "
            f"for {args.label_prefix}_{args.suffix}_D*"
        )
    missing = [d for d in defenders if d not in results]
    if missing:
        print(f"warning: no results yet for {missing}; merging what exists")

    summary_md = write_summary(run_root, args.suffix, args.variant, results, steps)
    print(f"wrote {summary_md}")
    rows = [
        (
            f"SafeSelfPlay {defender}, step {steps} ({args.variant})",
            _row(
                f"SafeSelfPlay {defender}, step {steps} ({args.variant})",
                results[defender]["metrics"],
            ),
        )
        for defender in sorted(results, key=lambda d: int(d[1:]))
    ]
    update_readme(rows, summary_md, args.variant, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
