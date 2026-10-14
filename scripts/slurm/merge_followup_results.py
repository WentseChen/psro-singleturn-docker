#!/usr/bin/env python3
"""Append standalone follow-up oracle evaluations to the README results table.

merge_eval_results.py keys off a run's state.json population and sorts
defenders numerically, so it cannot express follow-up runs like "D4_retrain"
or "D5 at 300 steps". This adds those rows using the identical format and the
same additive, idempotent label matching: a row is replaced in place when its
label already exists, and historical rows are never displaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
TABLE_HEADER_PREFIX = "| Model | Source | WG adv ASR"
COLUMNS = (
    "WG adv ASR", "WG vanilla ASR", "WJB harmful ASR", "DAN ASR",
    "HarmBench adv ASR", "OR-Bench RTA", "XSTest RTA",
    "WJB benign comply", "XSTest benign comply", "OR-Bench benign comply",
)


def _format(value) -> str:
    if value is None or isinstance(value, str):
        return "-"
    return f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("targets", nargs="+",
                        help="SLUG=<README row label> pairs")
    args = parser.parse_args()

    rows = []
    for target in args.targets:
        slug, _, label = target.partition("=")
        path = args.eval_output_root / slug / "comparison.json"
        if not path.is_file():
            raise SystemExit(f"missing comparison.json for {slug}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        measured = payload.get("measured") or {}
        metrics = measured.get(slug) or next(iter(measured.values()), {})
        cells = " | ".join(_format(metrics.get(c)) for c in COLUMNS)
        rows.append((label, f"| {label} | Our evaluation | {cells} |"))

    lines = README.read_text(encoding="utf-8").splitlines()
    try:
        header = next(i for i, line in enumerate(lines)
                      if line.startswith(TABLE_HEADER_PREFIX))
    except StopIteration:
        raise SystemExit("defender results table not found in README.md")

    end = header + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    body = lines[header + 2:end]
    existing = {line.split("|")[1].strip(): i for i, line in enumerate(body)}
    for label, row in rows:
        if label in existing:
            body[existing[label]] = row
        else:
            body.append(row)

    if args.dry_run:
        for _, row in rows:
            print(row)
        return 0
    README.write_text("\n".join(lines[:header + 2] + body + lines[end:]) + "\n",
                      encoding="utf-8")
    print(f"README.md updated with {len(rows)} follow-up row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
