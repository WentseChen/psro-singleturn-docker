#!/usr/bin/env python3
"""Seed a new local cold-PSRO run root from the archived 3x3 artifacts.

The archived run lived on a Modal volume that no longer exists, so every stored
path points at ``/output/...``.  The coordinator re-verifies those paths on
every call (``_verify_population_identity``) and re-derives the ``before_*``
matrix snapshots from ``state["cells"][*]["manifest_path"]``, so the copy is
only usable once both are rewritten in lockstep.

Two further edits are required and are recorded in ``seed_provenance.json``:

* ``contract.implementation_sha256`` records the current implementation for
  provenance, while the archived values remain in ``seed_provenance.json``.
  Source hashes no longer gate resume; parameter and artifact checks still do.
* ``oracle_specs["A3"]`` / ``matrices/before_A3.json`` and the partial
  ``training/A3`` directory are dropped, so A3 is trained fresh here rather than
  resumed.  Both are re-derived byte-identically from the nine seeded cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402

ARCHIVED_PSRO_ROOT = "/output/role_lora_zero_sum_psro"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rewrite(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_rewrite(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite(item, old, new) for key, item in value.items()}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed_dir: Path = args.seed_dir.resolve()
    destination: Path = (args.psro_root.resolve() / args.suffix)
    old_root = f"{ARCHIVED_PSRO_ROOT}/{seed_dir.name}"
    new_root = str(destination)

    if destination.exists():
        if not args.force:
            print(f"already seeded: {destination}")
            return 0
        shutil.rmtree(destination)

    print(f"seeding {destination} from {seed_dir}", flush=True)
    shutil.copytree(
        seed_dir,
        destination,
        ignore=shutil.ignore_patterns("SHA256SUMS", "VERIFIED_CHECKPOINT_SHA256.txt"),
    )

    dropped: list[str] = []
    partial_a3 = destination / "training" / "A3"
    if partial_a3.is_dir():
        shutil.rmtree(partial_a3)
        dropped.append("training/A3 (archived partial run, not resumed)")
    stale_snapshot = destination / "matrices" / "before_A3.json"
    if stale_snapshot.is_file():
        stale_snapshot.unlink()
        dropped.append("matrices/before_A3.json (re-derived from the 9 cells)")

    for path in sorted(destination.rglob("*.json")):
        _dump(path, _rewrite(_load(path), old_root, new_root))

    state = _load(destination / "state.json")
    archived_hashes = dict(state["contract"]["implementation_sha256"])
    psro, payoff = common.load_modules()
    head_hashes = common.implementation_hashes(psro)
    state["contract"]["implementation_sha256"] = head_hashes
    if state["oracle_specs"].pop("A3", None) is not None:
        dropped.append("oracle_specs.A3 (re-derived from the 9 cells)")
    state["matrix_snapshots"].pop("before_A3", None)
    state["completed"] = False
    state["stage"] = f"seeded_from_{seed_dir.name}"
    _dump(destination / "state.json", state)

    # Fail here rather than inside the driver: every check below is one the
    # coordinator repeats on its first call.
    common.retarget(
        psro,
        payoff,
        psro_root=destination.parent,
        raw_payoff_root=destination.parent.parent / "raw_payoff_v2",
        upstream_source=Path("/nonexistent"),
        judge_url="http://127.0.0.1:0/classify",
    )
    for label, record in state["population"].items():
        if label in {"A0", "D0"}:
            if record != psro._base_population_identity(label):
                raise SystemExit(f"base identity mismatch: {label}")
            continue
        psro._verify_population_identity(
            label, record, lora_rank=64, lora_alpha=64
        )
        print(f"verified {label}: {record['path']}", flush=True)

    from roll.utils.role_lora_cold_psro import next_action

    action = next_action(
        state["population"], state["cells"], state["evaluations"],
        generations=state["contract"]["generations"],
    )
    if action != {"kind": "oracle", "role": "attacker", "target": "A3"}:
        raise SystemExit(f"unexpected first action after seeding: {action}")

    provenance = {
        "schema_version": "cold-psro-slurm-seed-v1",
        "source_archive": str(seed_dir),
        "source_suffix": seed_dir.name,
        "run_root": new_root,
        "run_suffix": args.suffix,
        "path_rewrite": {"from": old_root, "to": new_root},
        "archived_implementation_sha256": archived_hashes,
        "head_implementation_sha256": head_hashes,
        "implementation_delta": sorted(
            key for key in head_hashes
            if head_hashes[key] != archived_hashes.get(key)
        ),
        "dropped": dropped,
        "seeded_population": sorted(state["population"]),
        "seeded_cells": sorted(state["cells"]),
        "first_action": action,
        "state_sha256": _sha256(destination / "state.json"),
    }
    _dump(destination / "seed_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
