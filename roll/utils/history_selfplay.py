"""Schedules for uniform-history and latest-opponent self-play baselines."""

from __future__ import annotations

from typing import Any


def history_schedule(method: str, rounds: int) -> list[dict[str, Any]]:
    if method not in {"uniform", "naive"}:
        raise ValueError("method must be uniform or naive")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    phases = []
    for generation in range(1, rounds + 1):
        for role, own, opponent, latest in (
            ("attacker", "A", "D", generation - 1),
            ("defender", "D", "A", generation),
        ):
            labels = (
                [f"{opponent}{i}" for i in range(latest + 1)]
                if method == "uniform" else [f"{opponent}{latest}"]
            )
            phases.append({
                "target": f"{own}{generation}", "role": role,
                "initialized_from": f"{own}{0 if method == 'uniform' else generation - 1}",
                "opponents": labels,
                "probabilities": [1.0 / len(labels)] * len(labels),
            })
    return phases
