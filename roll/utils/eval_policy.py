"""Evaluate a role as a single checkpoint or as the Nash mixture of a stage.

A PSRO stage does not have "the" defender -- it has a mixed strategy over the
defenders trained so far.  Reporting only a single checkpoint answers a question
the algorithm never asked, and picking the highest-weight component is not the
same thing either: at generation 8 the defender mixture is D7 0.574 / D6 0.392 /
D2 0.034, and no single member of that support behaves like the mixture.

Two evaluation modes, both supported here:

``expectation`` (default)
    A metric of a mixed strategy is the mixture-weighted mean of the component
    metrics: ``E_{d~q}[m(d)] = sum_d q_d * m(d)``.  This is exact -- no sampling
    noise at all -- and it reuses per-checkpoint results that already exist, so
    a mixture usually costs zero new GPU time.  Use it for any scalar metric
    (ASR, compliance, StrongREJECT score).

``sample``
    Draw one component per evaluation item, deterministically from the item
    index, and let that component generate.  Needed when the artifact you want
    is *text* from the mixture rather than a number -- external attacker
    transfer, qualitative reads, anything judged per-response.  The draw is
    reproducible and identical across arms given the same item count and seed,
    so paired comparisons stay paired.

Everything here is pure: no torch, no vLLM, no network.  The serving side reads
``assign_components`` and routes LoRA requests accordingly.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

#: ``nash:<role>`` or ``nash:<role>@<generation>``; anything else is a checkpoint.
_NASH_RE = re.compile(r"^nash:(attacker|defender)(?:@(\d+))?$", re.IGNORECASE)

SUPPORT_EPSILON = 1e-6


class PolicySpecError(ValueError):
    """Raised for a spec that cannot be resolved against a run's state."""


def parse_policy_spec(spec: str) -> dict[str, Any]:
    """Parse an evaluation target.

    ``"D5"``            -> a population label
    ``"/path/to/ckpt"`` -> an explicit adapter directory
    ``"nash:defender"`` -> the defender mixture at the latest solved generation
    ``"nash:attacker@5"``-> the attacker mixture at generation 5
    """
    text = str(spec).strip()
    if not text:
        raise PolicySpecError("empty policy spec")
    match = _NASH_RE.match(text)
    if match:
        role = match.group(1).lower()
        generation = int(match.group(2)) if match.group(2) else None
        return {"kind": "nash", "role": role, "generation": generation}
    if "/" in text or text.startswith("."):
        return {"kind": "checkpoint", "path": text, "label": text.rstrip("/").split("/")[-1]}
    return {"kind": "checkpoint", "label": text, "path": None}


def _solve_zero_sum(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float], float]:
    """Minimax LP for a zero-sum matrix given as attacker payoffs."""
    import numpy as np
    from scipy.optimize import linprog

    payoff = np.asarray(matrix, dtype=float)
    rows, cols = payoff.shape
    shift = payoff.min() - 1.0
    shifted = payoff - shift

    cost = np.zeros(rows + 1)
    cost[-1] = -1.0
    row_solution = linprog(
        cost,
        A_ub=np.hstack([-shifted.T, np.ones((cols, 1))]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([np.ones((1, rows)), np.zeros((1, 1))]),
        b_eq=[1.0],
        bounds=[(0, None)] * rows + [(None, None)],
        method="highs",
    )
    cost2 = np.zeros(cols + 1)
    cost2[-1] = 1.0
    col_solution = linprog(
        cost2,
        A_ub=np.hstack([shifted, -np.ones((rows, 1))]),
        b_ub=np.zeros(rows),
        A_eq=np.hstack([np.ones((1, cols)), np.zeros((1, 1))]),
        b_eq=[1.0],
        bounds=[(0, None)] * cols + [(None, None)],
        method="highs",
    )
    if not (row_solution.success and col_solution.success):
        raise PolicySpecError("meta-game LP did not solve")
    return (
        [float(x) for x in row_solution.x[:rows]],
        [float(x) for x in col_solution.x[:cols]],
        float(row_solution.x[-1] + shift),
    )


def nash_mixture(
    state: Mapping[str, Any],
    role: str,
    generation: int | None = None,
) -> list[dict[str, Any]]:
    """Return ``[{label, path, probability}, ...]`` over the support only.

    The mixture is solved from the leading ``(generation+1) x (generation+1)``
    submatrix of the recorded cells, i.e. exactly the game the algorithm faced
    when it trained that stage's best response. Components with negligible
    weight are dropped so a caller never spends GPU time on a 1e-15 entry.
    """
    role = role.lower()
    if role not in {"attacker", "defender"}:
        raise PolicySpecError(f"role must be attacker or defender, got {role!r}")

    population = state.get("population", {})
    cells = state.get("cells", {})
    attackers = sorted(
        (label for label in population if label.startswith("A")),
        key=lambda label: int(label[1:]),
    )
    defenders = sorted(
        (label for label in population if label.startswith("D")),
        key=lambda label: int(label[1:]),
    )
    if generation is None:
        generation = min(len(attackers), len(defenders)) - 1
    attackers = attackers[: generation + 1]
    defenders = defenders[: generation + 1]
    if not attackers or not defenders:
        raise PolicySpecError(f"no population for generation {generation}")

    matrix = []
    for attacker in attackers:
        row = []
        for defender in defenders:
            cell = cells.get(f"{attacker}__{defender}")
            if cell is None:
                raise PolicySpecError(
                    f"cell {attacker}__{defender} is missing; the mixture at "
                    f"generation {generation} is not yet defined"
                )
            row.append(float(cell["value"]))
        matrix.append(row)

    attacker_weights, defender_weights, _ = _solve_zero_sum(matrix)
    labels = attackers if role == "attacker" else defenders
    weights = attacker_weights if role == "attacker" else defender_weights

    support = [
        {
            "label": label,
            "path": (population[label] or {}).get("path"),
            "probability": weight,
        }
        for label, weight in zip(labels, weights)
        if weight > SUPPORT_EPSILON
    ]
    if not support:
        raise PolicySpecError(f"{role} mixture at generation {generation} is empty")
    total = sum(entry["probability"] for entry in support)
    for entry in support:                      # renormalize after dropping dust
        entry["probability"] /= total
    return support


def resolve(spec: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a spec to ``{kind, components: [{label, path, probability}]}``."""
    parsed = parse_policy_spec(spec)
    if parsed["kind"] == "checkpoint":
        path = parsed["path"]
        label = parsed["label"]
        if path is None:
            record = state.get("population", {}).get(label)
            if record is None:
                raise PolicySpecError(f"{label!r} is not in the population")
            path = record.get("path")
        return {
            "kind": "checkpoint",
            "components": [{"label": label, "path": path, "probability": 1.0}],
        }
    support = nash_mixture(state, parsed["role"], parsed["generation"])
    return {
        "kind": "nash",
        "role": parsed["role"],
        "generation": parsed["generation"],
        "components": support,
    }


def combine_metric(
    per_component: Mapping[str, float],
    components: Sequence[Mapping[str, Any]],
) -> float:
    """Mixture-weighted mean of a scalar metric -- the exact expected value."""
    missing = [c["label"] for c in components if c["label"] not in per_component]
    if missing:
        raise PolicySpecError(
            f"missing component metrics for {missing}; evaluate them first"
        )
    return sum(
        float(c["probability"]) * float(per_component[c["label"]]) for c in components
    )


def assign_components(
    n_items: int,
    components: Sequence[Mapping[str, Any]],
    *,
    seed: str = "psro-eval",
) -> list[int]:
    """Deterministically map each item index to a component index.

    Hash-based rather than RNG-based so the assignment depends only on
    ``(seed, item index)`` -- it does not drift if the caller evaluates a subset,
    reorders work, or parallelises, and two arms with the same seed and item
    count get byte-identical assignments.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    bounds, cumulative = [], 0.0
    for component in components:
        cumulative += float(component["probability"])
        bounds.append(cumulative)
    if not bounds or abs(bounds[-1] - 1.0) > 1e-6:
        raise PolicySpecError(f"component probabilities must sum to 1, got {cumulative}")

    out = []
    for index in range(n_items):
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        # 53 bits -> a uniform double in [0, 1)
        draw = int.from_bytes(digest[:7], "big") / float(1 << 56)
        for position, bound in enumerate(bounds):
            if draw < bound:
                out.append(position)
                break
        else:
            out.append(len(bounds) - 1)
    return out
