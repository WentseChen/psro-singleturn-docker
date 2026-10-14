"""Retention-aware candidate allocation for stratified payoff cells.

The payoff/cell generator draws harmful and benign candidates in strict 1:1
lockstep and keeps drawing until *both* strata hold ``episodes / 2`` valid
games.  Retention is not symmetric: a benign seed the attacker rewrites into a
harmful prompt is dropped (``benign_to_harmful_label_drift``) while harmful
seeds are retained essentially always.  Lockstep therefore over-generates the
harmful stratum by exactly the benign drop count -- work that is judged, paid
for, and then discarded -- and it can exhaust the candidate cap while the
harmful stratum has long been complete.

This module keeps the *accepted* set exactly 50/50 and changes only how many
candidates each stratum is asked to produce per wave.  Allocation is driven by
a Beta posterior lower confidence bound on each stratum's retention rate, so
an unmeasured stratum is never starved and a collapsing stratum is escalated
before the cap is reached.
"""

from __future__ import annotations

import math
from typing import Mapping

STRATA = ("harmful", "benign")

# Retention below this is treated as this value when sizing a wave, so a
# stratum that has retained nothing so far still gets a bounded-size probe
# instead of an allocation that diverges to the whole remaining budget.
MIN_PLANNING_RETENTION = 0.05


def _beta_lower_bound(
    retained: int,
    judged: int,
    *,
    prior_alpha: float,
    prior_beta: float,
    z: float,
) -> float:
    """Return a Wilson-style lower bound on a Beta posterior mean.

    A closed form keeps this dependency-free and deterministic.  With no
    observations it returns the prior mean, which is deliberately optimistic
    only up to the ``MIN_PLANNING_RETENTION`` floor applied by the caller.
    """

    if judged < 0 or retained < 0 or retained > judged:
        raise ValueError(f"invalid retention counts: {retained}/{judged}")
    alpha = retained + prior_alpha
    beta = (judged - retained) + prior_beta
    total = alpha + beta
    mean = alpha / total
    # Posterior variance of a Beta(alpha, beta).
    variance = (alpha * beta) / (total * total * (total + 1.0))
    return max(0.0, mean - z * math.sqrt(variance))


def stratum_retention_estimate(
    retained: int,
    judged: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    z: float = 1.28,
) -> float:
    """Conservative retention estimate for one stratum."""

    return _beta_lower_bound(
        retained, judged, prior_alpha=prior_alpha, prior_beta=prior_beta, z=z
    )


def plan_candidate_wave(
    *,
    deficits: Mapping[str, int],
    judged: Mapping[str, int],
    retained: Mapping[str, int],
    wave_budget: int,
    remaining_budget: int,
    min_retention: float = MIN_PLANNING_RETENTION,
    z: float = 1.28,
) -> dict[str, int]:
    """Split one wave's candidate budget across strata by observed retention.

    ``deficits`` are the still-missing valid games per stratum.  A stratum with
    no deficit receives nothing -- that omission is the entire saving relative
    to lockstep.  The returned counts sum to at most
    ``min(wave_budget, remaining_budget)``.
    """

    if wave_budget < 0 or remaining_budget < 0:
        raise ValueError("budgets must be non-negative")
    if not 0.0 < min_retention <= 1.0:
        raise ValueError("min_retention must be within (0, 1]")

    unknown = set(deficits) - set(STRATA)
    if unknown:
        raise ValueError(f"unknown strata: {sorted(unknown)}")

    budget = min(wave_budget, remaining_budget)
    if budget <= 0:
        return {label: 0 for label in STRATA}

    # Expected candidates required to close each deficit at current retention.
    need: dict[str, int] = {}
    for label in STRATA:
        deficit = int(deficits.get(label, 0))
        if deficit <= 0:
            need[label] = 0
            continue
        rate = max(
            stratum_retention_estimate(
                int(retained.get(label, 0)), int(judged.get(label, 0)), z=z
            ),
            min_retention,
        )
        need[label] = max(1, math.ceil(deficit / rate))

    total_need = sum(need.values())
    if total_need == 0:
        return {label: 0 for label in STRATA}
    if total_need <= budget:
        return dict(need)

    # Scale down proportionally, guaranteeing one candidate to every stratum
    # that still has a deficit so no stratum can stall.
    deficient = [label for label in STRATA if need[label] > 0]
    if budget < len(deficient):
        # Not enough budget to probe every deficient stratum; give it to the
        # stratum that is furthest from done.
        worst = max(deficient, key=lambda label: need[label])
        return {label: (budget if label == worst else 0) for label in STRATA}

    allocation = {label: 0 for label in STRATA}
    for label in deficient:
        allocation[label] = 1
    left = budget - len(deficient)
    if left > 0:
        weights = {label: need[label] / total_need for label in deficient}
        for label in deficient:
            allocation[label] += int(left * weights[label])
        # Hand any rounding remainder to the neediest stratum.
        remainder = budget - sum(allocation.values())
        if remainder > 0:
            allocation[max(deficient, key=lambda l: need[l])] += remainder
    return allocation


def projected_candidate_totals(
    *,
    required_per_stratum: int,
    retention: Mapping[str, float],
) -> dict[str, float]:
    """Candidates each policy is expected to generate, for cost comparison."""

    adaptive = {
        label: required_per_stratum / max(retention[label], 1e-9)
        for label in STRATA
    }
    # Lockstep must draw pairs until the slowest stratum is satisfied.
    lockstep_pairs = max(adaptive.values())
    return {
        "adaptive_total": sum(adaptive.values()),
        "lockstep_total": 2.0 * lockstep_pairs,
        "adaptive_by_stratum": adaptive,
        "saved": 2.0 * lockstep_pairs - sum(adaptive.values()),
    }
