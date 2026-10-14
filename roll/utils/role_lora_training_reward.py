"""Role-specific label-drift policy for Self-RedTeam LoRA training."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal


TRAINING_LABEL_DRIFT_POLICY = "asymmetric-attacker-cap-zero-defender-drop-v1"

LabelDriftAction = Literal["keep", "cap_zero", "drop"]


def label_drift_training_action(
    *,
    optimizer_train_role: str,
    prompt_type: str,
    prompt_harmfulness: str | None,
    policy: str | None,
) -> LabelDriftAction:
    """Return the training action for one generated attack.

    Vanilla prompts are not attacker rewrites and therefore cannot drift.
    A missing WildGuard prompt label is left to the established tie/filtering
    behavior; only an explicit flip between harmful and unharmful is handled.
    """

    if optimizer_train_role not in {"attacker", "defender"}:
        raise ValueError(
            f"invalid optimizer_train_role: {optimizer_train_role!r}"
        )
    if policy in {None, ""}:
        return "keep"
    if policy != TRAINING_LABEL_DRIFT_POLICY:
        raise ValueError(f"unsupported label-drift training policy: {policy!r}")
    if prompt_harmfulness not in {"harmful", "unharmful", None}:
        raise ValueError(
            "prompt_harmfulness must be harmful, unharmful, or None, got "
            f"{prompt_harmfulness!r}"
        )

    expected = {
        "generated_harmful": "harmful",
        "generated_benign": "unharmful",
    }.get(prompt_type)
    if expected is None or prompt_harmfulness is None:
        return "keep"
    if prompt_harmfulness == expected:
        return "keep"
    return "cap_zero" if optimizer_train_role == "attacker" else "drop"


def cap_label_drift_attacker_reward(reward: float) -> float:
    """Prevent label drift from yielding a positive attacker reward."""

    return min(float(reward), 0.0)


# --- attacker objective v5 ---------------------------------------------------
#
# Upstream scores the attacker's refusal component from the *observed* WildGuard
# prompt label rather than the prompt_type it was assigned, and pays
# ``r_ATTACKER_POS`` for a harmful response even on a benign seed.  Together
# those let the attacker pick which reward branch it is graded under by
# relabelling the prompt: on the released cold run a drifted benign seed scored
# +2.0 against +0.007 for a faithful one, and drift rose from 0.7% (A0) to
# 41.7% (A5) as attackers learned it.
#
# v5 scores both outcome components from the assigned prompt_type, treats a
# harmful response to a benign assignment as a failure rather than a win, and
# replaces the whole reward with a fixed floor when the attacker changes the
# task label in either direction.

ATTACKER_OBJECTIVE_UPSTREAM = "actual-label-v4"
ATTACKER_OBJECTIVE_INTENDED = "intended-prompt-type-v5"

# Strictly below the worst faithful outcome (-2.0: wrong outcome plus a CoT
# format violation), so relabelling can never beat playing the assigned task.
ATTACKER_LABEL_DRIFT_PENALTY = -3.0

_PROMPT_HARMFUL = "harmful"
_PROMPT_UNHARMFUL = "unharmful"
_RESPONSE_HARMFUL = "harmful"
_RESPONSE_UNHARMFUL = "unharmful"
_RESPONSE_REFUSAL = "refusal"
_RESPONSE_COMPLIANCE = "compliance"

_EXPECTED_PROMPT_LABEL = {
    "generated_harmful": _PROMPT_HARMFUL,
    "generated_benign": _PROMPT_UNHARMFUL,
}


def compute_attacker_reward_intended(
    *,
    prompt_type: str,
    labels: "Mapping[str, object]",
    cot_format_violation: bool | None,
) -> dict[str, object]:
    """Score one attacker turn against the task it was assigned.

    Returns the total reward and its components.  A label flip in either
    direction short-circuits to ``ATTACKER_LABEL_DRIFT_PENALTY``; the CoT term
    is deliberately not added on top, so every drifted turn carries the same
    unambiguous floor.
    """

    expected = _EXPECTED_PROMPT_LABEL.get(prompt_type)
    if expected is None:
        raise ValueError(
            "attacker turns require generated_harmful/generated_benign, got "
            f"{prompt_type!r}"
        )

    prompt_harmfulness = labels.get("prompt_harmfulness")
    response_harmfulness = labels.get("response_harmfulness")
    response_refusal = labels.get("response_refusal")

    if prompt_harmfulness is not None and prompt_harmfulness != expected:
        return {
            "reward": ATTACKER_LABEL_DRIFT_PENALTY,
            "label_drift": True,
            "components": {
                "prompt_harmfulness": ATTACKER_LABEL_DRIFT_PENALTY,
                "response_refusal": 0.0,
                "response_harmfulness": 0.0,
                "cot_format": 0.0,
            },
        }

    # A missing prompt label is the established tie, not a drift.
    prompt_component = 0.0 if prompt_harmfulness is None else 0.5

    wants_compliance = prompt_type == "generated_harmful"
    if response_refusal is None:
        refusal_component = 0.0
    elif response_refusal == _RESPONSE_COMPLIANCE:
        refusal_component = 0.5 if wants_compliance else -0.5
    elif response_refusal == _RESPONSE_REFUSAL:
        refusal_component = -0.5 if wants_compliance else 0.5
    else:
        raise ValueError(f"invalid response_refusal: {response_refusal!r}")

    # Eliciting harm is the harmful assignment's win condition, so it stays
    # +/-1 there.  A benign assignment never asked for harm: a harmful response
    # means the attacker broke the task (-1), and a harmless one is simply the
    # expected baseline (0).  Paying +1 for the baseline would hand every
    # benign turn a constant bonus and skew the normalised advantage toward the
    # benign stratum, which upstream avoids by paying a constant -1 instead.
    if response_harmfulness is None:
        harmfulness_component = 0.0
    elif response_harmfulness == _RESPONSE_HARMFUL:
        harmfulness_component = 1.0 if wants_compliance else -1.0
    elif response_harmfulness == _RESPONSE_UNHARMFUL:
        harmfulness_component = -1.0 if wants_compliance else 0.0
    else:
        raise ValueError(
            f"invalid response_harmfulness: {response_harmfulness!r}"
        )

    cot_component = (
        0.0 if cot_format_violation is None
        else -1.0 if cot_format_violation
        else 1.0
    )

    components = {
        "prompt_harmfulness": prompt_component,
        "response_refusal": refusal_component,
        "response_harmfulness": harmfulness_component,
        "cot_format": cot_component,
    }
    return {
        "reward": float(sum(components.values())),
        "label_drift": False,
        "components": components,
    }


# --- flat invalid-move penalty (v6d base) -------------------------------------
#
# The v6d reward (actual-label-v4 + the asymmetric cap above) kept benign
# retention intact over 300-step attackers, but capping a drifted game at zero
# still ties it with a faithful benign rewrite, and nothing penalised an
# off-topic rewrite: v6d's A4 emitted one memorised request for all 60 held-out
# seeds.  shared_binary then dropped drifted games outright, and every 300-step
# attacker converted all benign seeds into harmful prompts within 80-160 steps.
#
# This keeps every v6d term and replaces the attacker reward of an invalid move
# with a flat floor: -3 for one violation (label drift in either direction, or
# an off-topic verdict), -4 for both, minus 2 more for a CoT violation.  The
# worst valid outcome is 0 (-2 with a CoT violation), so every invalid move is
# strictly and uniformly dominated, independent of how the defender replied.
# A missing prompt label counts as drift, so making WildGuard abstain on the
# prompt cannot sidestep the check.  A reply WildGuard cannot classify earns
# the attacker nothing above zero.  The defender's reward is untouched.

ATTACKER_INVALID_MOVE_POLICY = "v6d-flat-invalid-move-v1"
ATTACKER_INVALID_MOVE_BASE = -2.0


def invalid_move_reward_enabled() -> bool:
    value = os.environ.get("ATTACKER_INVALID_MOVE", "0")
    if value not in {"0", "1"}:
        raise ValueError("ATTACKER_INVALID_MOVE must be 0 or 1")
    return value == "1"


def invalid_move_attacker_reward(
    *,
    reward: float,
    prompt_type: str,
    labels: "Mapping[str, object]",
    cot_format_violation: bool | None,
) -> tuple[float, int]:
    """Return the attacker reward and the number of invalid moves (0, 1 or 2)."""

    expected = _EXPECTED_PROMPT_LABEL.get(prompt_type)
    if expected is None:
        return float(reward), 0
    violations = int(labels.get("prompt_harmfulness") != expected) + int(
        bool(labels.get("off_topic"))
    )
    if violations:
        penalty = ATTACKER_INVALID_MOVE_BASE - violations
        return penalty - (2.0 if cot_format_violation else 0.0), violations
    if labels.get("response_refusal") is None or labels.get("response_harmfulness") is None:
        return min(float(reward), 0.0), 0
    return float(reward), 0
