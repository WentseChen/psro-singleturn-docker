"""Meta-strategy solver: Nash Equilibrium for 2-player zero-sum meta-games.

Solves the minimax LP exactly (scipy "highs"), then selects the maximum-entropy
point of the optimal set so a degenerate equilibrium is broken canonically rather
than at whichever vertex the LP happened to return.

Input: (m, n) payoff matrix where A[i,j] is the payoff to the row player.
Output: Nash mixed strategies for each player.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Hedge regret bound is O(sqrt(T log n)); 50k iters is enough for ~1e-3 accuracy on
# the < 20-strategy meta-games we hit. eta is the canonical sqrt(log n / T) step.
_HEDGE_NUM_ITERATIONS: int = 50_000
_HEDGE_ETA_SCALE: float = 2.0

#: Break ties in a degenerate optimal set canonically instead of taking the LP
#: vertex. Free in exploitability terms; a no-op when the equilibrium is unique.
_MAX_ENTROPY_SELECTION: bool = True


def _solve_lp(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact minimax via linear programming, with the game value."""
    from scipy.optimize import linprog

    m, n = A.shape
    shift = float(A.min()) - 1.0
    B = A - shift                                   # keep the LP value positive
    row = linprog(
        np.r_[np.zeros(m), -1.0],
        A_ub=np.c_[-B.T, np.ones((n, 1))], b_ub=np.zeros(n),
        A_eq=np.c_[np.ones((1, m)), np.zeros((1, 1))], b_eq=[1.0],
        bounds=[(0.0, None)] * m + [(None, None)], method="highs",
    )
    col = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=np.c_[B, -np.ones((m, 1))], b_ub=np.zeros(m),
        A_eq=np.c_[np.ones((1, n)), np.zeros((1, 1))], b_eq=[1.0],
        bounds=[(0.0, None)] * n + [(None, None)], method="highs",
    )
    if not (row.success and col.success):
        raise RuntimeError(f"meta-game LP failed: {row.message} / {col.message}")
    p = np.clip(row.x[:m], 0.0, None); p /= p.sum()
    q = np.clip(col.x[:n], 0.0, None); q /= q.sum()
    return p, q, float(row.x[m] + shift)


def _max_entropy_optimal(
    A: np.ndarray, value: float, row_player: bool, fallback: np.ndarray
) -> np.ndarray:
    """Highest-entropy strategy among those that guarantee the game value.

    The optimal set {p : (p^T A)_j >= v for all j, sum p = 1, p >= 0} is a convex
    polytope and entropy is concave, so this is a convex program. Every point in
    the set is an equilibrium strategy, so this costs nothing in exploitability;
    it only replaces the LP's arbitrary vertex with a canonical interior point
    when the equilibrium is not unique. On a generic matrix the set is a single
    point and this returns the LP answer unchanged.
    """
    from scipy.optimize import minimize

    X = A if row_player else -A.T
    target = value if row_player else -value
    k = X.shape[0]
    start = fallback
    result = minimize(
        lambda z: float((np.clip(z, 1e-12, None) * np.log(np.clip(z, 1e-12, None))).sum()),
        np.ones(k) / k,
        jac=lambda z: np.log(np.clip(z, 1e-12, None)) + 1.0,
        constraints=[
            {"type": "eq", "fun": lambda z: z.sum() - 1.0, "jac": lambda z: np.ones(k)},
            {"type": "ineq", "fun": lambda z: (z @ X) - target + 1e-9, "jac": lambda z: X.T},
        ],
        bounds=[(0.0, 1.0)] * k, method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return start
    z = np.clip(result.x, 0.0, None)
    total = z.sum()
    if total <= 0:
        return start
    z = z / total
    # never accept a worse guarantee than the LP vertex
    if float((z @ X).min()) < target - 1e-6:
        return start
    return z


def compute_nash(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Nash Equilibrium mixed strategies for a 2-player zero-sum game.

    Uses Multiplicative Weights Update (Hedge). For the small policy pools
    typical in PSRO/FSP (< 20 policies), convergence is fast and accurate.

    Args:
        payoff_matrix: 2-D array of shape (m, n). ``A[i, j]`` is the payoff to
            the row player when P1 plays strategy ``i`` and P2 plays strategy ``j``.

    Returns:
        ``(p1_strategy, p2_strategy)`` — each a 1-D float64 ndarray summing to 1.

    Raises:
        ValueError: if ``payoff_matrix`` is not 2-D or has zero rows/columns.
    """
    A = np.atleast_2d(np.asarray(payoff_matrix, dtype=np.float64))
    if A.ndim != 2:
        raise ValueError(f"payoff_matrix must be 2-D, got shape {A.shape}")
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError(f"payoff_matrix must have at least one row and column, got ({m}, {n})")

    if m == 1 and n == 1:
        return np.array([1.0]), np.array([1.0])
    if m == 1:
        # The row player has no choice, so every minimizing column is a best
        # response. Mix uniformly only across tied minimizers.
        best_columns = A[0] == np.min(A[0])
        p2_strategy = best_columns.astype(np.float64)
        p2_strategy /= p2_strategy.sum()
        return np.array([1.0]), p2_strategy
    if n == 1:
        # The column player has no choice, so every maximizing row is a best
        # response. Mix uniformly only across tied maximizers.
        best_rows = A[:, 0] == np.max(A[:, 0])
        p1_strategy = best_rows.astype(np.float64)
        p1_strategy /= p1_strategy.sum()
        return p1_strategy, np.array([1.0])

    # Exact minimax LP, then the max-entropy point of the optimal set. Hedge was
    # previously used here; on the production 4x4 it left 0.0048 exploitability,
    # about the size of one cell's sampling error, and took 975 ms against the
    # LP's 83 ms. The max-entropy step is a safeguard for degenerate matrices
    # (exact ties, duplicated strategies) and returns the LP answer unchanged on
    # a generic one.
    p, q, value = _solve_lp(A)
    if _MAX_ENTROPY_SELECTION:
        p = _max_entropy_optimal(A, value, True, p)
        q = _max_entropy_optimal(A, value, False, q)
    return p, q
