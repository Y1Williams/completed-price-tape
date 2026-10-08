"""Analytical envelope formulas with deterministic floating-point evaluation.

PAD is a numerical allowance. Total quadrature and roundoff errors require
separate bounds. The analytical arguments are described in the methods.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.signal import lfilter
from scipy.sparse import coo_matrix, lil_matrix

from .model import gauss

PAD = 2e-10


def cost_envelope(w, A, V, m=65536, beta_order=96):
    """Evaluate the continuous-control supporting-functional envelope."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    if m % n or m % 2:
        raise ValueError("The midpoint mesh must be even and divisible by n.")
    rates = np.repeat(w * n, m // n)
    dt = 1 / m
    beta, bw = gauss(0.5, 2, beta_order)
    hw = bw * (2 / 9) * (5 - 4 * beta)
    potential = np.zeros(m)
    for b, h in zip(beta, hw):
        decay = np.exp(-b * dt)
        step = -np.expm1(-b * dt) / b
        le = np.r_[0, lfilter([step], [1, -decay], rates)]
        ri = np.r_[lfilter([step], [1, -decay], rates[::-1])[::-1], 0]
        half = -np.expm1(-b * dt / 2) / b
        potential += h * (np.exp(-b * dt / 2) * (le[:-1] + ri[1:]) + 2 * rates * half)
    Fend = 1 - np.r_[0, np.cumsum(rates) * dt]
    k_end = np.r_[0, np.cumsum((Fend[:-1] + Fend[1:]) * dt / 2)]
    k_mid = k_end[:-1] + Fend[:-1] * dt / 2 - rates * dt * dt / 8
    d, variance = float(w @ A @ w), float(w @ V @ w)
    gradient = potential - d / variance * k_mid
    midpoint_support = 2 * np.partition(gradient, m // 2)[m // 2:].sum() / m
    lipschitz = 0.25 + d / variance
    slack = max(0, midpoint_support + lipschitz / (2 * m) - d + PAD)
    lower = d * d / variance - PAD
    upper = (d / np.sqrt(variance) + np.sqrt(6) * slack) ** 2 + PAD
    return {"lower": float(lower), "upper": float(upper), "d": d,
            "variance": variance, "gradient_lipschitz": float(lipschitz),
            "support_slack": float(slack), "midpoint_mesh": m, "beta_order": beta_order}


def pulse_information(U, order=96):
    """Earliest half-pulse tape information, and its infinite-window value.

    The finite-window pulse is a proved continuum maximizer only for U >= 2.
    For U < 2 the returned pulse value is a feasible-design lower bound.
    """
    if not np.isfinite(U) or U < 0:
        raise ValueError("U must be finite and nonnegative.")
    beta, bw = gauss(0.5, 2, order)
    h = bw * (2 / 9) * (5 - 4 * beta)
    B = beta[:, None] + beta[None, :]
    first = 4 * np.sum((h[:, None] * h[None, :]) * (-np.expm1(-B / 2)) / B)
    c = h * (-np.expm1(-beta / 2))
    infinite = first + 4 * np.sum(c[:, None] * c[None, :] / B)
    tail = 4 * np.sum((c[:, None] * c[None, :]) * np.exp(-B * (0.5 + U)) / B)
    return float(infinite - tail), float(infinite)


def sign_margins():
    pbar = 19 / 128 * (5 / 24 - 2 * np.log(2) / 9)
    return {
        "autocorrelation_derivative_margin": float(1 / 32 - (np.exp(1.5) - 1) * pbar),
        "H_second_derivative_at_2": float(7 * (22 - np.exp(3)) * np.exp(-4) / 36),
    }


def tape_lp_bound(P):
    """Optional LP relaxation on a finite grid, available through this function."""
    n = len(P)
    if n % 2:
        raise ValueError("The capped binary-control relaxation requires even n.")
    k = n // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    count = n + len(pairs)
    objective = np.r_[-np.diag(P), [-2 * P[i, j] for i, j in pairs]] * (2 / n) ** 2
    eq = lil_matrix((n + 1, count))
    eq[0, :n] = 1
    for i in range(n):
        eq[i + 1, i] = -(k - 1)
    rr, cc, vv, rhs = [], [], [], []
    for p, (i, j) in enumerate(pairs):
        ix = n + p
        eq[i + 1, ix], eq[j + 1, ix] = 1, 1
        r = len(rhs)
        rr.extend([r, r])
        cc.extend([ix, i])
        vv.extend([1, -1])
        rhs.append(0)
        r = len(rhs)
        rr.extend([r, r])
        cc.extend([ix, j])
        vv.extend([1, -1])
        rhs.append(0)
        r = len(rhs)
        rr.extend([r, r, r])
        cc.extend([i, j, ix])
        vv.extend([1, 1, -1])
        rhs.append(1)
    ub = coo_matrix((vv, (rr, cc)), shape=(len(rhs), count)).tocsr()
    result = linprog(objective, A_ub=ub, b_ub=rhs, A_eq=eq.tocsr(),
                     b_eq=np.r_[k, np.zeros(n)], bounds=(0, 1), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun), result
