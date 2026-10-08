"""Physical model, quadrature and piecewise-constant control matrices.

The entries of ``weights`` are block masses. Rates are ``n * weights``.
The admissible set is sum(weights) = 1 with 0 <= weights <= 2/n.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

BETA_MIN, BETA_MAX = 0.5, 2.0
R = 2.0
MODEL = {
    "T": 1, "Q": 1, "R": 2, "sigma": 1, "tau": 0,
    "beta_interval": [0.5, 2], "theta_interval": [-0.5, 0.5],
    "g_theta": "(2/3)*(1+theta*(5-4*beta)/3)",
}


@dataclass(frozen=True)
class Quadrature:
    """Orders used in the beta mixture, execution cells and recovery window."""

    beta_order: int = 64
    execution_order: int = 12
    recovery_order: int = 80

    def __post_init__(self):
        if min(self.beta_order, self.execution_order, self.recovery_order) < 2:
            raise ValueError("Quadrature orders must be at least 2.")

    def as_dict(self):
        return asdict(self)


DEFAULT_QUADRATURE = Quadrature()
REFINED_QUADRATURE = Quadrature(96, 24, 120)


def gauss(a, b, order):
    """Gauss-Legendre nodes and weights on [a, b]."""
    z, w = leggauss(order)
    return (a + b) / 2 + (b - a) * z / 2, (b - a) * w / 2


@lru_cache(maxsize=16)
def mixture_quadrature(order=64):
    beta, bw = gauss(BETA_MIN, BETA_MAX, order)
    hw = bw * (2 / 9) * (5 - 4 * beta)
    beta.setflags(write=False)
    hw.setflags(write=False)
    return beta, hw


def H(t, beta_order=64):
    """Zero-mass kernel tangent H(t), evaluated for nonnegative t."""
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("H requires nonnegative times.")
    beta, hw = mixture_quadrature(beta_order)
    z = t.reshape(-1)
    out = np.empty_like(z)
    for j in range(0, len(z), 4096):
        out[j:j + 4096] = np.expm1(-z[j:j + 4096, None] * beta) @ hw
    return out.reshape(t.shape)


def derivative_matrix(t, n, beta_order=64):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    left = np.arange(n) / n
    right = (np.arange(n) + 1) / n
    a = np.maximum(t[:, None] - left, 0)
    b = np.maximum(t[:, None] - right, 0)
    return n * (H(a, beta_order) - H(b, beta_order))


def matrices(n, U, order=None, *, quadrature=DEFAULT_QUADRATURE):
    """Return cost tangent A, cost variance V and tape information P.

    ``order`` sets the quadrature order within each execution interval.
    Supply a Quadrature instance to vary all three quadrature orders.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer at least 2.")
    if not np.isfinite(U) or U < 0:
        raise ValueError("U must be finite and nonnegative.")
    if order is not None:
        quadrature = Quadrature(quadrature.beta_order, order, quadrature.recovery_order)
    beta, hw = mixture_quadrature(quadrature.beta_order)
    delta = 1 / n
    centers = (np.arange(n) + 0.5) / n
    dist = np.abs(centers[:, None] - centers[None, :])
    kernel = np.exp(-beta[:, None, None] * dist) * (
        np.sinh(beta * delta / 2) / (beta * delta / 2)
    )[:, None, None] ** 2
    diag = 2 * (beta * delta + np.expm1(-beta * delta)) / (beta * delta) ** 2
    kernel[:, np.arange(n), np.arange(n)] = diag[:, None]
    A = 0.5 * np.einsum("b,bij->ij", hw, kernel)
    V = np.minimum(centers[:, None], centers[None, :])
    V[np.arange(n), np.arange(n)] = np.arange(n) / n + delta / 3
    zs, ws = gauss(0, delta, quadrature.execution_order)
    te = (np.arange(n)[:, None] * delta + zs).reshape(-1)
    we = np.tile(ws, n)
    Me = derivative_matrix(te, n, quadrature.beta_order)
    P = Me.T @ (we[:, None] * Me)
    if U > 0:
        tr, wr = gauss(1, 1 + U, quadrature.recovery_order)
        Mr = derivative_matrix(tr, n, quadrature.beta_order)
        P += Mr.T @ (wr[:, None] * Mr)
    return A, V, (P + P.T) / 2


def information(weights, A, V, P):
    w = np.asarray(weights, dtype=float)
    d, variance = float(w @ A @ w), float(w @ V @ w)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("The cost variance must be finite and positive.")
    return {"J_cost": d * d / variance, "J_tape": float(w @ P @ w),
            "d": d, "variance": variance}
