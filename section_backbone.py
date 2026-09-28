"""NumPy integration shared by fitted-model calls and section-family refitting.

The gates and secant endpoint solve are extracted from
analysis/section_family.py. Units are N and mm for dimensional calls; the
same functions integrate the normalized law with EI0 = Mu = 1.
"""
from __future__ import annotations

import numpy as np


def logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def base_grid(EI0, Mk, kappa_u, *, n_nodes=2001, knee_width=0.15,
              plateau_width=0.08, plateau_floor=0.02):
    """Return B, C, curvature and gates, so the base area equals r1*B + C."""
    kappa = np.linspace(0.0, kappa_u, n_nodes)
    kappa_k = Mk / EI0
    transition = logistic((kappa_k - kappa) / (knee_width * max(kappa_k, 1e-300)))
    plateau = plateau_floor + (1.0 - plateau_floor) * logistic(
        (kappa_u - kappa) / (plateau_width * kappa_u))
    B = float(np.trapezoid(EI0 * (1.0 - transition) * plateau, kappa))
    C = float(np.trapezoid(EI0 * transition * plateau, kappa))
    return B, C, kappa, transition, plateau


def base_area(EI0, Mk, r1, kappa_u):
    B, C, *_ = base_grid(EI0, Mk, kappa_u)
    return r1 * B + C


def solve_endpoint(EI0, Mk, r1, Mu, *, area=base_area, seed_fraction=0.95,
                   upper_bound=1.0, max_iter=6):
    """Solve the positive-tangent area identity; return (endpoint, converged).

    Refitting supplies its active shape through ``area`` and retains its
    dimensional bounds. A normalized call uses ``upper_bound=np.inf``.
    """
    if Mu <= Mk * 1.001:
        return 1.0e-6, False
    k0 = max((Mu - Mk) / (seed_fraction * r1 * EI0), 1.0e-8)
    f0 = area(EI0, Mk, r1, k0) - Mu
    k1 = k0 * (1.05 if f0 < 0 else 0.95)
    f1 = area(EI0, Mk, r1, k1) - Mu
    for _ in range(max_iter):
        if abs(f1) < 1.0e-9 * Mu or abs(f1 - f0) < 1.0e-30:
            break
        step = float(np.clip(k1 - f1 * (k1 - k0) / (f1 - f0), 1.0e-8, upper_bound))
        k0, f0 = k1, f1
        k1 = step
        f1 = area(EI0, Mk, r1, k1) - Mu
    return float(k1), bool(abs(f1) < 1.0e-4 * Mu)
