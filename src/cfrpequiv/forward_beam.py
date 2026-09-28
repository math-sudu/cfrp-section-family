"""Three-point-bending member forward map, with member/section equilibrium enforced EXACTLY.

Geometry (simply supported, central point load)
-----------------------------------------------
Clear span ``L`` [mm], central point load ``P`` [N], mid-span deflection ``delta`` [mm].
The simply-supported, centrally-loaded beam is STATICALLY DETERMINATE: the bending-moment
field is fixed by equilibrium alone, independent of the section law, as the triangle::

    M(x) = P*x/2          for x in [0, L/2]   (symmetric about mid-span)
    M_mid = P*L/4         (peak moment, at x = L/2)

This exact moment field is the binding de-convolution constraint of the inverse problem: it
ties the integrated member load-deflection response back to the section moment-curvature law
point-wise via ``kappa(x) = law.curvature(M(x))``, so the SECTION law is the recovered unknown
rather than the member curve. Equilibrium is not a penalty to be fitted; it is a hard property
of this forward map.

Mid-span deflection (unit-load / virtual-work method)
-----------------------------------------------------
For a unit mid-span point load the virtual moment is ``m_bar(x) = x/2`` on ``[0, L/2]``.
Virtual work over the (symmetric) half-spans gives::

    delta_mid(P) = 2 * integral_0^{L/2} kappa(x) * (x/2) dx

This integral is evaluated with Gauss-Legendre quadrature on ``[0, L/2]``. The map is
non-linear in ``P`` because ``kappa = law.curvature(M)`` is non-linear; it is monotone in ``P``
because the section law is monotone.

Constant-EI sanity (used by tests)
-----------------------------------
For a near-linear law (``kappa = M/EI0``)::

    delta_mid = 2 * int_0^{L/2} (P*x/(2*EI0)) * (x/2) dx
              = (P/(2*EI0)) * (L/2)^3/3
              = P*L^3/(48*EI0)

i.e. the Euler-Bernoulli three-point-bending result.

Peak load
---------
Section failure occurs when the mid-span moment reaches ``Mu`` -> ``P_u = 4*Mu/L``.

Equilibrium-consistency diagnostic
----------------------------------
``member_equilibrium_residual`` returns the inconsistency, for each measured ``(P, delta)``
pair, between the measured mid-span deflection and the deflection predicted by integrating the
section law over the (exact) equilibrium moment field. Because the moment field is already in
exact equilibrium with ``P`` by construction, this residual carries section-law / compatibility
information that is the same misfit the data block already minimizes -- it is therefore used as
a POST-FIT verification quantity (it should be ~0 once the forward map reproduces the measured
response), NOT as an extra fitted residual that would double-count the same deflection misfit
into the identifiability analysis. See ``identify.py`` for how it is reported.

Only numpy / scipy are used.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "deflection_at_load",
    "load_at_deflection",
    "peak_load",
    "member_equilibrium_residual",
]

# Default number of Gauss-Legendre nodes on the half-span [0, L/2]. The integrand
# kappa(x)*(x/2) is smooth (the section law is C1), so a modest order is accurate.
_DEFAULT_N_QUAD = 64


def _gauss_legendre_halfspan(span_mm: float, n_quad: int):
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to the half-span [0, L/2]."""
    half = 0.5 * float(span_mm)
    xi, wi = np.polynomial.legendre.leggauss(int(n_quad))
    # map xi in [-1,1] -> x in [0, half]
    x = 0.5 * half * (xi + 1.0)
    w = 0.5 * half * wi
    return x, w


def deflection_at_load(law, P_N, span_mm, n_quad: int = _DEFAULT_N_QUAD):
    """Mid-span deflection [mm] for central load(s) ``P_N`` [N] via virtual work.

    Vectorized over ``P_N`` (scalar or array). Uses ``law.curvature`` on the equilibrium
    moment field ``M(x)=P*x/2`` and integrates ``2 * int_0^{L/2} kappa(x)*(x/2) dx`` with
    Gauss-Legendre quadrature.
    """
    P = np.asarray(P_N, dtype=float)
    scalar = P.ndim == 0
    Pf = np.atleast_1d(P)

    x, w = _gauss_legendre_halfspan(span_mm, n_quad)  # (Q,)
    m_bar = 0.5 * x  # virtual moment for unit mid-span load

    # Moment field per load: shape (N, Q). M(x) = P * x / 2.
    M = (Pf[:, None] * x[None, :]) * 0.5
    kappa = law.curvature(M)  # vectorized over the whole (N, Q) grid
    kappa = np.asarray(kappa, dtype=float).reshape(M.shape)

    # delta = 2 * sum_q w_q * kappa(x_q) * m_bar(x_q)
    delta = 2.0 * (kappa * (m_bar[None, :] * w[None, :])).sum(axis=1)
    return float(delta[0]) if scalar else delta


def peak_load(law, span_mm) -> float:
    """Peak (failure) central load ``P_u = 4*Mu/L`` [N] from the section ultimate moment."""
    Mu, _kappa_u = law.failure()
    return 4.0 * float(Mu) / float(span_mm)


def load_at_deflection(law, delta_mm, span_mm, n_quad: int = _DEFAULT_N_QUAD):
    """Numerical inverse of :func:`deflection_at_load`: central load [N] for deflection(s) [mm].

    Vectorized over ``delta_mm``. ``deflection_at_load`` is monotone increasing in ``P`` (the
    section law is monotone), so each target deflection is bracketed in ``[0, P_u]`` and found
    by bisection. Deflections beyond the value at ``P_u`` are clipped to ``P_u`` (post-failure
    is not represented by the monotone backbone).
    """
    d = np.asarray(delta_mm, dtype=float)
    scalar = d.ndim == 0
    df = np.atleast_1d(d)

    P_u = peak_load(law, span_mm)
    delta_u = deflection_at_load(law, P_u, span_mm, n_quad=n_quad)

    out = np.empty_like(df)
    for i, dtarget in enumerate(df):
        if dtarget <= 0.0:
            out[i] = 0.0
            continue
        if dtarget >= delta_u:
            out[i] = P_u
            continue
        lo, hi = 0.0, P_u
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            dmid = deflection_at_load(law, mid, span_mm, n_quad=n_quad)
            if dmid < dtarget:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return float(out[0]) if scalar else out


def member_equilibrium_residual(law, P_N, delta_mm, span_mm, n_quad: int = _DEFAULT_N_QUAD):
    """Equilibrium-consistency residual for measured (P, delta) pairs (POST-FIT diagnostic).

    For each measured pair, the equilibrium moment field ``M(x)=P*x/2`` is exact (statically
    determinate beam); the section law maps it to a curvature field whose virtual-work integral
    yields the *predicted* mid-span deflection. The residual is the (signed) inconsistency with
    the *measured* deflection::

        residual_i = delta_pred(P_i) - delta_meas_i

    Normalized by a characteristic deflection so the term is dimensionless. This is the same
    deflection misfit the data block of the inverse loss already minimizes; it is therefore
    reported AFTER the fit as a verification that the equilibrium-consistent forward map solves
    the governing relation (the residual should be ~0), and is deliberately NOT stacked into the
    fitted residual vector -- doing so would double-count the data misfit into the K1 covariance
    / condition number / confidence band. Returns a 1-D array the length of the input pairs.
    """
    P = np.atleast_1d(np.asarray(P_N, dtype=float))
    d_meas = np.atleast_1d(np.asarray(delta_mm, dtype=float))
    if P.shape != d_meas.shape:
        raise ValueError(
            f"P_N and delta_mm must have matching shape, got {P.shape} vs {d_meas.shape}"
        )
    d_pred = np.atleast_1d(deflection_at_load(law, P, span_mm, n_quad=n_quad))
    # Characteristic scale: robust positive deflection magnitude of the measured set.
    scale = float(np.maximum(np.median(np.abs(d_meas)), 1e-9))
    return (d_pred - d_meas) / scale
