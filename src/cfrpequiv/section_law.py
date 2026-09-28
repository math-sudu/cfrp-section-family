"""Structured, monotone moment-curvature section law for the equivalent CFRP-reinforced section.

The recovered unknown of the E2 inverse problem is a homogenized equivalent section
constitutive law expressed in the moment-curvature plane: an effective flexural stiffness
``EI_eff(kappa)``, the moment-curvature relation ``M(kappa)``, and a strength-type section
failure pair ``(Mu, kappa_u)``.

Function class
--------------
The law is a *smoothed bi-linear* moment-curvature backbone with a final smooth plateau,
parameterized by the five physically meaningful numbers::

    SECTION_PARAM_NAMES = ["EI0", "Mcr", "r1", "Mu", "kappa_u"]

* ``EI0``     -- initial (uncracked) tangent flexural stiffness, ``M'(0) = EI0``    [N*mm^2]
* ``Mcr``     -- knee-LOCATION parameter: the knee is CENTERED at the curvature
                 ``kappa_cr = Mcr / EI0`` (i.e. Mcr is where the initial branch would
                 reach if extrapolated). Because the knee is a smooth (logistic) blend
                 centered there, the actual moment at the knee curvature satisfies
                 ``M(kappa_cr) ~ 0.9-0.95 * Mcr`` by construction. Mcr therefore fixes the
                 knee location, and is NOT the literal first-nonlinearity (cracking) moment. [N*mm]
* ``r1``      -- post-knee tangent stiffness ratio ``EI1/EI0``, ``0 < r1 < 1``      [-]
* ``Mu``      -- ultimate (plateau) moment                                          [N*mm]
* ``kappa_u`` -- ultimate curvature at section failure                             [1/mm]

Closed form
-----------
The moment is built as the integral of a strictly positive, smooth tangent-stiffness
profile so that ``M(kappa)`` is automatically C1 and monotonically increasing::

    EI_tangent(kappa) = EI0 * [ r1 + (1 - r1) * s_knee(kappa) ] * g_plateau(kappa)
    M(kappa)          = integral_0^kappa EI_tangent(t) dt

where

* ``s_knee`` is a smooth step from 1 down to 0 (a logistic in *moment*-like coordinate),
  giving the bi-linear knee: tangent ``EI0`` below the knee, ``r1*EI0`` above it. The knee
  is centered on the curvature ``kappa_cr`` that the initial branch would reach at ``Mcr``
  (``kappa_cr = Mcr / EI0``), with a softening width proportional to ``kappa_cr``.
* ``g_plateau`` is a smooth multiplier that decays the tangent toward zero as
  ``kappa -> kappa_u``, producing a smooth approach to the plateau ``Mu``. Its amplitude is
  calibrated by a single scalar so that ``M(kappa_u) = Mu`` exactly (to solver tolerance).

The integral is evaluated with a fixed dense Gauss-Legendre / cumulative-trapezoid rule on
[0, kappa_u]; values are obtained by interpolation, which keeps ``moment`` cheap, smooth and
vectorizable while remaining differentiable enough for a finite-difference Jacobian.

Only numpy / scipy are used. The class is deterministic and stateless after construction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SECTION_PARAM_NAMES",
    "SectionLaw",
    "make_section_law",
    "shape_constants",
    "SHAPE_CONSTANT_NAMES",
]

# Locked parameter ordering (do not reorder; identify.py and export rely on this).
SECTION_PARAM_NAMES = ["EI0", "Mcr", "r1", "Mu", "kappa_u"]

# Number of quadrature / interpolation nodes used to build the M(kappa) backbone.
_N_NODES = 2001

# --- shape constants of the law FORM (not identified from data) ------------------------------
#
# These are smoothing scales, not fitted quantities. The law is a bi-linear-with-plateau
# backbone that has to be C1 for a finite-difference Jacobian to mean anything, so each corner
# of the piecewise shape is replaced by a logistic blend, and each blend needs a width. The
# widths are set as fractions of the curvature scale the corner sits on -- the knee on
# kappa_cr, the plateau on kappa_u -- so that one number serves every configuration in a family
# whose curvatures span an order of magnitude. Their VALUES are chosen small enough that the
# bi-linear character stays visible and large enough that the blend spans many backbone nodes;
# nothing in the data selects them. That is why the manuscript reports the identified family's
# sensitivity to doubling and halving each of them rather than asserting the choice.
#
# They live here as module attributes, read at call time, so the sensitivity sweep can move
# them through :func:`shape_constants` without threading a parameter through the whole fitting
# stack. ``analysis.section_family`` mirrors these by reading THIS module at call time;
# ``analysis.jax_family`` snapshots the defaults at import and is not swept.
KNEE_WIDTH_FRAC = 0.15      # knee blend width, as a fraction of kappa_cr = Mcr/EI0
PLATEAU_WIDTH_FRAC = 0.08   # plateau blend width, as a fraction of kappa_u
PLATEAU_FLOOR = 0.02        # residual tangent fraction at kappa_u, keeping M strictly monotone
BUMP_HI_FRAC = 0.97         # area-correction window: upper gate, as a fraction of kappa_u
BUMP_WIDTH_FRAC = 0.06      # area-correction window: gate width, as a fraction of kappa_u
BUMP_FLOOR_FRAC = 0.95      # clamp keeping the corrected tangent at >= 5% of the base minimum

SHAPE_CONSTANT_NAMES = (
    "KNEE_WIDTH_FRAC",
    "PLATEAU_WIDTH_FRAC",
    "PLATEAU_FLOOR",
    "BUMP_HI_FRAC",
    "BUMP_WIDTH_FRAC",
    "BUMP_FLOOR_FRAC",
)


@contextmanager
def shape_constants(**overrides):
    """Temporarily replace one or more shape constants of the law form.

    For the sensitivity study the law form owes its hard-coded widths::

        with shape_constants(KNEE_WIDTH_FRAC=0.30):
            ...  # every law built in here uses the doubled knee width

    Laws already constructed are unaffected -- the constants are read while a backbone is
    built. The override reaches every consumer that reads this module at call time, which is
    :class:`SectionLaw` itself and ``analysis.section_family``'s kappa_u solve; it does NOT
    reach ``analysis.jax_family``, whose jitted forward model snapshots the defaults, so a
    sweep that needs the Bayesian layer has to say so.
    """
    unknown = set(overrides) - set(SHAPE_CONSTANT_NAMES)
    if unknown:
        raise ValueError(f"not shape constants of the law form: {sorted(unknown)}")
    previous = {name: globals()[name] for name in overrides}
    globals().update(overrides)
    try:
        yield
    finally:
        globals().update(previous)

# np.trapz was removed in NumPy 2.0 in favor of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _softplus(x: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softplus ``log(1 + exp(beta*x)) / beta`` (unused entry kept tiny)."""
    bx = beta * np.asarray(x, dtype=float)
    return np.where(bx > 30.0, x, np.log1p(np.exp(np.minimum(bx, 30.0))) / beta)


def _logistic(x: np.ndarray) -> np.ndarray:
    """Stable logistic sigmoid 1 / (1 + exp(-x))."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _coerce_params(params) -> dict:
    """Accept a dict (any subset order) or an array in SECTION_PARAM_NAMES order -> dict."""
    if isinstance(params, dict):
        missing = [k for k in SECTION_PARAM_NAMES if k not in params]
        if missing:
            raise KeyError(f"section params missing keys: {missing}")
        return {k: float(params[k]) for k in SECTION_PARAM_NAMES}
    arr = np.asarray(params, dtype=float).ravel()
    if arr.size != len(SECTION_PARAM_NAMES):
        raise ValueError(
            f"expected {len(SECTION_PARAM_NAMES)} params {SECTION_PARAM_NAMES}, got {arr.size}"
        )
    return {k: float(v) for k, v in zip(SECTION_PARAM_NAMES, arr)}


@dataclass(frozen=True)
class _Backbone:
    """Precomputed dense (kappa, M) backbone plus tangent samples for interpolation."""

    kappa: np.ndarray  # monotone nodes on [0, kappa_u]
    moment: np.ndarray  # cumulative integral of EI_tangent -> M(kappa_node)
    tangent: np.ndarray  # EI_tangent at each node


class SectionLaw:
    """Smoothed bi-linear-with-plateau moment-curvature section law.

    Construct from a parameter dict or an array ordered as ``SECTION_PARAM_NAMES``.
    All public methods accept numpy arrays and python scalars and return the matching shape.
    """

    def __init__(self, params):
        p = _coerce_params(params)
        self.EI0 = p["EI0"]
        self.Mcr = p["Mcr"]
        self.r1 = p["r1"]
        self.Mu = p["Mu"]
        self.kappa_u = p["kappa_u"]

        # --- light validation: the law must be physically orderable ---------------------
        if not (self.EI0 > 0.0):
            raise ValueError(f"EI0 must be > 0, got {self.EI0}")
        if not (0.0 < self.r1 < 1.0):
            raise ValueError(f"r1 must satisfy 0 < r1 < 1, got {self.r1}")
        if not (self.kappa_u > 0.0):
            raise ValueError(f"kappa_u must be > 0, got {self.kappa_u}")
        if not (self.Mcr > 0.0):
            raise ValueError(f"Mcr must be > 0, got {self.Mcr}")
        if not (self.Mu > self.Mcr):
            raise ValueError(f"Mu must exceed Mcr, got Mu={self.Mu}, Mcr={self.Mcr}")

        # Curvature at which the initial (uncracked) branch reaches the knee moment Mcr.
        self._kappa_cr = self.Mcr / self.EI0
        # Smoothing widths (relative -> absolute). Chosen small vs the relevant scale so the
        # bi-linear character is sharp but C1; see the module's shape-constant block.
        self._w_knee = KNEE_WIDTH_FRAC * max(self._kappa_cr, 1e-300)
        self._w_plateau = PLATEAU_WIDTH_FRAC * self.kappa_u

        self._backbone = self._build_backbone()

    # ------------------------------------------------------------------ construction -----
    #
    # Tangent profile design
    # ----------------------
    # The tangent stiffness is assembled from three physically separable pieces:
    #
    #   EI_tangent(kappa) = EI_base(kappa) * p_plateau(kappa)  +  c * bump(kappa)
    #
    #   * EI_base = EI0 * [r1 + (1-r1)*s_knee]   -- the smoothed bi-linear: slope EI0 below
    #     the knee curvature kappa_cr = Mcr/EI0, blending to slope r1*EI0 above it. This fixes
    #     M'(0)=EI0, the knee location (Mcr), and the post-cracking ratio r1.
    #   * p_plateau(kappa)  -- a smooth multiplier that is 1 over most of the range and decays
    #     to a small floor as kappa -> kappa_u, so the tangent becomes small near kappa_u: the
    #     strength-type plateau (a smooth approach to Mu, dM/dkappa -> small at failure).
    #   * c * bump(kappa)   -- an area-correction term. ``bump`` is a non-negative smooth window
    #     that vanishes at both kappa=0 and kappa=kappa_u (so it perturbs neither the initial
    #     tangent nor the plateau tangent) and is peaked in the upper-mid range. Its coefficient
    #     c is solved so the total integral equals Mu EXACTLY. Because the integral is linear in
    #     c, a single closed-form solve suffices (no iteration). c>0 adds hardening when the base
    #     undershoots Mu; c<0 trims area when it overshoots, and is clamped so the tangent stays
    #     strictly positive (monotone M).
    #
    def _base_tangent(self, kappa: np.ndarray) -> np.ndarray:
        """Smoothed bi-linear base tangent times the plateau-suppression multiplier."""
        kappa = np.asarray(kappa, dtype=float)
        # Smooth knee: 1 below kappa_cr, -> 0 above, blending EI0 -> r1*EI0.
        s_knee = _logistic((self._kappa_cr - kappa) / self._w_knee)
        ratio = self.r1 + (1.0 - self.r1) * s_knee  # in (r1, 1)
        # Plateau suppression: ~1 below kappa_u, decays to a small floor at kappa_u.
        s_plateau = _logistic((self.kappa_u - kappa) / self._w_plateau)
        p_plateau = self._plateau_floor + (1.0 - self._plateau_floor) * s_plateau
        return self.EI0 * ratio * p_plateau

    def _bump_shape(self, kappa: np.ndarray) -> np.ndarray:
        """Non-negative area-correction window: EXACTLY 0 at kappa=0 and kappa=kappa_u, peaked
        between.

        Built as the product of two logistic gates (rises after the knee, falls before kappa_u)
        so it leaves the post-knee/plateau regions shaped but the endpoints untouched. The raw
        product has small but non-zero logistic tails at the endpoints; those tails are removed
        with a smooth (cubic) endpoint taper so the bump cannot leak into the initial tangent
        M'(0)=EI0 or the plateau tangent at kappa_u, regardless of how large the bump
        coefficient becomes.
        """
        kappa = np.asarray(kappa, dtype=float)
        rise = _logistic((kappa - self._bump_lo) / self._w_bump)
        fall = _logistic((self._bump_hi - kappa) / self._w_bump)
        raw = rise * fall
        # Smooth cubic taper that is 0 at both ends of [0, kappa_u] and 1 in the interior,
        # forcing exact endpoint vanishing without introducing a kink.
        t = np.clip(kappa / self.kappa_u, 0.0, 1.0)
        taper = (16.0 * t * t * (1.0 - t) * (1.0 - t))  # peak 1 at t=0.5, 0 at t=0 and t=1
        taper = np.clip(taper, 0.0, 1.0)
        return raw * taper

    def _tangent_shape(self, kappa: np.ndarray) -> np.ndarray:
        """Total tangent EI_tangent(kappa) = base*plateau + c*bump (c precomputed)."""
        return self._base_tangent(kappa) + self._bump_coeff * self._bump_shape(kappa)

    def _build_backbone(self) -> _Backbone:
        kappa = np.linspace(0.0, self.kappa_u, _N_NODES)

        # --- shape-parameter setup (all derived from the 5 section params) ---------------
        # Plateau tangent floor: a small fraction of the post-knee stiffness so dM/dkappa is
        # small (but positive) at kappa_u -> a smooth, monotone approach to the plateau.
        self._plateau_floor = PLATEAU_FLOOR
        # Bump window placement: rise just past the knee, fall just before kappa_u.
        self._bump_lo = min(2.0 * self._kappa_cr, 0.4 * self.kappa_u)
        self._bump_hi = BUMP_HI_FRAC * self.kappa_u
        self._w_bump = BUMP_WIDTH_FRAC * self.kappa_u

        # --- solve the bump coefficient c so the total moment integral equals Mu ----------
        base = self._base_tangent(kappa)
        bump = self._bump_shape(kappa)
        m_base = float(_trapz(base, kappa))
        area_bump = float(_trapz(bump, kappa))
        if area_bump <= 0.0:
            c = 0.0
        else:
            c = (self.Mu - m_base) / area_bump
        # Keep tangent strictly positive everywhere (monotone M): if c<0 would drive the
        # tangent non-positive, clamp it. base is >0 by construction; bump in [0, ~1].
        if c < 0.0:
            bump_max = float(np.max(bump))
            if bump_max > 0.0:
                base_min = float(np.min(base))
                c_floor = -BUMP_FLOOR_FRAC * base_min / bump_max  # tangent stays > 0
                c = max(c, c_floor)
        self._bump_coeff = c

        tangent = self._tangent_shape(kappa)
        # Cumulative trapezoid integral -> M at each node, M(0)=0.
        dk = np.diff(kappa)
        seg = 0.5 * (tangent[1:] + tangent[:-1]) * dk
        moment = np.concatenate([[0.0], np.cumsum(seg)])
        return _Backbone(kappa=kappa, moment=moment, tangent=tangent)

    # ----------------------------------------------------------------- public methods ----
    def moment(self, kappa):
        """Return M(kappa) [N*mm]. Vectorized; clamps to [0, kappa_u] range of the backbone.

        For kappa < 0 the law is extended anti-symmetrically only in the sense M(0)=0 and the
        initial slope EI0; physical use is kappa >= 0. Beyond kappa_u, M is held at Mu (the
        plateau), consistent with a strength-type section-failure cap.
        """
        k = np.asarray(kappa, dtype=float)
        scalar = k.ndim == 0
        kf = np.atleast_1d(k)
        bb = self._backbone
        out = np.interp(kf, bb.kappa, bb.moment, left=bb.moment[0], right=bb.moment[-1])
        # Below zero curvature: linearize through the origin with the initial tangent.
        neg = kf < 0.0
        if np.any(neg):
            out = np.where(neg, self.EI0 * kf, out)
        return float(out[0]) if scalar else out

    def curvature(self, moment):
        """Numerical inverse kappa(M): solve M(kappa) = moment on [0, kappa_u]. Vectorized.

        Moments above the backbone endpoint map to kappa_u for numerical
        residual evaluation. This continuation is not a post-peak section law.
        Negative moments map by the initial tangent.
        """
        m = np.asarray(moment, dtype=float)
        scalar = m.ndim == 0
        mf = np.atleast_1d(m)
        bb = self._backbone
        # bb.moment is monotonically increasing -> invertible by interp(x=M, y=kappa).
        out = np.interp(mf, bb.moment, bb.kappa, left=bb.kappa[0], right=bb.kappa[-1])
        neg = mf < 0.0
        if np.any(neg):
            out = np.where(neg, mf / self.EI0, out)
        return float(out[0]) if scalar else out

    def EI_secant(self, kappa):
        """Secant stiffness M(kappa)/kappa [N*mm^2]. Returns EI0 at kappa -> 0. Vectorized."""
        k = np.asarray(kappa, dtype=float)
        scalar = k.ndim == 0
        kf = np.atleast_1d(k)
        m = np.atleast_1d(self.moment(kf))
        out = np.empty_like(kf)
        small = np.abs(kf) < 1e-12
        out[~small] = m[~small] / kf[~small]
        out[small] = self.EI0  # limit M/kappa -> M'(0) = EI0
        return float(out[0]) if scalar else out

    def EI_tangent(self, kappa):
        """Tangent stiffness dM/dkappa [N*mm^2]. Vectorized.

        Evaluated directly from the (precomputed-amplitude) tangent profile, so it is exactly
        the derivative of ``moment`` up to interpolation of the same nodes.
        """
        k = np.asarray(kappa, dtype=float)
        scalar = k.ndim == 0
        kf = np.atleast_1d(k)
        out = self._tangent_shape(np.clip(kf, 0.0, self.kappa_u))
        # For kappa < 0 the extension is linear with slope EI0; beyond kappa_u hold the
        # (small, plateau-floor) tangent evaluated at kappa_u.
        tan_u = float(self._tangent_shape(np.array([self.kappa_u]))[0])
        out = np.where(kf < 0.0, self.EI0, out)
        out = np.where(kf > self.kappa_u, tan_u, out)
        return float(out[0]) if scalar else out

    def failure(self):
        """Return the section failure pair ``(Mu, kappa_u)``."""
        return (self.Mu, self.kappa_u)

    # ------------------------------------------------------------------------ helpers ----
    def as_array(self) -> np.ndarray:
        """Return the 5 params as an array in SECTION_PARAM_NAMES order."""
        return np.array([self.EI0, self.Mcr, self.r1, self.Mu, self.kappa_u], dtype=float)

    def as_dict(self) -> dict:
        """Return the 5 params as a dict keyed by SECTION_PARAM_NAMES."""
        return {k: float(v) for k, v in zip(SECTION_PARAM_NAMES, self.as_array())}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "SectionLaw("
            f"EI0={self.EI0:.4g}, Mcr={self.Mcr:.4g}, r1={self.r1:.4g}, "
            f"Mu={self.Mu:.4g}, kappa_u={self.kappa_u:.4g})"
        )


def make_section_law(params) -> SectionLaw:
    """Factory: build a :class:`SectionLaw` from a param dict or array.

    Parameters
    ----------
    params : Mapping or array-like
        Either a dict containing all of ``SECTION_PARAM_NAMES`` or an array-like of length 5
        ordered as ``SECTION_PARAM_NAMES``.
    """
    return SectionLaw(params)
