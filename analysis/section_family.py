"""Mechanics-constrained descriptor map for the shared section family.

Member stiffness and capacity depend on eight configuration features.
The transition fraction and tangent ratio are shared; terminal curvature
is determined by the integrated stiffness-to-strength condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from section_backbone import base_grid, solve_endpoint

from analysis.paths import ensure_local_importable

ensure_local_importable()

from cfrpequiv import section_law as _section_law  # noqa: E402
from cfrpequiv.config_features import ConfigFeatures  # noqa: E402
from analysis.elastic_ceiling import elastic_ceiling_Nmm2  # noqa: E402

__all__ = [
    "SECTION_PARAM_NAMES",
    "PRODUCTION_VARIANT",
    "Variant",
    "VARIANTS",
    "feature_vector",
    "feature_names",
    "grid_quantities",
    "kappa_u_from_area_identity",
    "r1_max_feasible",
    "params_for",
    "theta_init",
    "theta_bounds",
    "unpack_theta",
    "free_slots",
]

# --- locked vocabulary (must agree with cfrpequiv.section_law.SECTION_PARAM_NAMES) -----------
# The retained JSON/API key Mcr is the manuscript's transition moment scale Mk.
# Its gate centre is kappa_k = Mk/EI0; the curve ordinate is M(kappa_k), not Mk.
SECTION_PARAM_NAMES = ("EI0", "Mcr", "r1", "Mu", "kappa_u")

# --- descriptor normalization (mirrors config_features.D_REF / D_SCALE) ----------------------
D_REF, D_SCALE = 32.0, 5.0

# --- activation scales (mirror config_features._SOFTPLUS_SCALE / R1_LO / R1_HI) --------------
SOFTPLUS_SCALE = {"EI0": 1.0e9, "Mcr": 1.0e5, "Mu": 5.0e5, "kappa_u": 1.0e-3}
R1_LO, R1_HI = 0.02, 0.98

# --- bias-column targets: a bare ~32 mm square steel tube (config_features._BIAS_BASELINE) ---
BIAS_BASELINE = {"EI0": 3.0e9, "Mcr": 2.0e5, "r1": 0.25, "Mu": 1.3e6, "kappa_u": 2.0e-3}

# --- backbone discretization ------------------------------------------------------------------
# The shape constants of the law FORM belong to cfrpequiv.section_law and are read from it at
# CALL time inside grid_quantities, so a sensitivity sweep entered through
# ``section_law.shape_constants`` moves the kappa_u solve and the delivered law together. The
# names below are import-time snapshots of the same source, kept because analysis.jax_family
# imports them; they cannot drift from section_law, but they do not follow an override.
N_NODES = _section_law._N_NODES
PLATEAU_FLOOR = _section_law.PLATEAU_FLOOR
KNEE_WIDTH_FRAC = _section_law.KNEE_WIDTH_FRAC
PLATEAU_WIDTH_FRAC = _section_law.PLATEAU_WIDTH_FRAC
BUMP_HI_FRAC = _section_law.BUMP_HI_FRAC
BUMP_WIDTH_FRAC = _section_law.BUMP_WIDTH_FRAC
BUMP_FLOOR_FRAC = _section_law.BUMP_FLOOR_FRAC

# --- pre-activation box (mirrors config_features.theta_map_bounds) ---------------------------
PREACT_LO, PREACT_HI = -12.0, 12.0


def softplus(z):
    """Numerically stable ``log(1 + exp(z))`` (config_features._softplus)."""
    return np.logaddexp(0.0, z)


def inv_softplus(y: float) -> float:
    """Inverse of :func:`softplus` for ``y > 0`` (config_features._inv_softplus)."""
    return float(np.log(np.expm1(y)))


def logistic(z):
    """Stable logistic sigmoid, matching ``section_law._logistic``.

    ``config_features._sigmoid`` writes the naive ``1/(1+exp(-z))`` instead. The two agree to
    within one ulp over the pre-activation box, and ``tests/test_regression_v0.py`` is what
    proves the choice does not move the archived fit: if it ever did, the archived condition
    number would not be a reproducible quantity in the first place.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ============================================================================================
# Descriptor features
# ============================================================================================
_FEATURE_NAMES_7 = ("bias", "diameter", "t", "t2", "grout", "doc13", "t_doc13")
_FEATURE_NAMES_8 = ("bias", "diameter", "t", "t2", "grout", "grout_t", "doc13", "t_doc13")

# --- what one nominal millimetre of wrap is made of ------------------------------------------
# The wrap descriptor `t` is a NOMINAL thickness, and the two campaigns' cloth records show it
# does not buy the same laminate on both sides. Per ply, from each campaign's own record:
#
#   doc04  300 g/m^2 Grade-I UNIDIRECTIONAL cloth at 0.167 mm/ply, fibres along the member axis
#   doc13  3K 200 g/m^2 bidirectional PLAIN WEAVE  at 0.300 mm/ply, warp and weft balanced
#
# Two independent factors separate them, and it is worth naming both because only one of them is
# about the weave:
#
#  1. The quoted ply thicknesses are not the same KIND of number. 300 g/m^2 / 1800 kg/m^3 =
#     0.1667 mm and 200 g/m^2 / 1800 kg/m^3 = 0.1111 mm, so both UNIDIRECTIONAL cloths are quoted
#     at their dry fibre-equivalent thickness (areal weight over carbon density). The plain
#     weave's 0.300 mm is a fabric caliper -- 2.70x its own fibre-equivalent 0.111 mm -- because a
#     crimped woven cloth occupies that much space. So a nominal millimetre of weave contains
#     2.70x less fibre of ANY orientation than a nominal millimetre of the UD cloth.
#  2. A balanced plain weave puts half of that fibre around the section rather than along it, so
#     the AXIAL half is halved again.
#
# 2.70 x 2.00 = 5.39, which is the disparity in axial fibre a nominal millimetre delivers. These
# tables carry the per-campaign areal weight per nominal millimetre, normalized to doc04 so the
# doc04 column of the descriptor is numerically unchanged and the rescale is a statement about
# doc13 alone.
_PLY = {                      # (areal weight [g/m^2], nominal ply thickness [mm], axial fraction)
    "doc04": (300.0, 0.167, 1.0),
    "doc13": (200.0, 0.300, 0.5),
}
AXIAL_AREAL_G_M2_PER_MM = {k: w * f / t for k, (w, t, f) in _PLY.items()}
TOTAL_AREAL_G_M2_PER_MM = {k: w / t for k, (w, t, _f) in _PLY.items()}

# Descriptor conventions. "nominal" is the production encoding and leaves `t` untouched.
DESCRIPTOR_SCALES: Dict[str, Dict[str, float]] = {
    "nominal": {k: 1.0 for k in _PLY},
    "axial_areal": {k: v / AXIAL_AREAL_G_M2_PER_MM["doc04"]
                    for k, v in AXIAL_AREAL_G_M2_PER_MM.items()},
    "total_areal": {k: v / TOTAL_AREAL_G_M2_PER_MM["doc04"]
                    for k, v in TOTAL_AREAL_G_M2_PER_MM.items()},
    # Same physical ratio as `axial_areal`, normalized on the SECOND campaign instead of the
    # first. Only the ratio between the two campaigns carries physical content; the normalizer
    # is a global rescale of the descriptor, which the coefficients absorb -- EXCEPT against the
    # +-12 pre-activation box, which is not scale-free. Normalizing on doc04 pushes doc13's
    # column to 0.186 and its square to 0.034, so the convex coefficients must grow to
    # compensate and can rail. This convention pushes them the other way, so the same physical
    # claim is tested at an interior optimum and a bound-constrained arm cannot be mistaken for
    # a verdict about the currency.
    "axial_areal_doc13ref": {k: v / AXIAL_AREAL_G_M2_PER_MM["doc13"]
                             for k, v in AXIAL_AREAL_G_M2_PER_MM.items()},
}


def feature_names(feature_dim: int) -> Tuple[str, ...]:
    """Ordered descriptor names for a 7- or 8-column encoding."""
    if feature_dim == 7:
        return _FEATURE_NAMES_7
    if feature_dim == 8:
        return _FEATURE_NAMES_8
    raise ValueError(f"feature_dim must be 7 or 8, got {feature_dim}")


def feature_vector(
    cfg: ConfigFeatures, feature_dim: int = 8, descriptor: str = "nominal"
) -> np.ndarray:
    """Descriptor vector ``phi(cfg)``.

    ``feature_dim=7`` is the archived encoding (``config_features.feature_vector``).
    ``feature_dim=8`` adds the ``grout x thickness`` column, without which the map cannot
    represent the measured super-additivity: the cement core suppresses the tube-wall local
    buckling that otherwise terminates the wrap's contribution, so the grout benefit is not
    independent of wrap thickness. Measured grout increment on ``P_u`` is 3.74 / 4.89 / 5.20 kN
    at ``t=0`` against 9.86 / 15.22 / 21.15 kN at ``t=1`` (x2.6 / x3.1 / x4.1), while the
    archived 7-column map can only emit a constant ~7.2 kN.

    ``descriptor`` selects what the thickness columns MEASURE (:data:`DESCRIPTOR_SCALES`).
    ``"nominal"`` is the production convention: ``t`` is the nominal wrap thickness in mm on
    both campaigns. ``"axial_areal"`` and ``"total_areal"`` replace it with the axial (resp.
    total) fibre areal weight that nominal thickness delivers in each campaign's own cloth,
    in units of the first campaign's, so a nominal millimetre of doc13 plain weave enters as
    0.186 (resp. 0.371) instead of 1. The rescale is a per-campaign constant, so the columns
    ``t`` and ``t_doc13`` span the same space under every convention and the LINEAR wrap
    response is untouched; what changes is ``t2`` and ``grout_t``, which carry one shared
    coefficient each and therefore transport doc13's measured curvature onto doc04 at a rate
    the convention -- not the data -- fixes.
    """
    doc13 = 1.0 if cfg.system == "doc13" else 0.0
    try:
        scale = DESCRIPTOR_SCALES[descriptor][cfg.system]
    except KeyError as exc:                                   # unknown convention or system
        raise ValueError(f"descriptor {descriptor!r} has no scale for system "
                         f"{cfg.system!r}; known: {sorted(DESCRIPTOR_SCALES)}") from exc
    t = float(cfg.cfrp_thickness_mm) * scale
    g = float(cfg.cement)
    d_hat = (float(cfg.diameter_mm) - D_REF) / D_SCALE
    if feature_dim == 7:
        return np.array([1.0, d_hat, t, t * t, g, doc13, t * doc13], dtype=float)
    if feature_dim == 8:
        return np.array([1.0, d_hat, t, t * t, g, g * t, doc13, t * doc13], dtype=float)
    raise ValueError(f"feature_dim must be 7 or 8, got {feature_dim}")


# ============================================================================================
# Grid quantities: the r1-independent halves of the base area
# ============================================================================================
def grid_quantities(EI0: float, Mcr: float, kappa_u: float):
    """Return ``(B, C, area_bump, max_bump, kappa, s_knee, p_plateau, bump)`` on ``[0, kappa_u]``.

    Mirrors ``section_law._build_backbone`` node-for-node. ``B`` and ``C`` are the two halves of
    the base area that do not depend on ``r1``::

        base(kappa) = EI0 * [r1 + (1-r1)*s_knee] * p_plateau
                    = r1 * [EI0*(1-s_knee)*p_plateau]  +  [EI0*s_knee*p_plateau]

    so ``integral base dkappa = r1*B + C`` exactly. That linearity is what makes both the
    feasible-``r1`` ceiling and the ``kappa_u`` solve closed-form / cheap rather than nested
    root-finds inside every residual evaluation.
    """
    # Read at call time so section_law.shape_constants moves this solve with the law it mirrors.
    knee_frac = _section_law.KNEE_WIDTH_FRAC
    plateau_frac = _section_law.PLATEAU_WIDTH_FRAC
    plateau_floor = _section_law.PLATEAU_FLOOR

    B, C, kappa, s_knee, p_plateau = base_grid(
        EI0, Mcr, kappa_u, n_nodes=N_NODES, knee_width=knee_frac,
        plateau_width=plateau_frac, plateau_floor=plateau_floor)
    kappa_cr = Mcr / EI0

    bump_lo = min(2.0 * kappa_cr, 0.4 * kappa_u)
    bump_hi = _section_law.BUMP_HI_FRAC * kappa_u
    w_bump = _section_law.BUMP_WIDTH_FRAC * kappa_u
    rise = logistic((kappa - bump_lo) / w_bump)
    fall = logistic((bump_hi - kappa) / w_bump)
    tau = np.clip(kappa / kappa_u, 0.0, 1.0)
    taper = np.clip(16.0 * tau * tau * (1.0 - tau) * (1.0 - tau), 0.0, 1.0)
    bump = rise * fall * taper

    return (B, C, float(np.trapezoid(bump, kappa)), float(np.max(bump)),
            kappa, s_knee, p_plateau, bump)


def _base_area(EI0: float, Mcr: float, r1: float, kappa_u: float) -> float:
    """``integral_0^kappa_u base dkappa = r1*B + C`` for a trial ``kappa_u``."""
    B, C, _ab, _mb, _k, _sk, _pp, _b = grid_quantities(EI0, Mcr, kappa_u)
    return r1 * B + C


# ============================================================================================
# The kappa_u solve (the identity that retires the clamp)
# ============================================================================================
def kappa_u_from_area_identity(
    EI0: float, Mcr: float, r1: float, Mu: float, *, max_iter: int = 6
) -> Tuple[float, bool]:
    """Solve ``r1*B(kappa_u) + C(kappa_u) = Mu`` for ``kappa_u``. Returns ``(kappa_u, ok)``.

    The area is strictly increasing in ``kappa_u`` (a longer integration range over a positive
    integrand), so this is a well-posed 1-D monotone root find. Seeded from the large-``kappa_u``
    asymptote ``area ~ r1*EI0*kappa_u*0.95 + Mcr`` and closed by a secant iteration: ~5 area
    evaluations against the ~30 a bisection needs, which matters because this sits inside every
    residual evaluation of a multi-restart fit.

    ``ok=False`` marks a parameter set the law cannot be built from (``Mu`` at or below ``Mcr``,
    or the secant failing to converge); the caller turns that into a large finite residual rather
    than an exception, so the optimizer walks away from it instead of crashing.
    """
    return solve_endpoint(EI0, Mcr, r1, Mu, area=_base_area,
                          seed_fraction=_section_law.BUMP_FLOOR_FRAC,
                          max_iter=max_iter)


def r1_max_feasible(
    EI0: float, Mcr: float, Mu: float, kappa_u: float, gq=None
) -> Tuple[float, bool]:
    """Largest ``r1`` for which ``section_law``'s bump solve stays UNCLAMPED. Closed form.

    Diagnostic only -- the derived-``kappa_u`` family can never clamp, so nothing in the fit
    calls this. It is what makes the clamp story checkable: for the bare 30 mm tube
    ``r1_max = 0.499``, while the archived ridge froze ``r1`` at 0.829 -- 1.66x its feasible
    maximum. That is why the clamp binds there.

    Writing ``base(kappa)/EI0 = r1*u + v`` with ``u = (1-s_knee)*p_plateau`` and
    ``v = s_knee*p_plateau``, the minimum over kappa is attained at the LAST node for every
    ``r1`` in ``(R1_LO, R1_HI)``: ``u`` rises to ``p_plateau(kappa_u) ~ 0.51`` while ``v`` decays
    to ~0, and ``r1 < 1 < 1/0.51``. (Checked over 2000 random parameter draws with zero
    exceptions.) So ``min(base) = EI0*(r1*u[-1] + v[-1])`` is linear in ``r1``, hence

        g(r1) = (Mu - r1*B - C)/area_bump + 0.95*EI0*(r1*u[-1] + v[-1])/max_bump

    is linear too, and ``g(r1_max) = 0`` solves directly. Matches a 60-step bisection to 1.1e-16.
    """
    if gq is None:
        gq = grid_quantities(EI0, Mcr, kappa_u)
    B, C, area_bump, max_bump, _k, s_knee, p_plateau, _b = gq
    u_end = float((1.0 - s_knee[-1]) * p_plateau[-1])
    v_end = float(s_knee[-1] * p_plateau[-1])
    bump_floor = _section_law.BUMP_FLOOR_FRAC
    num = (Mu - C) / area_bump + bump_floor * EI0 * v_end / max_bump
    den = B / area_bump - bump_floor * EI0 * u_end / max_bump
    if den <= 0.0:
        return R1_HI, True          # g is non-decreasing: every r1 is feasible
    r1_max = num / den
    if r1_max >= R1_HI:
        return R1_HI, True          # unconstrained
    if r1_max <= R1_LO:
        return R1_LO, False         # nothing feasible (cannot happen while Mu > Mcr)
    return float(r1_max), True


# ============================================================================================
# Variants
# ============================================================================================
@dataclass(frozen=True)
class Variant:
    """One descriptor-map specification.

    Attributes
    ----------
    rows:
        The section parameters carried as fitted coefficient rows, in order. A parameter absent
        from ``rows`` is derived (only ``kappa_u`` is, and only when ``derive_kappa_u``).
    feature_dim:
        7 (archived encoding) or 8 (with the ``grout x thickness`` column).
    nonneg:
        ``(row_name, feature_name)`` slots bounded ``>= 0``. This is the admissibility
        constraint: it makes the map's response to a descriptor -- not just each emitted law --
        obey composite-member mechanics.
    bias_only_rows:
        Rows whose non-bias coefficients are REMOVED from the parameter vector (not penalized
        toward zero). A row listed here is asserted config-independent.
    drop_slots:
        Individual ``(row_name, feature_name)`` coefficients REMOVED from the parameter vector,
        the per-coefficient counterpart of ``bias_only_rows``. This is what makes a nested-map
        comparison possible: a restricted map that drops one descriptor effect is the same
        variant with one slot fewer, so its fit, its condition number and its held-out score are
        produced by the identical harness and differ only in the restriction. Removal is again
        outright rather than a pin to a near-zero box, for the reason ``free_slots`` states. A
        dropped slot may not also appear in ``nonneg`` -- ``theta_bounds`` raises if it does,
        since an admissibility floor on a coefficient that is not fitted is a contradiction.
    ridge_lambda / ridge_rows:
        L2 penalty on the non-bias coefficients of the named rows, appended last in the residual.
        Retained only to reproduce the archived baseline; the live variants set it to zero.
    derive_kappa_u:
        Solve ``kappa_u`` from the area identity instead of fitting it.
    mcr_as_fraction_of_Mu:
        Activate the knee row as ``Mcr = Mu * sigmoid(z)`` instead of ``1e5 * softplus(z)``, so
        the law's constructibility requirement ``Mu > Mcr`` holds by construction rather than by
        a soft hinge that the optimizer can walk through. See :func:`_activate_mcr`.
    ceiling_moduli:
        Overrides for the elastic ceiling's constituent moduli, as ``(name, value)`` pairs
        (:func:`analysis.elastic_ceiling.resolve_moduli`). Empty means the documented defaults,
        which is what every live variant carries. It lives on the variant rather than in a
        module global because the restart pool runs in separate processes: a global set in the
        parent would be ignored in every worker, and the fit would report the default ceiling
        under a swept name.
    descriptor:
        Which wrap-descriptor convention the thickness columns carry
        (:data:`DESCRIPTOR_SCALES`). ``"nominal"`` is the production convention. The
        alternatives rescale the descriptor per campaign by what a nominal millimetre of that
        campaign's cloth actually delivers, which leaves the linear wrap response invariant and
        changes only what ``t2`` and ``grout_t`` transport from one campaign to the other.
    ei0_ceiling_penalty:
        Weight of a one-sided admissibility penalty ``w * relu(EI0(cfg)/EI_ceiling(cfg) - 1)``
        appended to the residual for every configuration, where ``EI_ceiling`` is the section's
        full-composite-action elastic stiffness (:mod:`analysis.elastic_ceiling`). It caps the
        identified elastic stiffness at the largest value the section's own materials and
        geometry can supply: the measured low-load slope of the deep first-campaign tubes reads
        stiffer than a bare steel section can bend, and an unbounded softplus ``EI0`` reproduced
        that faithfully as an unphysical value (worst +36% over the pure-steel ceiling on the
        shared-map bare 35 mm cell; +77% on its free per-cell fit). ``EI0`` stays a well-behaved
        ``1e9*softplus(z)`` -- a hard ``EI_ceiling*sigmoid(z)`` reparameterization saturates
        where the data wants the ceiling (a bare tube's composite-action efficiency is exactly 1)
        and rails the coefficient, destroying the condition number; the one-sided penalty holds
        the cap without a null Jacobian direction. Zero (the default) leaves the residual exactly
        the archived stacked form, so the ``archived_v0`` regression fixture is untouched. The
        cap binds on the bare and cement-only deep cells and stays slack on the CFRP-wrapped
        cells, whose measured stiffness the wrap justifies. ``P_u = 4*Mu/L`` is independent of
        ``EI0``, so the calibration and held-out peak errors are preserved; the price is a larger
        low-load curve-shape residual on the capped cells.
    """

    name: str
    rows: Tuple[str, ...]
    feature_dim: int
    nonneg: Tuple[Tuple[str, str], ...] = ()
    bias_only_rows: Tuple[str, ...] = ()
    drop_slots: Tuple[Tuple[str, str], ...] = ()
    ridge_lambda: float = 0.0
    ridge_rows: Tuple[str, ...] = ()
    derive_kappa_u: bool = False
    mcr_as_fraction_of_Mu: bool = False
    ei0_ceiling_penalty: float = 0.0
    descriptor: str = "nominal"
    ceiling_moduli: Tuple[Tuple[str, float], ...] = ()
    note: str = ""


# The grout admissibility encoding: the cement infill may not make the section softer or weaker,
# at any wrap thickness. Non-negative coefficients on BOTH grout columns of the EI0 and Mu rows
# give exactly that, because phi[grout] = g and phi[grout_t] = g*t are non-negative and softplus
# is monotone: EI0(g=1) - EI0(g=0) >= 0 for every t >= 0.
_GROUT_NONNEG = (("EI0", "grout"), ("EI0", "grout_t"), ("Mu", "grout"), ("Mu", "grout_t"))

PRODUCTION_VARIANT = "derived_ku_fixed_knee_mcr_frac_eiceiling"
VARIANTS = {PRODUCTION_VARIANT: Variant(
        name="derived_ku_fixed_knee_mcr_frac_eiceiling",
        rows=("EI0", "Mcr", "r1", "Mu"),
        feature_dim=8,
        nonneg=_GROUT_NONNEG,
        bias_only_rows=("Mcr", "r1"),
        ridge_lambda=0.0,
        derive_kappa_u=True,
        mcr_as_fraction_of_Mu=True,
        ei0_ceiling_penalty=20.0,
        note=(
            "Production family with an integral endpoint, shared transition fraction and "
            "shared tangent ratio. A one-sided penalty on EI0 uses the full-composite "
            "elastic ceiling from analysis.elastic_ceiling. The observed low-load slopes "
            "that exceed this ceiling retain a curve residual in the constrained fit. "
            "Peak conversion is P_u = 4*Mu/L; the coefficients are fitted jointly to "
            "curves, capacities and mechanical constraints. Reserved scores were visible "
            "during map development."
        ),
    )}


def free_slots(variant: Variant) -> Tuple[Tuple[int, int], ...]:
    """The ``(row_index, feature_index)`` slots this variant actually fits.

    A ``bias_only`` row contributes its bias slot alone, so the parameter count -- and therefore
    the Jacobian's column count and the reported condition number -- reflects the real degrees of
    freedom. Pinning a coefficient to a degenerate ``[-1e-12, 1e-12]`` box instead would leave a
    null column in the Jacobian and report an identifiability that is an artifact of the pinning.
    """
    names = feature_names(variant.feature_dim)
    dropped = set()
    for row_name, feat_name in variant.drop_slots:
        if row_name not in variant.rows:
            raise ValueError(f"{variant.name}: drop_slots names row {row_name!r}, not in rows")
        if feat_name not in names:
            raise ValueError(f"{variant.name}: drop_slots names feature {feat_name!r}, "
                             f"not in the {variant.feature_dim}-column encoding")
        dropped.add((variant.rows.index(row_name), names.index(feat_name)))
    slots = []
    for i, row in enumerate(variant.rows):
        cols = (0,) if row in variant.bias_only_rows else range(variant.feature_dim)
        for c in cols:
            if (i, c) not in dropped:
                slots.append((i, c))
    return tuple(slots)


def unpack_theta(variant: Variant, theta: Sequence[float]) -> np.ndarray:
    """Flat free parameters -> the full ``(len(rows), feature_dim)`` coefficient matrix."""
    A = np.zeros((len(variant.rows), variant.feature_dim), dtype=float)
    for k, (i, c) in enumerate(free_slots(variant)):
        A[i, c] = theta[k]
    return A


def theta_bounds(variant: Variant) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-activation box, with the admissibility slots floored at zero."""
    slots = free_slots(variant)
    lo = np.full(len(slots), PREACT_LO, dtype=float)
    hi = np.full(len(slots), PREACT_HI, dtype=float)
    names = feature_names(variant.feature_dim)
    index = {slot: k for k, slot in enumerate(slots)}
    for row_name, feat_name in variant.nonneg:
        i = variant.rows.index(row_name)
        c = names.index(feat_name)
        if (i, c) not in index:
            raise ValueError(
                f"{variant.name}: admissibility slot ({row_name}, {feat_name}) is not fitted"
            )
        lo[index[(i, c)]] = 0.0
    return lo, hi


def _logit(u: float) -> float:
    return float(np.log(u / (1.0 - u)))


def theta_init(variant: Variant, seed: int, perturb: float = 0.05) -> np.ndarray:
    """Seeded start: bias columns at the physical baseline, others small random.

    The random draw is taken over the FULL coefficient matrix and only then restricted to the
    free slots, so a ``bias_only`` variant consumes the same random stream as the full one and
    the seeds stay comparable across variants.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((len(variant.rows), variant.feature_dim), dtype=float)
    for i, name in enumerate(variant.rows):
        if name == "r1":
            A[i, 0] = _logit((BIAS_BASELINE[name] - R1_LO) / (R1_HI - R1_LO))
        elif name == "Mcr" and variant.mcr_as_fraction_of_Mu:
            A[i, 0] = _logit(BIAS_BASELINE["Mcr"] / BIAS_BASELINE["Mu"])
        else:
            A[i, 0] = inv_softplus(BIAS_BASELINE[name] / SOFTPLUS_SCALE[name])
    A[:, 1:] = perturb * rng.standard_normal((len(variant.rows), variant.feature_dim - 1))
    return np.array([A[i, c] for (i, c) in free_slots(variant)], dtype=float)


# ============================================================================================
# The map
# ============================================================================================
def _activate(row_name: str, z: float) -> float:
    """Pre-activation -> physical parameter (config_features._activate)."""
    if row_name == "r1":
        return R1_LO + (R1_HI - R1_LO) * float(logistic(np.asarray(z)))
    return SOFTPLUS_SCALE[row_name] * float(softplus(z))


def params_for(
    variant: Variant, theta: Sequence[float], cfg: ConfigFeatures
) -> Tuple[Dict[str, float], bool]:
    """Descriptors -> the 5 section-law parameters. Returns ``(params, ok)``.

    ``ok=False`` marks a parameter set from which ``make_section_law`` cannot build an orderable
    law; the residual turns it into a large finite penalty. Under ``mcr_as_fraction_of_Mu`` the
    only way to reach ``ok=False`` is the ``kappa_u`` solve failing, because ``Mu > Mcr`` then
    holds identically.
    """
    A = unpack_theta(variant, theta)
    z = A @ feature_vector(cfg, variant.feature_dim, variant.descriptor)
    params: Dict[str, float] = {}
    for i, name in enumerate(variant.rows):
        if name == "Mcr" and variant.mcr_as_fraction_of_Mu:
            continue  # needs Mu; filled in below
        params[name] = _activate(name, z[i])
    if variant.mcr_as_fraction_of_Mu:
        i = variant.rows.index("Mcr")
        params["Mcr"] = params["Mu"] * float(logistic(np.asarray(z[i])))

    if variant.derive_kappa_u:
        params["kappa_u"], ok = kappa_u_from_area_identity(
            params["EI0"], params["Mcr"], params["r1"], params["Mu"]
        )
    else:
        ok = True
    return params, ok
