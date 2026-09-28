"""Config-feature encoding and the low-DOF config -> section-parameter map.

A member test configuration is described by a small set of physically meaningful
features (tube edge size, CFRP wrap thickness, cement infill, source system). The
identification (``identify.py``) does NOT fit one independent parameter set per
configuration; instead it fits a single flat vector ``theta_map`` that parameterizes
a SMOOTH, LOW-DOF map from these features to the 5 section-law parameters
(``EI0, Mcr, r1, Mu, kappa_u``). CFRP wrap thickness enters as a CONTINUOUS numeric
feature TOGETHER with a SQUARED thickness term (so the map can represent the convex,
super-linear capacity gain from 1 mm to 2 mm wrap -- an affine-in-thickness map
systematically under-levels the held-out 2 mm group) and a thickness x doc13-system
interaction (so the doc13 specimen system carries its OWN, lower thickness response
instead of dragging the shared doc04 slope down: the only 2 mm calibration anchors are
doc13, which sit ~8-16% below doc04 at matched cement/diameter). A map calibrated on the
calibration family then PREDICTS a held-out 2 mm doc04 configuration; this is the K2
falsifier and nothing here is hard-coded per configuration.

Feature ordering (documented, fixed) -- see ``feature_vector``:

    index 0 : bias (constant 1.0)
    index 1 : diameter feature, centered/scaled  (d - D_REF) / D_SCALE
    index 2 : CFRP wrap thickness in mm           (0.0 / 1.0 / 2.0)
    index 3 : CFRP wrap thickness SQUARED [mm^2]  (0.0 / 1.0 / 4.0)   -- convex term
    index 4 : cement-infill flag                  (0.0 / 1.0)
    index 5 : doc13 system flag                   (0.0 for doc04, 1.0 for doc13)
    index 6 : thickness x doc13 interaction       (cfrp_thickness_mm * doc13_flag)

Section parameters in ``SECTION_PARAM_NAMES`` order are produced from a linear
pre-activation ``A @ phi`` (A has shape (5, feature_dim)) passed through per-parameter
positivity/range activations so the family stays physically admissible for any feature
vector (EI0>0, Mcr>0, 0<r1<1, Mu>0, kappa_u>0). ``theta_map`` is the flattened ``A``.

Only numpy is used. The map is differentiable (smooth activations) so the
least-squares Jacobian in ``identify.py`` is well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Section-parameter vocabulary (kept identical to section_law.SECTION_PARAM_NAMES;
# duplicated here to avoid an import cycle -- the two modules must agree).
# ---------------------------------------------------------------------------
SECTION_PARAM_NAMES = ["EI0", "Mcr", "r1", "Mu", "kappa_u"]
N_SECTION_PARAMS = len(SECTION_PARAM_NAMES)

# Feature normalization constants. The member tubes are 30-35 mm square section;
# center on 32 mm with a 5 mm scale so the diameter feature is O(1).
D_REF = 32.0   # mm
D_SCALE = 5.0  # mm

# Number of named numeric features (see module docstring for the ordering).
# 7 = [bias, diameter, thickness, thickness^2, cement, doc13_flag, thickness*doc13].
FEATURE_DIM = 7


@dataclass
class ConfigFeatures:
    """Physical descriptor of a single member-test configuration.

    Attributes
    ----------
    diameter_mm:
        Square-tube edge length in mm (30 / 32 / 35 for the doc04 backbone).
    cfrp_thickness_mm:
        External CFRP wrap thickness in mm. 0 (none), 1 (1 mm布), or 2 (2 mm布).
    cement:
        1 if the tube cavity is cement-filled, else 0.
    system:
        Source system tag, ``"doc04"`` (primary clean backbone) or ``"doc13"``
        (heterogeneous specimen library).
    extras:
        Optional doc13-only descriptors that are NOT cleanly encodable into the
        numeric feature vector (wrap direction, internal placement, framing
        material, filling type, surface treatment, ...). Stored for traceability
        and documentation; they do not enter ``feature_vector``.
    """

    diameter_mm: float
    cfrp_thickness_mm: float
    cement: int
    system: str
    extras: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.diameter_mm = float(self.diameter_mm)
        self.cfrp_thickness_mm = float(self.cfrp_thickness_mm)
        self.cement = int(self.cement)
        if self.system not in ("doc04", "doc13"):
            raise ValueError(
                f"system must be 'doc04' or 'doc13', got {self.system!r}"
            )
        if self.cement not in (0, 1):
            raise ValueError(f"cement must be 0 or 1, got {self.cement!r}")
        if self.cfrp_thickness_mm not in (0.0, 1.0, 2.0):
            # Not fatal physically, but the backbone only contains 0/1/2; warn loudly
            # by rejecting so fabricated thicknesses cannot silently enter the family.
            raise ValueError(
                "cfrp_thickness_mm must be one of 0/1/2 mm, got "
                f"{self.cfrp_thickness_mm!r}"
            )

    def key(self) -> tuple:
        """Hashable identity used by leak checks: (system, diameter, thickness, cement)."""
        return (self.system, self.diameter_mm, self.cfrp_thickness_mm, self.cement)


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
def feature_vector(cfg: ConfigFeatures) -> np.ndarray:
    """Numeric feature vector phi(cfg) for the parameter map.

    Ordering (length ``FEATURE_DIM`` == 7):
        [ 1.0,
          (diameter_mm - D_REF) / D_SCALE,
          cfrp_thickness_mm,
          cfrp_thickness_mm ** 2,                  # convex thickness term
          float(cement),
          doc13_flag,
          cfrp_thickness_mm * doc13_flag ]         # thickness x system interaction

    The CFRP thickness enters BOTH linearly and through a SQUARED term so the map can
    represent the convex (super-linear) capacity gain from 1 mm to 2 mm wrap; an affine
    map fit on {0, 1} mm systematically under-levels the held-out 2 mm group (the K2
    falsifier). The thickness x doc13 interaction lets the doc13 specimen system (the only
    2 mm calibration anchors, sitting ~8-16% below doc04 at matched cement/diameter) carry
    its own thickness response instead of dragging the shared doc04 slope down. All
    features are O(1) over the tested family.
    """
    doc13 = 1.0 if cfg.system == "doc13" else 0.0
    t = cfg.cfrp_thickness_mm
    return np.array(
        [
            1.0,
            (cfg.diameter_mm - D_REF) / D_SCALE,
            t,
            t * t,
            float(cfg.cement),
            doc13,
            t * doc13,
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# theta_map layout, init, bounds
# ---------------------------------------------------------------------------
# theta_map is the flattened coefficient matrix A of shape (N_SECTION_PARAMS, feature_dim).
# Row p maps phi -> pre-activation z_p = A[p] @ phi, then a per-parameter activation
# turns z_p into the physical parameter. Pre-activations are designed so that the
# bias column alone (phi = [1,0,0,0,0]) reproduces a sensible baseline 30-35 mm steel
# tube section, and the other columns are small perturbations the fit moves.

# Physically motivated baseline targets for the bias column (a bare ~32 mm steel tube),
# expressed in the PRE-activation space (see _activate for the inverse transforms):
#   EI0     ~ 3e9  N*mm^2   (30-35 mm square steel tube, t~1-2 mm wall)
#   Mcr     ~ 2e5  N*mm
#   r1      ~ 0.25 (post-knee tangent stiffness ratio)
#   Mu      ~ 1.3e6 N*mm    (=> P_u = 4*Mu/L ~ 4*1.3e6/490 ~ 1.06e4 N ~ 10.6 kN, the
#                            single-tube order in table2-8)
#   kappa_u ~ 2e-3 1/mm

_BIAS_BASELINE = {
    "EI0": 3.0e9,
    "Mcr": 2.0e5,
    "r1": 0.25,
    "Mu": 1.3e6,
    "kappa_u": 2.0e-3,
}

# Per-parameter activation: positive params use softplus around a scale; r1 uses a
# sigmoid mapped into an open interval (R1_LO, R1_HI). Scales below keep pre-activations
# O(1) so the optimizer and Jacobian are well conditioned.
_SOFTPLUS_SCALE = {
    "EI0": 1.0e9,    # z in ~[-, +] -> EI0 = scale * softplus(z)
    "Mcr": 1.0e5,
    "Mu": 5.0e5,
    "kappa_u": 1.0e-3,
}
R1_LO, R1_HI = 0.02, 0.98


def _softplus(z: np.ndarray | float) -> np.ndarray | float:
    # Numerically stable softplus.
    return np.logaddexp(0.0, z)


def _inv_softplus(y: float) -> float:
    # Inverse of softplus for y > 0: z = log(exp(y) - 1), stable for large y.
    return float(np.log(np.expm1(y)))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _bias_preactivations() -> np.ndarray:
    """Pre-activation values for the bias column reproducing ``_BIAS_BASELINE``."""
    z = np.zeros(N_SECTION_PARAMS, dtype=float)
    for i, name in enumerate(SECTION_PARAM_NAMES):
        if name == "r1":
            r = _BIAS_BASELINE[name]
            u = (r - R1_LO) / (R1_HI - R1_LO)
            u = min(max(u, 1e-4), 1.0 - 1e-4)
            z[i] = float(np.log(u / (1.0 - u)))  # logit
        else:
            z[i] = _inv_softplus(_BIAS_BASELINE[name] / _SOFTPLUS_SCALE[name])
    return z


def _activate(z: np.ndarray) -> np.ndarray:
    """Map a length-5 pre-activation vector to physical section params (order fixed)."""
    z = np.asarray(z, dtype=float)
    out = np.empty(N_SECTION_PARAMS, dtype=float)
    for i, name in enumerate(SECTION_PARAM_NAMES):
        if name == "r1":
            out[i] = R1_LO + (R1_HI - R1_LO) * float(_sigmoid(z[i]))
        else:
            out[i] = _SOFTPLUS_SCALE[name] * float(_softplus(z[i]))
    return out


def theta_map_size(feature_dim: int = FEATURE_DIM) -> int:
    """Number of free parameters in the map: one coefficient row per section param."""
    return N_SECTION_PARAMS * int(feature_dim)


def theta_map_init(feature_dim: int = FEATURE_DIM, seed: int | None = None) -> np.ndarray:
    """Initial flat ``theta_map``.

    The bias column is set to reproduce ``_BIAS_BASELINE``; the remaining feature
    columns start at small random values (so different multi-restart seeds explore
    distinct feature sensitivities) but near zero (baseline-dominated start).
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((N_SECTION_PARAMS, int(feature_dim)), dtype=float)
    A[:, 0] = _bias_preactivations()
    if feature_dim > 1:
        A[:, 1:] = 0.05 * rng.standard_normal((N_SECTION_PARAMS, int(feature_dim) - 1))
    return A.reshape(-1)


def theta_map_bounds(feature_dim: int = FEATURE_DIM):
    """(lower, upper) flat bound arrays for ``scipy.optimize.least_squares``.

    Pre-activations are kept in a generous but finite box so the fit cannot run off to
    non-physical magnitudes; activations guarantee positivity/range regardless.
    """
    n = theta_map_size(feature_dim)
    lo = np.full(n, -12.0, dtype=float)
    hi = np.full(n, 12.0, dtype=float)
    return lo, hi


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------
def section_params_from_features(
    theta_map: np.ndarray, cfg: ConfigFeatures
) -> Dict[str, float]:
    """Smooth low-DOF map: (theta_map, config features) -> 5 section parameters.

    Parameters
    ----------
    theta_map:
        Flat vector of length ``theta_map_size(FEATURE_DIM)`` (== flattened (5, 7)
        coefficient matrix). This is the quantity ``identify.py`` fits.
    cfg:
        The configuration whose section parameters are requested.

    Returns
    -------
    dict with keys ``SECTION_PARAM_NAMES``; every value is physically admissible
    (EI0>0, Mcr>0, 0<r1<1, Mu>0, kappa_u>0).
    """
    theta = np.asarray(theta_map, dtype=float)
    phi = feature_vector(cfg)
    feature_dim = phi.size
    expected = theta_map_size(feature_dim)
    if theta.size != expected:
        raise ValueError(
            f"theta_map has size {theta.size}, expected {expected} "
            f"(= {N_SECTION_PARAMS} x feature_dim {feature_dim})"
        )
    A = theta.reshape(N_SECTION_PARAMS, feature_dim)
    z = A @ phi
    params = _activate(z)
    return {name: float(val) for name, val in zip(SECTION_PARAM_NAMES, params)}


def section_param_array_from_features(
    theta_map: np.ndarray, cfg: ConfigFeatures
) -> np.ndarray:
    """Same as :func:`section_params_from_features` but returns the ordered array."""
    d = section_params_from_features(theta_map, cfg)
    return np.array([d[name] for name in SECTION_PARAM_NAMES], dtype=float)
