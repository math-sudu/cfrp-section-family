"""Concentric-section elastic stiffness used by the one-sided fit penalty.

The reference sums E*I for the steel annulus, cement core and nominal
wrap annulus. Identification uses constituent moduli of 200, 25 and
241 GPa, respectively, with a 2 mm tube wall. These are model inputs:
the cement modulus was not measured; the cloth modulus is the first
campaign's supplier value, carried across both campaigns. The nominal
wrap annulus does not resolve the second campaign's balanced weave."""

from __future__ import annotations

from analysis.paths import ensure_local_importable

ensure_local_importable()

from cfrpequiv.config_features import ConfigFeatures  # noqa: E402

# Constituent moduli in MPa; wall thickness in mm.
E_STEEL_MPA = 200.0e3
E_CEMENT_MPA = 25.0e3
E_CFRP_MPA = 241.0e3
WALL_MM = 2.0

# Axial fibre per nominal millimetre, first campaign over second (analysis.section_family's
# ply records). The campaign-corrected cloth arm divides the second campaign's modulus by it.
_AXIAL_FIBRE_RATIO = 5.389221556886226

DEFAULT_MODULI = {
    "E_steel_MPa": E_STEEL_MPA,
    "E_cement_MPa": E_CEMENT_MPA,
    "E_cfrp_MPa": E_CFRP_MPA,
    "E_cfrp_MPa_doc13": E_CFRP_MPA,     # the carried value; the corrected arm lowers this one
}


def resolve_moduli(moduli=None) -> dict:
    """Merge an override onto :data:`DEFAULT_MODULI`.

    ``moduli`` is a mapping or a tuple of ``(name, value)`` pairs -- the tuple form is what a
    ``Variant`` carries, so a swept ceiling survives the restart pool's pickling. A module-level
    global would be set in the parent process and silently ignored in every worker, which is a
    wrong answer rather than an error.
    """
    out = dict(DEFAULT_MODULI)
    if not moduli:
        return out
    items = moduli.items() if hasattr(moduli, "items") else moduli
    for k, v in items:
        if k not in DEFAULT_MODULI:
            raise ValueError(f"unknown ceiling modulus {k!r}; known: {sorted(DEFAULT_MODULI)}")
        out[k] = float(v)
    return out


def _I_square(outer: float, inner: float) -> float:
    """Second moment of a square annulus (outer, inner edge) about its centroid, mm^4."""
    return (outer**4 - inner**4) / 12.0


def steel_EI_Nmm2(d_mm: float, moduli=None) -> float:
    """Bare steel hollow-square-tube elastic flexural stiffness, N.mm^2."""
    return resolve_moduli(moduli)["E_steel_MPa"] * _I_square(d_mm, d_mm - 2.0 * WALL_MM)


def elastic_ceiling_Nmm2(cfg: ConfigFeatures, moduli=None) -> float:
    """Full-composite-action elastic flexural stiffness ceiling for ``cfg``, N.mm^2.

    ``EI_ceiling = E_s*I_steel + [cement] E_c*I_core + [wrap] E_cfrp*I_wrap``, every term
    about the common centroid. This is the largest elastic stiffness the section's
    materials and geometry can supply (perfect bond, plane sections, fully effective
    wrap); the identified ``EI0`` is expressed as a fraction of it.
    """
    m = resolve_moduli(moduli)
    d = float(cfg.diameter_mm)
    inner = d - 2.0 * WALL_MM
    ei = m["E_steel_MPa"] * _I_square(d, inner)
    if int(cfg.cement) == 1:
        ei += m["E_cement_MPa"] * (inner**4) / 12.0
    t_w = float(cfg.cfrp_thickness_mm)
    if t_w > 0.0:
        e_cfrp = m["E_cfrp_MPa_doc13"] if cfg.system == "doc13" else m["E_cfrp_MPa"]
        ei += e_cfrp * _I_square(d + 2.0 * t_w, d)
    return float(ei)


def elastic_ceiling_kNm2(cfg: ConfigFeatures, moduli=None) -> float:
    """Convenience: ceiling in kN.m^2 (1e9 N.mm^2 == 1 kN.m^2)."""
    return elastic_ceiling_Nmm2(cfg, moduli) / 1.0e9


if __name__ == "__main__":
    # Self-check: bare tubes must reproduce the pure-steel ceiling exactly.
    for d in (30.0, 32.0, 35.0):
        cfg = ConfigFeatures(diameter_mm=d, cfrp_thickness_mm=0.0, cement=0, system="doc04")
        print(f"bare {d:.0f}mm  ceiling = {elastic_ceiling_kNm2(cfg):6.3f} kN.m^2  "
              f"(steel E*I = {steel_EI_Nmm2(d)/1e9:6.3f})")
