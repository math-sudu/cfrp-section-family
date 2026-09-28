"""Read the bundled digitized member curves and tabulated peak loads.

Campaign identifiers doc04 and doc13 denote campaigns 1 and 2. Use
analysis.member_data.load_partitions for the paper's 18/4 partition;
it assigns the second-campaign roles by specimen code."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cfrpequiv.config_features import ConfigFeatures

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
# This file lives at <root>/src/cfrpequiv/data_io.py -> parents[2] is <root>.
ROOT = Path(__file__).resolve().parents[2]

DOC04_DIR = "data/members/campaign1"
DOC13_DIR = "data/members/campaign2"

DOC04_CURVES = "load_deflection_curves.csv"
DOC04_PEAKS_1MM = "table2-8_peak_load_summary_kn.csv"
DOC04_PEAKS_2MM = "table4-1_test_vs_sim_2mm_peak_load_kn.csv"

DOC13_CURVES = "pressure_deflection.csv"
DOC13_CAPACITY = "ultimate_bearing_capacity.csv"
DOC13_MATRIX = "specimen_matrix.csv"

CLEAR_SPAN_MM = 490.0  # informational; the span used by the forward model lives in config.

# ---------------------------------------------------------------------------
# doc04 composite_type -> (cfrp_thickness_mm, cement) and the zh peak-table labels.
# ---------------------------------------------------------------------------
DOC04_COMPOSITE_TYPES = (
    "single_tube",
    "tube_cement",
    "tube_cfrp1mm",
    "tube_cement_cfrp1mm",
    "tube_cfrp2mm",
    "tube_cement_cfrp2mm",
)

# Calibration vs held-out composite types (diameter restriction applied separately).
CALIBRATION_COMPOSITES = ("single_tube", "tube_cement", "tube_cfrp1mm", "tube_cement_cfrp1mm")
HELD_OUT_COMPOSITES = ("tube_cfrp2mm", "tube_cement_cfrp2mm")
HELD_OUT_DIAMETERS = (30.0, 32.0)  # 2 mm CFRP curves/peaks exist only for 30/32 mm.

# Chinese row labels in table2-8 (1 mm family) -> doc04 composite_type.
ZH_LABEL_1MM = {
    "单钢管": "single_tube",
    "钢管+水泥": "tube_cement",
    "钢管+CFRP布": "tube_cfrp1mm",
    "钢管+水泥+CFRP布": "tube_cement_cfrp1mm",
}
# Chinese row labels in table4-1 (held-out 2 mm) -> doc04 composite_type.
ZH_LABEL_2MM = {
    "钢管+2mmCFRP布": "tube_cfrp2mm",
    "钢管+水泥+2mmCFRP布": "tube_cement_cfrp2mm",
}


def _cfrp_thickness_of(composite_type: str) -> float:
    """0 for single/cement, 1 for *cfrp1mm, 2 for *cfrp2mm."""
    if "cfrp2mm" in composite_type:
        return 2.0
    if "cfrp1mm" in composite_type:
        return 1.0
    return 0.0


def _cement_of(composite_type: str) -> int:
    """1 if cement appears in the composite type, else 0."""
    return 1 if "cement" in composite_type else 0


# ---------------------------------------------------------------------------
# MemberCurve container
# ---------------------------------------------------------------------------
@dataclass
class MemberCurve:
    """One measured three-point-bending member response plus its config + peak.

    Attributes
    ----------
    config:
        :class:`ConfigFeatures` descriptor used by the parameter map.
    deflection_mm, load_kN:
        Equal-length sorted arrays of the measured P-delta curve (load in kN, as
        stored; the forward model converts kN -> N at use time).
    peak_kN:
        Reported ultimate / peak load for this configuration (kN). For doc04 this is
        the reported table peak; for doc13 it is the curated ultimate capacity when
        available, else the curve maximum (``peak_source`` records which).
    source_id:
        Provenance tag, e.g. ``"doc04:tube_cfrp1mm@30mm"`` or ``"doc13:C-3"``.
    composite_type:
        doc04 composite type, or the doc13 specimen id for doc13 members.
    diameter_mm:
        Tube edge length in mm.
    peak_source:
        ``"table"`` / ``"capacity_csv"`` / ``"curve_max"`` / ``"none"`` -- where
        ``peak_kN`` came from (None encoded as NaN when truly unavailable).
    """

    config: ConfigFeatures
    deflection_mm: np.ndarray
    load_kN: np.ndarray
    peak_kN: float
    source_id: str
    composite_type: str
    diameter_mm: float
    peak_source: str = "none"

    def partition_key(self) -> Tuple[str, float]:
        """Leak-check identity at the (composite_type, diameter) granularity."""
        return (self.composite_type, self.diameter_mm)


# ---------------------------------------------------------------------------
# Low-level CSV helpers
# ---------------------------------------------------------------------------
def _read_rows(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"curated CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(s: str) -> float:
    return float(str(s).strip())


def _sorted_curve(defl: List[float], load: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Sort a (deflection, load) point cloud by deflection ascending."""
    d = np.asarray(defl, dtype=float)
    p = np.asarray(load, dtype=float)
    order = np.argsort(d, kind="stable")
    return d[order], p[order]


def _truncate_to_loading_branch(mc: "MemberCurve") -> None:
    """Trim ``mc`` to its loading branch (points up to peak load), IN PLACE.

    The equivalent-section law is a MONOTONE moment-curvature backbone with a strength
    plateau: it represents only the ascending load-deflection envelope up to section
    failure. The measured curves continue past the peak into the post-peak softening branch
    (load decreasing while mid-span deflection keeps growing), which the monotone law cannot
    represent and which lies OUTSIDE the method's stated domain. Those points are dropped so
    the fit and the curve metric reflect the model's valid regime. ``peak_kN`` (the reported
    table peak) is unchanged. Curves are sorted by deflection ascending, so the loading branch
    is the prefix up to and including the peak-load index.
    """
    load = np.asarray(mc.load_kN, dtype=float)
    if load.size <= 2:
        return
    i = int(np.argmax(load))
    if i < 1:  # degenerate (peak at the very start) -> keep as-is
        return
    mc.deflection_mm = np.asarray(mc.deflection_mm, dtype=float)[: i + 1]
    mc.load_kN = load[: i + 1]


# ---------------------------------------------------------------------------
# doc04 readers
# ---------------------------------------------------------------------------
def _read_doc04_peaks(data_root: Path) -> Dict[Tuple[str, float], float]:
    """Return {(composite_type, diameter_mm): peak_kN} from both doc04 peak tables.

    table2-8 supplies the 1 mm-family calibration peaks at 30/32/35 mm.
    table4-1 supplies the held-out 2 mm peaks at 30/32 mm using the 试验 rows only.
    """
    peaks: Dict[Tuple[str, float], float] = {}

    # --- 1 mm family (calibration) ---
    p1 = data_root / DOC04_DIR / DOC04_PEAKS_1MM
    for row in _read_rows(p1):
        label = row["composite_type_zh"].strip()
        ctype = ZH_LABEL_1MM.get(label)
        if ctype is None:
            # Unknown label -> document by skipping; do not invent a mapping.
            continue
        for col, dia in (("30mm_kn", 30.0), ("32mm_kn", 32.0), ("35mm_kn", 35.0)):
            val = row.get(col, "").strip()
            if val:
                peaks[(ctype, dia)] = _float(val)

    # --- 2 mm family (held-out); 试验 (test) rows only, 模拟 (sim) excluded ---
    p2 = data_root / DOC04_DIR / DOC04_PEAKS_2MM
    for row in _read_rows(p2):
        if row.get("data_source_zh", "").strip() != "试验":
            continue
        label = row["composite_type_zh"].strip()
        ctype = ZH_LABEL_2MM.get(label)
        if ctype is None:
            continue
        for col, dia in (("30mm_kn", 30.0), ("32mm_kn", 32.0)):
            val = row.get(col, "").strip()
            if val:
                peaks[(ctype, dia)] = _float(val)

    return peaks


def _read_doc04_curves(data_root: Path) -> List[MemberCurve]:
    """All doc04 member curves (both calibration and held-out composite types)."""
    rows = _read_rows(data_root / DOC04_DIR / DOC04_CURVES)
    peaks = _read_doc04_peaks(data_root)

    # group (composite_type, diameter) -> point lists
    grouped: Dict[Tuple[str, float], Tuple[List[float], List[float]]] = {}
    for r in rows:
        if r.get("data_source", "").strip() != "test":  # defensive filter
            continue
        ctype = r["composite_type"].strip()
        dia = _float(r["tube_diameter_mm"])
        d = _float(r["deflection_mm"])
        ld = _float(r["load_kn"])
        grouped.setdefault((ctype, dia), ([], []))
        grouped[(ctype, dia)][0].append(d)
        grouped[(ctype, dia)][1].append(ld)

    curves: List[MemberCurve] = []
    for (ctype, dia), (dd, ll) in sorted(grouped.items()):
        defl, load = _sorted_curve(dd, ll)
        cfg = ConfigFeatures(
            diameter_mm=dia,
            cfrp_thickness_mm=_cfrp_thickness_of(ctype),
            cement=_cement_of(ctype),
            system="doc04",
        )
        peak = peaks.get((ctype, dia))
        if peak is not None:
            peak_kN, psrc = peak, "table"
        else:
            peak_kN, psrc = float(np.max(load)) if load.size else float("nan"), "curve_max"
        curves.append(
            MemberCurve(
                config=cfg,
                deflection_mm=defl,
                load_kN=load,
                peak_kN=peak_kN,
                source_id=f"doc04:{ctype}@{int(dia)}mm",
                composite_type=ctype,
                diameter_mm=dia,
                peak_source=psrc,
            )
        )
    return curves


# ---------------------------------------------------------------------------
# doc13 readers
# ---------------------------------------------------------------------------
# doc13 is heterogeneous. We encode a specimen into the numeric family ONLY when it is
# an ordinary steel square tube with at most {external CFRP wrap of known thickness,
# cement infill}. Anything else (stainless/aluminium/PVC framing, epoxy filling,
# CFRP-fibre cement, embedded round tubes, bolts, surface grinding, axial/circumferential
# wrap direction) is recorded in ``extras``/``skip_reason`` and NOT turned into a clean
# (diameter, thickness, cement) member -- per the spec, document rather than fabricate.


def _encode_doc13_composition(comp_en: str) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    """Parse a doc13 English composition string.

    Returns ``(encodable, extras)`` where ``encodable`` is None when the specimen is not
    cleanly representable as an ordinary steel-tube (diameter, thickness, cement) member.
    ``encodable`` (when not None) carries diameter_mm / cfrp_thickness_mm / cement.
    ``extras`` always records the parsed qualitative descriptors for traceability.
    """
    s = comp_en.lower()
    extras: Dict[str, object] = {"composition_en": comp_en}

    # diameter
    import re

    m = re.search(r"(\d+)\s*mm", s)
    diameter = float(m.group(1)) if m else None

    # framing material -- only plain steel tubes are cleanly encodable
    if "stainless steel" in s:
        framing = "stainless_steel"
    elif "aluminium" in s or "aluminum" in s:
        framing = "aluminium"
    elif "pvc" in s:
        framing = "pvc"
    elif "steel tube" in s:
        framing = "steel"
    else:
        framing = "unknown"
    extras["framing"] = framing

    # cement infill
    cement = 1 if "cement" in s else 0

    # CFRP wrap thickness + direction. Match the thickness token that directly
    # qualifies the CFRP descriptor (e.g. "1mm CFRP", "2mm axial CFRP"); a bare "2mm"
    # substring must NOT be picked up from the diameter token "32mm".
    thickness = 0.0
    if "cfrp" in s:
        mt = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+(?:axial\s+|circumferential\s+)?cfrp", s)
        if mt is not None:
            thickness = float(mt.group(1))
    if "circumferential cfrp" in s:
        extras["wrap_direction"] = "circumferential"
    elif "axial cfrp" in s:
        extras["wrap_direction"] = "axial"
    elif "cfrp wrap" in s:
        extras["wrap_direction"] = "ordinary"

    # disqualifying / non-clean features
    non_clean = []
    if "embedded" in s:
        non_clean.append("embedded_tube")
        extras["internal_placement"] = "embedded"
    if "epoxy" in s:
        non_clean.append("epoxy_filling")
        extras["filling"] = "epoxy"
    if "cfrp-fibre cement" in s or "cfrp fibre cement" in s or "0.6%" in s:
        non_clean.append("cfrp_fibre_cement")
        extras["filling"] = "cfrp_fibre_cement"
    if "bolt" in s:
        non_clean.append("bolt")
    if "ground" in s:
        non_clean.append("surface_ground")
    if "axial cfrp" in s or "circumferential cfrp" in s:
        # Directional wrap is not the ordinary isotropic external布 -> keep out of the
        # clean numeric family (still recorded in extras).
        non_clean.append("directional_wrap")
    if framing != "steel":
        non_clean.append(f"framing_{framing}")
    if diameter is None:
        non_clean.append("no_diameter")

    if non_clean:
        extras["skip_reason"] = ",".join(non_clean)
        return None, extras

    encodable = {
        "diameter_mm": diameter,
        "cfrp_thickness_mm": thickness,
        "cement": cement,
    }
    return encodable, extras


def _read_doc13_capacities(data_root: Path) -> Dict[str, float]:
    """Map doc13 specimen id -> ultimate capacity (kN) where the table is specimen-keyed.

    ``ultimate_bearing_capacity.csv`` is heterogeneous: only the ``x_role == 'specimen'``
    rows are keyed by specimen id; the diameter-swept rows are not used here (their
    condition series are not a 1:1 specimen mapping). Returns {} when none match.
    """
    caps: Dict[str, float] = {}
    rows = _read_rows(data_root / DOC13_DIR / DOC13_CAPACITY)
    for r in rows:
        if r.get("x_role", "").strip() != "specimen":
            continue
        spec = r.get("x_category", "").strip()
        val = r.get("ultimate_capacity_kN", "").strip()
        if spec and val:
            caps[spec] = _float(val)
    return caps


def _read_doc13_curves(data_root: Path) -> Tuple[List[MemberCurve], List[Dict[str, object]]]:
    """doc13 calibration member curves (cleanly encodable only) + a skipped-log.

    Returns ``(curves, skipped)`` where ``skipped`` is a list of dicts describing each
    doc13 specimen that could not be cleanly encoded (for documentation, not fabrication).
    """
    rows = _read_rows(data_root / DOC13_DIR / DOC13_CURVES)
    caps = _read_doc13_capacities(data_root)

    # group by specimen; remember its (English) composition string
    grouped: Dict[str, Tuple[List[float], List[float], str]] = {}
    for r in rows:
        if "point_type" in r and r.get("point_type", "").strip() not in ("measured", "interpolated"):
            continue
        spec = r["specimen"].strip()
        comp = r.get("composition", "").strip()
        d = _float(r["deflection_mm"])
        ld = _float(r["load_kN"])
        if spec not in grouped:
            grouped[spec] = ([], [], comp)
        grouped[spec][0].append(d)
        grouped[spec][1].append(ld)

    curves: List[MemberCurve] = []
    skipped: List[Dict[str, object]] = []
    for spec in sorted(grouped):
        dd, ll, comp = grouped[spec]
        encodable, extras = _encode_doc13_composition(comp)
        if encodable is None:
            skipped.append({"specimen": spec, **extras})
            continue
        defl, load = _sorted_curve(dd, ll)
        cfg = ConfigFeatures(
            diameter_mm=encodable["diameter_mm"],
            cfrp_thickness_mm=encodable["cfrp_thickness_mm"],
            cement=encodable["cement"],
            system="doc13",
            extras=extras,
        )
        cap = caps.get(spec)
        if cap is not None:
            peak_kN, psrc = cap, "capacity_csv"
        else:
            peak_kN, psrc = float(np.max(load)) if load.size else float("nan"), "curve_max"
        curves.append(
            MemberCurve(
                config=cfg,
                deflection_mm=defl,
                load_kN=load,
                peak_kN=peak_kN,
                source_id=f"doc13:{spec}",
                composite_type=spec,  # specimen id is the doc13 type granularity
                diameter_mm=encodable["diameter_mm"],
                peak_source=psrc,
            )
        )
    return curves, skipped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _is_doc04_calibration(mc: MemberCurve) -> bool:
    return (
        mc.config.system == "doc04"
        and mc.composite_type in CALIBRATION_COMPOSITES
    )


def _is_doc04_held_out(mc: MemberCurve) -> bool:
    return (
        mc.config.system == "doc04"
        and mc.composite_type in HELD_OUT_COMPOSITES
        and mc.diameter_mm in HELD_OUT_DIAMETERS
    )


def load_member_curves(
    data_root: Path | str = ROOT, partition: str = "all", truncate_post_peak: bool = False
) -> List[MemberCurve]:
    """Load measured member curves for the requested partition.

    Parameters
    ----------
    data_root:
        Project root (the directory that contains ``results/``). Defaults to the
        repository root inferred from this file's location.
    partition:
        ``"calibration"``, ``"held_out"``, or ``"all"``.

        * calibration = doc04 calibration composites x {30,32,35} PLUS all cleanly
          encodable doc13 measured curves.
        * held_out    = doc04 {tube_cfrp2mm, tube_cement_cfrp2mm} x {30,32} only.
        * all         = calibration + held_out (doc13 only appears in calibration).
    truncate_post_peak:
        When True, trim each curve to its loading branch (points up to peak load) so the
        monotone section law is fit/scored only on its valid ascending domain; the post-peak
        softening branch is dropped. ``peak_kN`` is unchanged. Default False.

    Returns
    -------
    list[MemberCurve]. doc13 specimens that cannot be cleanly encoded are silently
    excluded from the family here (use :func:`doc13_skipped_specimens` to inspect them).
    """
    if partition not in ("calibration", "held_out", "all"):
        raise ValueError(
            f"partition must be 'calibration'|'held_out'|'all', got {partition!r}"
        )
    root = Path(data_root)

    doc04 = _read_doc04_curves(root)
    doc04_cal = [mc for mc in doc04 if _is_doc04_calibration(mc)]
    doc04_held = [mc for mc in doc04 if _is_doc04_held_out(mc)]

    doc13_cal, _skipped = _read_doc13_curves(root)

    if partition == "held_out":
        result = list(doc04_held)
    elif partition == "calibration":
        result = doc04_cal + doc13_cal
    else:  # all
        result = doc04_cal + doc13_cal + doc04_held

    if truncate_post_peak:
        for mc in result:
            _truncate_to_loading_branch(mc)
    return result


def doc13_skipped_specimens(data_root: Path | str = ROOT) -> List[Dict[str, object]]:
    """Return the documented list of doc13 specimens not cleanly encodable.

    Each entry carries the specimen id, the parsed qualitative descriptors, and a
    ``skip_reason`` -- this is the audit trail proving doc13 curves were documented
    rather than fabricated into clean features.
    """
    _curves, skipped = _read_doc13_curves(Path(data_root))
    return skipped


def assert_no_leak(
    calibration: List[MemberCurve], held_out: List[MemberCurve]
) -> None:
    """Raise if any held-out (composite_type, diameter) appears in calibration.

    Guards K2 non-circularity. The check is at the (composite_type, diameter)
    granularity for doc04 (the partition is defined on that key). doc13 specimens use
    their specimen id as ``composite_type`` and live only in calibration, so they
    cannot collide with the doc04 held-out 2 mm group.
    """
    cal_keys = {mc.partition_key() for mc in calibration}
    held_keys = {mc.partition_key() for mc in held_out}
    leaked = sorted(cal_keys & held_keys)
    if leaked:
        raise AssertionError(
            "calibration/held-out leak detected at (composite_type, diameter): "
            f"{leaked}"
        )
    # Belt-and-suspenders: the held-out set must be exactly the locked 2 mm group.
    bad = [k for k in held_keys
           if not (k[0] in HELD_OUT_COMPOSITES and k[1] in HELD_OUT_DIAMETERS)]
    if bad:
        raise AssertionError(
            f"held_out contains configs outside the locked 2 mm group: {bad}"
        )
