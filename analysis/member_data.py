"""Load the reported calibration subset and additional campaign responses.

Roles are assigned by source configuration, independently of English material
labels. S30 (source C) is a stainless-steel family member with a digitized
response; it is evaluated separately from the six second-campaign fit records.
Repeated figure printings and interpolated points remain one configuration.
"""
from __future__ import annotations

import csv
from pathlib import Path

from analysis.paths import TUST_ROOT, ensure_local_importable

ensure_local_importable()
from cfrpequiv.config_features import ConfigFeatures
from cfrpequiv.data_io import (
    DOC13_DIR, DOC13_CURVES, MemberCurve, _read_doc04_curves,
    _read_doc13_capacities, _sorted_curve, _truncate_to_loading_branch,
    _is_doc04_calibration, _is_doc04_held_out, assert_no_leak,
)

# Source codes map to (depth in mm, nominal wrap in mm, infill indicator).
# These are the ordinary-wrap records in the source specimen matrix.
DOC13_CONFIGURATIONS = {
    "A": (35, 0, 0), "B": (32, 0, 0), "C": (30, 0, 0),
    "A-1": (35, 0, 1), "A-2": (35, 1, 1), "A-3": (35, 2, 1),
    "B-1": (32, 0, 1), "B-2": (32, 1, 0), "B-3": (32, 1, 1),
    "C-1": (30, 0, 1), "C-2": (30, 2, 0), "C-3": (30, 2, 1),
}
DOC13_CALIBRATION = ("A-1", "A-3", "B-3", "C-1", "C-2", "C-3")
DOC13_ADDITIONAL = ("A", "B", "C", "B-1", "B-2", "A-2")


def load_doc13_curve(specimen: str, *, data_root: Path = TUST_ROOT,
                     truncate_post_peak: bool = True) -> MemberCurve:
    """Read one configuration's digitized mean response, retaining its traces."""
    depth, wrap, infill = DOC13_CONFIGURATIONS[specimen]
    path = data_root / DOC13_DIR / DOC13_CURVES
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["specimen"] == specimen
                and r["point_type"] in ("measured", "interpolated")]
    if not rows:
        raise ValueError(f"No digitized response for second-campaign {specimen}")
    deflection, load = _sorted_curve(
        [float(r["deflection_mm"]) for r in rows],
        [float(r["load_kN"]) for r in rows],
    )
    capacities = _read_doc13_capacities(data_root)
    peak = capacities.get(specimen)
    curve = MemberCurve(
        ConfigFeatures(depth, wrap, infill, "doc13", extras={
            "framing": "stainless_steel", "source_specimen": specimen,
            "source_figures": sorted({r["figure_id"] for r in rows}),
        }),
        deflection, load, float(max(load)) if peak is None else peak,
        f"doc13:{specimen}", specimen, float(depth),
        "curve_max" if peak is None else "capacity_csv",
    )
    if truncate_post_peak:
        _truncate_to_loading_branch(curve)
    return curve


def load_partitions(cfg: dict):
    """Return the 18 fit records and four development-reserved records."""
    truncate = bool(cfg.get("truncate_post_peak", True))
    first = _read_doc04_curves(TUST_ROOT)
    calibration = [mc for mc in first if _is_doc04_calibration(mc)]
    reserved = [mc for mc in first if _is_doc04_held_out(mc)]
    if truncate:
        for curve in calibration + reserved:
            _truncate_to_loading_branch(curve)
    calibration += [load_doc13_curve(s, truncate_post_peak=truncate)
                    for s in DOC13_CALIBRATION]
    assert_no_leak(calibration, reserved)
    return calibration, reserved
