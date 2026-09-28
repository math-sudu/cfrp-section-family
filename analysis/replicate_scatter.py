"""Duplicate-specimen scatter of the measured peak loads.

The identification is fitted to, and scored against, one peak load per
configuration.  Each of those numbers is a two-specimen mean: the source
reports twelve formal groups, "each group made up of two specimens, and the
final data take the mean of the two specimens' measurements", and confirms the
count family by family as "three groups, six tubes".  The per-specimen values
survive in the curated per-family tables ``table2-4`` .. ``table2-7``, whose two
data rows are the two specimens of each group and whose third row is the mean
that every downstream table carries.

This module recovers the pair spread those tables preserve, so a held-out error
can be read against the dispersion of the quantity it is an error in.  The
scatter is a property of the measurement, not of the fit: nothing here touches
the section family or the identification.

Scope, stated because it bounds what the table can answer: the four held-out
2 mm cells belong to the source's four orthogonal groups, which report a single
test value per cell (``table4-1``) with no per-specimen split.  Their own
replicate dispersion is therefore not recoverable, and the calibration pairs
are the only available estimate of the fabrication-and-test scatter of this
specimen population.

For a pair the sample standard deviation is ``|x1 - x2| / sqrt(2)``; the
relative range ``|x1 - x2| / mean`` is reported beside it because it is the
form a reader compares an error percentage against directly.

Usage
-----
    python -m analysis.replicate_scatter
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

from analysis.paths import TUST_ROOT, RESULTS_DIR

__all__ = ["load_pairs", "summarize", "main"]

CURATED = TUST_ROOT / "data/members/campaign1"

# curated per-family peak tables -> the doc04 composite_type they hold
FAMILY_TABLES = {
    "single_tube": "table2-4_single_tube_peak_load_kn.csv",
    "tube_cement": "table2-5_tube_cement_peak_load_kn.csv",
    "tube_cfrp1mm": "table2-6_tube_cfrp1mm_peak_load_kn.csv",
    "tube_cement_cfrp1mm": "table2-7_tube_cement_cfrp1mm_peak_load_kn.csv",
}

# The two data rows are the two specimens; the third is the mean the family carries.
SPECIMEN_ROWS = ("第一次加载", "第二次加载")
MEAN_ROW = "平均"

DIAMETER_COLUMNS = {"30mm_kn": 30.0, "32mm_kn": 32.0, "35mm_kn": 35.0}


def load_pairs(curated: Path | None = None) -> List[Dict[str, object]]:
    """Read the twelve calibration configurations' specimen pairs from the curated CSVs."""
    root = Path(curated) if curated is not None else CURATED
    pairs: List[Dict[str, object]] = []
    for composite_type, filename in FAMILY_TABLES.items():
        path = root / filename
        rows = {r["loading_run"].strip(): r for r in csv.DictReader(path.read_text(encoding="utf-8").splitlines())}
        missing = [k for k in (*SPECIMEN_ROWS, MEAN_ROW) if k not in rows]
        if missing:
            raise ValueError(f"{path.name}: missing rows {missing}")
        for column, diameter in DIAMETER_COLUMNS.items():
            values = [float(rows[k][column]) for k in SPECIMEN_ROWS]
            reported_mean = float(rows[MEAN_ROW][column])
            mean = sum(values) / 2.0
            rng = abs(values[0] - values[1])
            pairs.append({
                "source_id": f"doc04:{composite_type}@{diameter:.0f}mm",
                "composite_type": composite_type,
                "diameter_mm": diameter,
                "specimen_1_kN": values[0],
                "specimen_2_kN": values[1],
                "mean_kN": mean,
                "reported_mean_kN": reported_mean,
                "range_kN": rng,
                "sd_kN": rng / math.sqrt(2.0),
                "relative_range_pct": 100.0 * rng / mean,
                "cv_pct": 100.0 * (rng / math.sqrt(2.0)) / mean,
                # the curated mean is rounded to 2 decimals; keep the residual visible
                "mean_rounding_kN": reported_mean - mean,
            })
    return pairs


# Every number in these tables carries two decimals, so two roundings separate the
# carried mean from the mean of the listed pair: each listed specimen is its true value
# to within 0.005 kN, which moves their average by up to 0.005 kN, and the carried mean
# is itself rounded, for another 0.005 kN. A gap wider than their sum cannot be produced
# by rounding, and means the listed pair and the carried mean are not the same two
# measurements.
ROUNDING_TOLERANCE_KN = 0.010 + 1e-9


def _pool(pairs: List[Dict[str, object]]) -> Dict[str, float]:
    """Pooled dispersion of a set of pairs, and the noise floor it implies."""
    cvs = [float(p["cv_pct"]) for p in pairs]
    rels = sorted(float(p["relative_range_pct"]) for p in pairs)
    n = len(rels)
    # Pooled within-configuration CV: root mean square, the estimator that adds the
    # per-pair variances rather than averaging their square roots.
    pooled_cv = math.sqrt(sum(c * c for c in cvs) / n)
    median = 0.5 * (rels[n // 2 - 1] + rels[n // 2]) if n % 2 == 0 else rels[n // 2]
    return {
        "n": n,
        "pooled_cv_pct": pooled_cv,
        # The quantity the identification is fitted to is the two-specimen MEAN, whose
        # standard error is the specimen CV over sqrt(2). This -- not the specimen CV --
        # is the measurement floor the noise model should carry.
        "mean_standard_error_pct": pooled_cv / math.sqrt(2.0),
        "median_relative_range_pct": median,
        "min_relative_range_pct": rels[0],
        "max_relative_range_pct": rels[-1],
    }


def summarize(pairs: List[Dict[str, object]]) -> Dict[str, object]:
    """Pool the pair dispersions, with and without the arithmetically inconsistent pairs."""
    inconsistent = [
        p for p in pairs if abs(float(p["mean_rounding_kN"])) > ROUNDING_TOLERANCE_KN
    ]
    clean = [p for p in pairs if p not in inconsistent]
    worst = max(pairs, key=lambda p: float(p["relative_range_pct"]))
    return {
        "n_configurations": len(pairs),
        "n_specimens": 2 * len(pairs),
        "all_pairs": _pool(pairs),
        "consistent_pairs_only": _pool(clean),
        "arithmetically_inconsistent": [
            {
                "source_id": p["source_id"],
                "pair_mean_kN": p["mean_kN"],
                "carried_mean_kN": p["reported_mean_kN"],
                "gap_kN": p["mean_rounding_kN"],
                "relative_range_pct": p["relative_range_pct"],
            }
            for p in inconsistent
        ],
        "worst_configuration": worst["source_id"],
        "worst_relative_range_pct": worst["relative_range_pct"],
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output JSON (default results/analysis/replicate_scatter.json)")
    args = ap.parse_args(argv)

    pairs = load_pairs()
    summary = summarize(pairs)

    print(f"{'configuration':<34}{'spec 1':>9}{'spec 2':>9}{'mean':>9}{'range':>8}{'rel. range':>12}")
    for p in pairs:
        print("%-34s%9.2f%9.2f%9.2f%8.2f%11.2f%%" % (
            p["source_id"], p["specimen_1_kN"], p["specimen_2_kN"],
            p["mean_kN"], p["range_kN"], p["relative_range_pct"]))
    print()
    print("n = %d configurations (%d specimens)" % (summary["n_configurations"], summary["n_specimens"]))
    for label, key in (("all pairs", "all_pairs"), ("consistent pairs only", "consistent_pairs_only")):
        s = summary[key]
        print("%-24s n=%2d  rel. range %.2f-%.2f%% (median %.2f%%)  pooled CV %.2f%%  "
              "SE of the 2-specimen mean %.2f%%"
              % (label, s["n"], s["min_relative_range_pct"], s["max_relative_range_pct"],
                 s["median_relative_range_pct"], s["pooled_cv_pct"], s["mean_standard_error_pct"]))
    print("worst pair: %s at %.2f%%" % (summary["worst_configuration"], summary["worst_relative_range_pct"]))
    for bad in summary["arithmetically_inconsistent"]:
        print("  carried mean disagrees with the listed pair: %s  pair %.3f vs carried %.2f kN "
              "(gap %+.3f kN)" % (bad["source_id"], bad["pair_mean_kN"],
                                  bad["carried_mean_kN"], bad["gap_kN"]))

    out = Path(args.out) if args.out else RESULTS_DIR / "replicate_scatter.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "replicate_scatter_v1",
        "source": "doc04 curated table2-4 .. table2-7 (two specimens per configuration)",
        "held_out_note": (
            "the four held-out 2 mm cells come from the source's orthogonal groups, "
            "which report one test value per cell with no per-specimen split; their "
            "replicate dispersion is not recoverable"
        ),
        "pairs": pairs,
        "summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
