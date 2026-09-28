"""Evaluate the bundled fit against member records without optimization."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.member_data import DOC13_ADDITIONAL, DOC13_CONFIGURATIONS, load_doc13_curve
from analysis.paths import TUST_ROOT
from analysis.refit import load_config, load_partitions, score
from analysis.section_family import PRODUCTION_VARIANT, VARIANTS, free_slots
from section_model import DEFAULT_FIT, SectionFamily


def evaluate():
    cfg = load_config()
    calibration, reserved = load_partitions(cfg)
    fit = json.loads(DEFAULT_FIT.read_text(encoding="utf-8"))
    variant = VARIANTS[PRODUCTION_VARIANT]
    coefficients = np.asarray(fit["A"])
    theta = np.asarray([coefficients[i, j] for i, j in free_slots(variant)])
    args = (cfg["span_mm"], cfg["forward"]["n_quad"])
    output = {"calibration": score(variant, theta, calibration, *args),
              "development_reserved": score(variant, theta, reserved, *args)}
    with (TUST_ROOT / "data/members/campaign2/capacity_ductility.csv").open(
            encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    family = SectionFamily()
    additional = []
    for specimen in DOC13_ADDITIONAL:
        capacities = {float(row["ultimate_capacity_kN"]) for row in rows
                      if row["specimen"] == specimen}
        if len(capacities) != 1:
            raise ValueError(f"Ambiguous tabulated capacity for {specimen}: {capacities}")
        observed = capacities.pop()
        depth, wrap, infill = DOC13_CONFIGURATIONS[specimen]
        section = family.section(depth_mm=depth, wrap_mm=wrap, grouted=infill, campaign=2)
        predicted = 4.0 * section.Mu / cfg["span_mm"]["doc13"] / 1000.0
        additional.append({"specimen": specimen, "peak_meas_kN": observed,
                           "peak_pred_kN": predicted,
                           "peak_rel_err": (predicted - observed) / observed})
    output["additional_capacities"] = additional
    output["additional_S30_response"] = score(variant, theta, [load_doc13_curve("C")], *args)[0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TUST_ROOT / "results/evaluation.json")
    args = parser.parse_args()
    report = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for group in ("calibration", "development_reserved", "additional_capacities"):
        records = report[group]
        mae = np.mean([abs(row["peak_rel_err"]) for row in records])
        print(f"{group}: n={len(records)}, mean absolute peak-load error={100*mae:.3f}%")
