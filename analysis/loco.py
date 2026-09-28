"""Leave-one-configuration-out cross-validation of a fixed map specification.

Each of the 22 configurations is a fold target once, with the other 21 used
for coefficient fitting. The specification was selected during development;
this procedure does not repeat model selection inside each fold. The original
four-configuration reserved block is a different evaluation: all four are
excluded together and its scores were visible during model development.
Reporting only the 18 non-reserved targets changes the score average, not
any fold's training membership.

Usage: python -m analysis.loco
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from analysis.paths import RESULTS_DIR
from analysis.refit import fit_variant, load_config, load_partitions
from analysis.section_family import PRODUCTION_VARIANT, VARIANTS, free_slots

__all__ = ["run_loco", "main"]


# doc04 is the first (square steel tube) campaign, doc13 the second (stainless) one.
def _campaign_of(source_id: str) -> str:
    return "first" if str(source_id).startswith("doc04") else "second"


def _depth_of(config_key) -> float:
    """Tube depth in mm: the second entry of the (doc, diameter, wrap, grout) key."""
    return float(config_key[1])


def _summarize(rows: Sequence[dict], label: str) -> dict:
    if not rows:
        return {"group": label, "n": 0}
    err = np.array([r["peak_rel_err"] for r in rows], dtype=float)
    over = err[err > 0]
    under = err[err < 0]
    return {
        "group": label,
        "n": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "signed_mean": float(np.mean(err)),
        "worst_abs": float(np.max(np.abs(err))),
        "worst_unconservative": float(np.max(over)) if over.size else 0.0,
        "worst_conservative": float(np.min(under)) if under.size else 0.0,
        "rms": float(np.sqrt(np.mean(err ** 2))),
    }


def run_loco(variant_name: str, cfg: dict, *, n_jobs: int = 1) -> dict:
    """Refit the family once per configuration, each time withholding that one cell."""
    calib, held = load_partitions(cfg)
    pool = list(calib) + list(held)
    block_held_ids = {mc.source_id for mc in held}
    n = len(pool)

    rows: List[dict] = []
    t0 = time.time()
    for i, mc in enumerate(pool):
        train = [c for j, c in enumerate(pool) if j != i]
        res = fit_variant(variant_name, train, [mc], cfg, n_jobs=n_jobs)
        scored = res.held_out[0]
        row = {
            "fold": i,
            "source_id": scored["source_id"],
            "config_key": scored["config_key"],
            "depth_mm": _depth_of(scored["config_key"]),
            "campaign": _campaign_of(scored["source_id"]),
            "in_manuscript_held_out_block": scored["source_id"] in block_held_ids,
            "peak_meas_kN": scored["peak_meas_kN"],
            "peak_pred_kN": scored["peak_pred_kN"],
            "peak_rel_err": scored["peak_rel_err"],
            "curve_rel_l2": scored.get("curve_rel_l2"),
            "n_train": len(train),
            "condition_number": res.condition_number,
        }
        rows.append(row)
        print(
            f"  [{i + 1:2d}/{n}] {row['source_id']:34s} "
            f"d={row['depth_mm']:.0f}mm  err={row['peak_rel_err'] * 100:+6.1f}%",
            flush=True,
        )

    depths = sorted({r["depth_mm"] for r in rows})
    artifact = {
        "kind": "leave_one_configuration_out_cross_validation",
        "variant": variant_name,
        "n_configurations": n,
        "n_coefficients": len(free_slots(VARIANTS[variant_name])),
        "n_train_per_fold": n - 1,
        "wall_s": round(time.time() - t0, 1),
        "overall": _summarize(rows, "all"),
        "by_depth": [
            _summarize([r for r in rows if r["depth_mm"] == d], f"{d:.0f} mm")
            for d in depths
        ],
        "by_campaign": [
            _summarize([r for r in rows if r["campaign"] == c], c)
            for c in ("first", "second")
        ],
        "manuscript_block": _summarize(
            [r for r in rows if r["in_manuscript_held_out_block"]],
            "manuscript reserved 2 mm block, scored one at a time",
        ),
        "folds": rows,
        "note": (
            f"Each fold refits the fixed map specification on {n - 1} configurations "
            "and scores the withheld one. The original reserved block excludes all four "
            "first-campaign 2 mm configurations together. Cross-validation retains three "
            "of them in training when the fourth is the target. Excluding those four "
            "targets from a summary changes only the score average."
        ),
    }
    return artifact


def _print_report(a: dict) -> None:
    o = a["overall"]
    print(f"\n=== LOCO {a['variant']} ===  {a['n_configurations']} folds, "
          f"{a['n_coefficients']} coefficients on {a['n_train_per_fold']} configurations, "
          f"{a['wall_s']}s")
    print(f"  overall   MAE {o['mae']:.4f}  signed {o['signed_mean'] * 100:+.1f}%  "
          f"worst {o['worst_abs'] * 100:.1f}%  "
          f"(unconservative {o['worst_unconservative'] * 100:+.1f}%, "
          f"conservative {o['worst_conservative'] * 100:+.1f}%)")
    for g in a["by_depth"] + a["by_campaign"]:
        if g["n"]:
            print(f"  {g['group']:>10s}  n={g['n']:2d}  MAE {g['mae']:.4f}  "
                  f"worst {g['worst_abs'] * 100:.1f}%  "
                  f"(unc {g['worst_unconservative'] * 100:+.1f}%, "
                  f"con {g['worst_conservative'] * 100:+.1f}%)")
    b = a["manuscript_block"]
    if b["n"]:
        print(f"  reserved 2 mm block one-at-a-time: n={b['n']} MAE {b['mae']:.4f} "
              f"worst {b['worst_abs'] * 100:.1f}%")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", default=PRODUCTION_VARIANT, choices=sorted(VARIANTS))
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None,
                   help="JSON output path (default results/analysis/loco_<variant>.json)")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    artifact = run_loco(args.variant, cfg, n_jobs=args.n_jobs)
    _print_report(artifact)

    out = Path(args.out) if args.out else (RESULTS_DIR / f"loco_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
