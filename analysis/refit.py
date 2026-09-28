"""Joint nonlinear least-squares identification from member bending records.

Residuals combine deflection compatibility, measured peak loads, the
stiffness floor and a one-sided elastic-stiffness penalty. Six seeded
restarts fit the 18 calibration configurations and score the four
development-reserved configurations. Run python -m analysis.refit."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.optimize import least_squares

from analysis.paths import TUST_ROOT, RESULTS_DIR, ensure_local_importable
from analysis.elastic_ceiling import elastic_ceiling_Nmm2
from analysis.member_data import load_partitions
from analysis.section_family import (
    PRODUCTION_VARIANT,
    Variant,
    VARIANTS,
    feature_names,
    free_slots,
    params_for,
    theta_bounds,
    theta_init,
    unpack_theta,
)

ensure_local_importable()

from cfrpequiv.forward_beam import deflection_at_load, peak_load  # noqa: E402
from cfrpequiv.section_law import make_section_law  # noqa: E402

__all__ = [
    "FitResult", "load_config", "load_partitions", "fit_variant", "score", "span_for", "main",
]

KN_TO_N = 1.0e3
DEFAULT_CONFIG = "configs/identification.yaml"
DEFAULT_SPAN_MM = 490.0

# The two campaigns did not bend their members over the same span, and only the first
# records one. doc04's records give 500 mm tubes at a clear span of 490 mm. doc13's press
# seats its specimen on a 450 mm platform and its support blocks stand on that platform,
# which bounds its reaction spacing at 450 mm; blocks of any finite width put the true
# value below it, and the records do not fix a block width. 450 mm is therefore carried
# here as the bound rather than as a measurement, and every doc13 EI_0 and M_u identified
# at it is an UPPER estimate -- the admissible band runs down to about 400 mm, which is
# what a 50 mm block at each platform edge would leave. The parent's run config still
# reads a single 490 mm: it records the archived v0 run, which was fit before this bound
# was read out of the records, and reproducing that fit needs its historical span
# (``load_config(campaign_spans=False)``).
CAMPAIGN_SPAN_MM = {"doc04": 490.0, "doc13": 450.0}
CAMPAIGN_SPAN_BAND_MM = {"doc13": (400.0, 450.0)}


def span_for(span_cfg, system: str) -> float:
    """Clear span [mm] of one configuration's campaign.

    ``span_mm`` is a scalar or a per-campaign mapping like ``{doc04: 490, doc13: 430}``.
    The two campaigns ran on different machines and only the first states a clear span
    in its records (``L = 500`` mm tubes at ``l = 490`` mm); the second's press seats
    its specimen on a 450 mm platform, which bounds its reaction spacing below the
    value the first campaign fixes. The span enters every member moment linearly
    (``M_u = P_u L / 4``) and every stiffness read from a deflection cubically
    (``delta = P L^3 / 48 EI``), so it cannot be shared across the two.
    """
    if isinstance(span_cfg, dict):
        return float(span_cfg.get(system, DEFAULT_SPAN_MM))
    return float(span_cfg)


# ============================================================================================
# Config + data
# ============================================================================================
def load_config(path: Optional[str] = None, *, campaign_spans: bool = True) -> dict:
    """Read local settings, including an explicitly supplied campaign span.

    ``campaign_spans=False`` selects the historical single-span baseline.
    """
    p = Path(path) if path else (TUST_ROOT / DEFAULT_CONFIG)
    if not p.is_absolute():
        p = TUST_ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    if not campaign_spans:
        cfg["span_mm"] = DEFAULT_SPAN_MM
    return cfg


# ============================================================================================
# Residual
# ============================================================================================
def build_blocks(curves, cfg: dict) -> Tuple[List[dict], object, float, int]:
    """Pre-extract the weighted, down-sampled residual blocks once per fit.

    ``residual.defl_window_min_mm`` masks curve points below a trusted deflection
    threshold -- a scalar, or a per-campaign mapping like ``{doc04: 0.5, doc13:
    1.75}`` since the two campaigns' source figures differ in resolution. It
    exists for the SR2-B10 trusted-window sensitivity refits; the production
    config does not set it, so the default (0 = no masking) reproduces the
    archived fit exactly. The peak block is never masked: the peaks are tabulated
    capacities that do not pass through the digitized trace.

    ``span_mm`` takes the same two forms (:func:`span_for`), and each block carries
    its own resolved span so one fit can hold two campaigns on two machines.
    """
    span_cfg = cfg.get("span_mm", DEFAULT_SPAN_MM)
    n_quad = int((cfg.get("forward", {}) or {}).get("n_quad", 64))
    res_cfg = cfg.get("residual", {}) or {}
    max_points = int(res_cfg.get("max_points_per_curve", 24))
    defl_floor = float(res_cfg.get("defl_scale_floor_mm", 0.05))
    window = res_cfg.get("defl_window_min_mm", 0.0)
    weights = cfg.get("loss_weights", {}) or {}
    w_data = float(weights.get("data", 1.0))
    w_peak = float(weights.get("peak", 1.0))
    w_prior = float(weights.get("prior", 0.05))

    blocks = []
    for mc in curves:
        defl, load_kN = mc.deflection_mm, mc.load_kN
        w_min = (float(window.get(mc.config.system, 0.0)) if isinstance(window, dict)
                 else float(window))
        if w_min > 0.0:
            keep = defl >= w_min
            defl, load_kN = defl[keep], load_kN[keep]
        if max_points > 0 and defl.size > max_points:
            idx = np.unique(np.linspace(0, defl.size - 1, max_points).round().astype(int))
            defl, load_kN = defl[idx], load_kN[idx]
        d_abs = np.abs(defl)
        floor_pt = max(defl_floor, 0.05 * (float(d_abs.max()) if d_abs.size else 0.0))
        blocks.append({
            "config": mc.config,
            "source_id": mc.source_id,
            "span": span_for(span_cfg, mc.config.system),
            "P_N": load_kN * KN_TO_N,
            "delta_meas": defl,
            "scale": np.maximum(d_abs, floor_pt),
            "w_point": w_data / np.sqrt(max(defl.size, 1)),
            "P_u_meas": float(mc.peak_kN) * KN_TO_N,
            "peak_scale": abs(float(mc.peak_kN) * KN_TO_N),
            "w_peak": w_peak,
        })
    return blocks, span_cfg, w_prior, n_quad


def _ridge_slot_indices(variant: Variant) -> List[int]:
    """Positions in ``theta`` of the non-bias coefficients the archived ridge penalizes."""
    slots = free_slots(variant)
    index = {slot: k for k, slot in enumerate(slots)}
    out: List[int] = []
    for row_name in variant.ridge_rows:
        i = variant.rows.index(row_name)
        for c in range(1, variant.feature_dim):
            if (i, c) in index:
                out.append(index[(i, c)])
    return out


def residual(
    theta: np.ndarray, variant: Variant, blocks: List[dict],
    w_prior: float, n_quad: int, ridge_idx: Sequence[int],
) -> np.ndarray:
    """Stacked ``[data | peak | priors (| ridge)]`` residual for one ``theta``.

    Each block carries its own campaign's span (:func:`build_blocks`).
    """
    laws: Dict[str, tuple] = {}
    for blk in blocks:
        params, ok = params_for(variant, theta, blk["config"])
        try:
            laws[blk["source_id"]] = (make_section_law(params), params, ok)
        except Exception:
            laws[blk["source_id"]] = (None, params, False)

    parts: List[np.ndarray] = []
    for blk in blocks:
        law, _params, ok = laws[blk["source_id"]]
        if not ok:
            parts.append(np.full(blk["delta_meas"].shape, 1.0e3))
            continue
        d_pred = np.atleast_1d(
            deflection_at_load(law, blk["P_N"], blk["span"], n_quad=n_quad)
        )
        parts.append(blk["w_point"] * (d_pred - blk["delta_meas"]) / blk["scale"])
    for blk in blocks:
        law, _params, ok = laws[blk["source_id"]]
        if not ok:
            parts.append(np.array([1.0e3]))
            continue
        parts.append(np.array([
            blk["w_peak"] * (peak_load(law, blk["span"]) - blk["P_u_meas"])
            / blk["peak_scale"]
        ]))
    priors: List[float] = []
    for blk in blocks:
        _law, params, ok = laws[blk["source_id"]]
        if not ok:
            priors.extend([1.0e3, 1.0e3, 1.0e3])
            continue
        priors.append(max(0.0, (1.0e6 - params["EI0"]) / 1.0e6))
        priors.append(max(0.0, 0.02 - params["r1"]) + max(0.0, params["r1"] - 0.98))
        priors.append(max(0.0, (params["Mcr"] - params["Mu"]) / max(params["Mu"], 1e-30)))
    parts.append(w_prior * np.asarray(priors))

    if variant.ei0_ceiling_penalty > 0.0:
        # One-sided admissibility cap: the identified elastic stiffness may not exceed the
        # section's full-composite-action mechanical stiffness (analysis.elastic_ceiling). The
        # measured low-load deflection of the deep first-campaign tubes reads stiffer than the
        # section can bend; without this the softplus EI0 reproduces that unphysically.
        pen: List[float] = []
        for blk in blocks:
            _law, params, ok = laws[blk["source_id"]]
            if not ok:
                pen.append(0.0)
                continue
            ceil = elastic_ceiling_Nmm2(blk["config"], variant.ceiling_moduli)
            pen.append(max(0.0, params["EI0"] / ceil - 1.0))
        parts.append(variant.ei0_ceiling_penalty * np.asarray(pen))

    if variant.ridge_lambda > 0.0 and len(ridge_idx):
        parts.append(np.sqrt(variant.ridge_lambda) * np.asarray(theta)[list(ridge_idx)])
    return np.concatenate(parts)


# ============================================================================================
# Scoring
# ============================================================================================
def _law_deviation(params: Dict[str, float]) -> float:
    """``M(kappa_u)/Mu - 1`` of the DELIVERED law: the clamp's fingerprint.

    Zero is what ``section_law``'s docstring promises. The archived family misses it in 9 of 22
    cells (worst 52.5%) and never reports that it did.
    """
    law = make_section_law(params)
    return float(law._backbone.moment[-1] / law.Mu - 1.0)


def score(variant: Variant, theta: np.ndarray, curves, span_cfg, n_quad: int) -> List[dict]:
    """Per-config predicted-vs-measured peak, curve misfit, params, and law consistency.

    ``span_cfg`` is a scalar span or a per-campaign mapping (:func:`span_for`).
    """
    rows: List[dict] = []
    for mc in curves:
        params, ok = params_for(variant, theta, mc.config)
        law = make_section_law(params)
        span = span_for(span_cfg, mc.config.system)
        p_pred_kN = peak_load(law, span) / KN_TO_N
        d_pred = np.atleast_1d(
            deflection_at_load(law, mc.load_kN * KN_TO_N, span, n_quad=n_quad)
        )
        resid = d_pred - mc.deflection_mm
        denom = float(np.linalg.norm(mc.deflection_mm))
        rows.append({
            "source_id": mc.source_id,
            "config_key": list(mc.config.key()),
            "span_mm": float(span),
            "peak_meas_kN": float(mc.peak_kN),
            "peak_pred_kN": float(p_pred_kN),
            "peak_rel_err": float((p_pred_kN - mc.peak_kN) / mc.peak_kN),
            "curve_rel_l2": float(np.linalg.norm(resid) / denom) if denom > 0 else float("nan"),
            "curve_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
            "n_points": int(mc.deflection_mm.size),
            "section_params": {k: float(v) for k, v in params.items()},
            "feasible": bool(ok),
            "law_dev_Mku_over_Mu": _law_deviation(params),
        })
    return rows


# ============================================================================================
# The fit
# ============================================================================================
@dataclass
class FitResult:
    variant: str
    theta: np.ndarray
    A: np.ndarray
    cost: float
    condition_number: float
    n_coefficients: int
    calibration: List[dict] = field(default_factory=list)
    held_out: List[dict] = field(default_factory=list)
    restarts: List[dict] = field(default_factory=list)
    wall_s: float = 0.0

    def summary(self) -> dict:
        """The four numbers every variant is judged on, plus the consistency check."""
        ce = np.array([r["peak_rel_err"] for r in self.calibration])
        he = np.array([r["peak_rel_err"] for r in self.held_out])
        every = self.calibration + self.held_out
        devs = np.abs([r["law_dev_Mku_over_Mu"] for r in every])
        return {
            "variant": self.variant,
            "n_coefficients": int(self.n_coefficients),
            "condition_number": float(self.condition_number),
            "calib_worst_pct": float(np.abs(ce).max() * 100),
            "calib_mae": float(np.abs(ce).mean()),
            "calib_signed_mean_pct": float(ce.mean() * 100),
            "held_worst_pct": float(np.abs(he).max() * 100),
            "held_mae": float(np.abs(he).mean()),
            "held_signed_mean_pct": float(he.mean() * 100),
            "clamped_cells": int((devs > 1e-6).sum()),
            "n_cells": int(devs.size),
            "worst_law_dev_pct": float(devs.max() * 100),
            "cost": float(self.cost),
            "wall_s": float(self.wall_s),
        }

    def to_json(self) -> dict:
        return {
            "summary": self.summary(),
            "variant": self.variant,
            "theta": np.asarray(self.theta).tolist(),
            "A": np.asarray(self.A).tolist(),
            "calibration": self.calibration,
            "held_out": self.held_out,
            "restarts": self.restarts,
        }


def _run_one_restart(task):
    """One seeded ``least_squares`` from its own init. Module-level so it pickles for a pool."""
    variant, curves, cfg, restart_index, seed = task
    blocks, _span_cfg, w_prior, n_quad = build_blocks(curves, cfg)
    lo, hi = theta_bounds(variant)
    ridge_idx = _ridge_slot_indices(variant)
    opt = cfg.get("optimizer", {}) or {}
    perturb = float((cfg.get("feature_map", {}) or {}).get("init_perturb_scale", 0.05))
    x0 = np.clip(theta_init(variant, seed, perturb), lo, hi)
    t0 = time.time()
    sol = least_squares(
        residual, x0, args=(variant, blocks, w_prior, n_quad, ridge_idx),
        bounds=(lo, hi), method=str(opt.get("method", "trf")),
        max_nfev=int(opt.get("max_nfev", 4000)),
        xtol=float(opt.get("xtol", 1e-8)), ftol=float(opt.get("ftol", 1e-8)),
        gtol=float(opt.get("gtol", 1e-8)),
    )
    record = {
        "restart": restart_index, "seed": int(seed), "success": bool(sol.success),
        "status": int(sol.status), "cost": float(sol.cost),
        "nfev": int(getattr(sol, "nfev", -1)), "wall_s": round(time.time() - t0, 1),
    }
    return record, np.asarray(sol.x), float(sol.cost), np.asarray(sol.jac)


def fit_variant(
    variant_name: str, calib, held, cfg: dict, *, n_jobs: int = 1
) -> FitResult:
    """Multi-restart fit on calibration only; the held-out group is scored, never fitted.

    Restarts are independent and deterministic in their seed, so running them across processes
    gives exactly the serial result -- only the wall clock changes.
    """
    variant = VARIANTS[variant_name]
    t0 = time.time()
    opt = cfg.get("optimizer", {}) or {}
    n_restarts = int(opt.get("n_restarts", 6))
    rng = np.random.default_rng(int(opt.get("seed", 0)))
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_restarts)]

    tasks = [(variant, list(calib), cfg, i, s) for i, s in enumerate(seeds)]
    results = []
    if n_jobs > 1 and len(tasks) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks))) as ex:
                results = list(ex.map(_run_one_restart, tasks))
        except Exception:
            results = []
    if not results:
        results = [_run_one_restart(t) for t in tasks]
    results.sort(key=lambda r: r[0]["restart"])

    best_i = int(np.argmin([r[2] for r in results]))
    _rec, theta, cost, jac = results[best_i]

    _u, s, _vt = np.linalg.svd(jac, full_matrices=False)
    cond = float(s.max() / s[s > 0].min()) if np.any(s > 0) else float("inf")

    span_cfg = cfg.get("span_mm", DEFAULT_SPAN_MM)
    n_quad = int((cfg.get("forward", {}) or {}).get("n_quad", 64))
    return FitResult(
        variant=variant_name, theta=theta, A=unpack_theta(variant, theta), cost=cost,
        condition_number=cond, n_coefficients=len(free_slots(variant)),
        calibration=score(variant, theta, calib, span_cfg, n_quad),
        held_out=score(variant, theta, held, span_cfg, n_quad),
        restarts=[r[0] for r in results], wall_s=round(time.time() - t0, 1),
    )


# ============================================================================================
# CLI
# ============================================================================================
def _print_report(result: FitResult) -> None:
    variant = VARIANTS[result.variant]
    s = result.summary()
    names = feature_names(variant.feature_dim)
    print(f"=== {result.variant} ===   {s['n_coefficients']} coefficients, "
          f"{s['wall_s']}s  (ridge_lambda={variant.ridge_lambda})")
    print(f"  cond               {s['condition_number']:.4f}")
    print(f"  law M(ku)=Mu       {s['clamped_cells']}/{s['n_cells']} cells violate it; "
          f"worst {s['worst_law_dev_pct']:.4f}%")
    print(f"  calibration        worst {s['calib_worst_pct']:6.2f}%   MAE {s['calib_mae']:.4f}   "
          f"signed mean {s['calib_signed_mean_pct']:+.2f}%")
    print(f"  held out (K2)      worst {s['held_worst_pct']:6.2f}%   MAE {s['held_mae']:.4f}   "
          f"signed mean {s['held_signed_mean_pct']:+.2f}%")
    print(f"  coefficients A ({' x '.join([str(len(variant.rows)), str(variant.feature_dim)])}):")
    print("        " + "".join(f"{n:>10s}" for n in names))
    for i, row in enumerate(variant.rows):
        print(f"  {row:6s}" + "".join(f"{v:10.4f}" for v in result.A[i]))
    at_bound = [(variant.rows[i], names[c], float(result.A[i, c]))
                for (i, c) in free_slots(variant) if abs(result.A[i, c]) > 11.99]
    if at_bound:
        print(f"  !! at the +-12 pre-activation box bound: {at_bound}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default=PRODUCTION_VARIANT, choices=sorted(VARIANTS),
                        help=f"which family to fit (default: the production family "
                             f"{PRODUCTION_VARIANT})")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None,
                        help="JSON output path (default results/analysis/<variant>.json). "
                             "Each variant gets its own file: a shared path would silently "
                             "overwrite the previous run's numbers.")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    calib, held = load_partitions(cfg)
    result = fit_variant(args.variant, calib, held, cfg, n_jobs=args.n_jobs)
    _print_report(result)

    out = Path(args.out) if args.out else (RESULTS_DIR / f"{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=1), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
