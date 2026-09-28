"""Recompute the section-mapped lining example with incremental dissipation.

Run one cell with --case baseline or --case bare_018 (also filled, wrapped,
hybrid; coverages 018 and 036). The output checkpoints the last converged
increment. All response measures use the jointly minimized phase field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
from analysis.paths import ensure_local_importable
ensure_local_importable()
from fem_check.solver import solve_load_step, solve_alpha_qp, alpha_tolerance
from analysis.crack_metrics import crack_length_from_elements
from analysis.lining_inputs import inputs, SCHEMES

OUTPUT = PAPER / "results/analysis/lining"


def measures(problem, alpha, centroids):
    _, length = crack_length_from_elements(
        problem.elem_mean(alpha), centroids, problem.T, problem.X, problem.Y,
        threshold=0.5, min_cluster=3)
    return dict(crack_length_mm=length,
                damage_ratio=float(np.mean(alpha > 0.95)),
                crack_surface_mm=float(np.sum(problem.area * 3 / 8 * (
                    problem.elem_mean(alpha) / problem.l0
                    + problem.l0 * sum(g*g for g in problem.alpha_grad(alpha))))))


def run(case, steps, max_lambda, stop_step, max_sweeps=600, output=OUTPUT):
    out = Path(output) / case
    out.mkdir(parents=True, exist_ok=True)
    p, seed, section, config, meta = inputs(case, steps, max_lambda)
    centroids = np.column_stack([p.X[p.T].mean(axis=1), p.Y[p.T].mean(axis=1)])
    checkpoint = out / "checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            if float(z["max_lambda"]) != max_lambda or int(z["steps"]) != steps:
                raise ValueError("Checkpoint belongs to a different load schedule")
            u, v, alpha = z["u"].copy(), z["v"].copy(), z["alpha"].copy()
            rows = json.loads(str(z["rows"]))
            initial = json.loads(str(z["initial"]))
        for row in rows:
            row["growth_trigger"] = row["damage_ratio"] > max(
                1.5 * initial["damage_ratio"], initial["damage_ratio"] + 1e-4)
        start = len(rows)
    else:
        u = np.zeros(p.n_nodes)
        v = np.zeros(p.n_nodes + p.n_rotations)
        # Establish the unloaded diffuse seed before measuring extension.
        eta = p.viscous_eta
        p.viscous_eta = 0.
        H, f, _ = p.alpha_quadratic(u, v, seed)
        alpha, residual, _, converged = solve_alpha_qp(
            H, f, seed, seed, np.ones_like(seed), alpha_tolerance(p))
        if not converged:
            raise RuntimeError(f"Unloaded seed did not equilibrate: residual={residual}")
        p.viscous_eta = eta
        initial = measures(p, alpha, centroids)
        rows, start = [], 0
        np.savez_compressed(out / "initial.npz", X=p.X, Y=p.Y, T=p.T,
                            alpha=alpha, raw_seed=seed, boundary=p.bnd_idx)
    params = section.parameters() if section else None
    settings = dict(case=case, steps=steps, max_lambda=max_lambda,
                    eta_Pa_s=p.viscous_eta, dt_s=p.viscous_dt,
                    E_MPa=p.E, nu=p.nu, Gc_N_mm=p.Gc_mm, l0_mm=p.l0,
                    residual_stiffness=p.residual_stiffness,
                    max_stagger_sweeps=max_sweeps,
                    damage_source="jointly minimized nodal field and its element averages",
                    initial=initial, section=params, boundary=meta["bc_dict"],
                    section_mapping=config.get("double_layer"),
                    formulation="eta*(alpha-alpha_prev)^2/(2*dt)",
                    overlay_element="planar Timoshenko frame with independent section rotations")
    if p.overlay:
        settings["smeared_stiffness"] = dict(
            EA_N_per_mm=float(p.overlay.EA_N),
            kGA_N_per_mm=float(p.overlay.kGA_N),
            EI_N_mm=float(p.overlay.EI_N_mm2))
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    for i in range(start, min(steps, stop_step or steps)):
        lam = max_lambda * (i + 1) / steps
        u, v, new_alpha, rec = solve_load_step(p, i, lam, u, v, alpha, alpha,
                                              max_sweeps=max_sweeps)
        print(f"{case} lambda={lam:.3f} converged={rec.converged} "
              f"sweeps={rec.n_stagger} seconds={rec.time_sec:.1f}", flush=True)
        if not rec.converged:
            raise RuntimeError(f"Unconverged increment: {rec.as_dict()}")
        alpha = new_alpha
        row = rec.as_dict() | measures(p, alpha, centroids)
        row["growth_trigger"] = row["damage_ratio"] > max(
            1.5 * initial["damage_ratio"], initial["damage_ratio"] + 1e-4)
        if section:
            forces = p.frame_forces(u, v)
            kmax = float(np.max(np.abs(forces[:, [2, 5]]))) / p.overlay.EI_N_mm2
            row.update(max_curvature_mm_inv=kmax,
                       elastic_moment_ratio=section.EI0 * kmax / section.Mu,
                       curvature_ratio=kmax / section.kappa_u)
        rows.append(row)
        if abs(lam - 1.) < 1e-9 or i == steps-1 or (row["growth_trigger"] and not any(r["growth_trigger"] for r in rows[:-1])):
            np.savez_compressed(out / f"field_{i+1:03d}.npz", u=u, v=v, alpha=alpha,
                                lambda_value=lam)
        tmp = out / "checkpoint.tmp.npz"
        np.savez_compressed(tmp, u=u, v=v, alpha=alpha, rows=json.dumps(rows),
                            initial=json.dumps(initial), steps=steps, max_lambda=max_lambda)
        os.replace(tmp, checkpoint)
        (out / "history.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if len(rows) == steps:
        result = dict(case=case, onset_lambda=next((r["lmbda"] for r in rows if r["growth_trigger"]), None),
                      all_converged=all(r["converged"] for r in rows),
                      initial=initial, design=next(r for r in rows if abs(r["lmbda"]-1.) < 1e-9),
                      final=rows[-1], total_seconds=sum(r["time_sec"] for r in rows))
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True,
                        choices=["baseline"] + [f"{s}_{c}" for s in SCHEMES for c in ("018", "036")])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-lambda", type=float, default=2.)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--max-sweeps", type=int, default=600)
    parser.add_argument("--out", type=Path, default=OUTPUT,
                        help="Result directory (each case gets its own subdirectory)")
    args = parser.parse_args()
    run(args.case, args.steps, args.max_lambda, args.stop_step, args.max_sweeps, args.out)
