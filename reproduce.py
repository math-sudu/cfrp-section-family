"""Reproduce tube and beam predictions using the supplied fitted parameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from frp_flexure import beam, tube_mechanics, tube_network
from frp_flexure.data import first_crossing, load_beams, load_strains, load_tubes, parameters


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def tube_results(parameter_path=None):
    traces = load_tubes()
    by_name = {t.key: t for t in traces}
    result = {}
    for name, model in parameters("tubes", parameter_path).items():
        tests = [by_name[k] for k in model["testing"]]
        central = {k: [] for k in ("A", "S", "M_T")}
        common = {k: [] for k in central}
        windows = {k: [] for k in central}
        rows = []

        def prediction(kind, trace, loads):
            if kind == "M_T":
                return tube_mechanics.predict(model[kind], model["peak_model"], trace, loads)
            return tube_network.predict(model["peak_model"], model[kind], trace.features, loads)

        for t in tests:
            for pair in tube_mechanics.QUERY_PAIRS:
                loads = np.array(pair) * t.peak
                observed = first_crossing(t.branch, loads[1]) - first_crossing(t.branch, loads[0])
                for kind in central:
                    estimate = prediction(kind, t, loads)
                    if estimate is not None:
                        central[kind].append(estimate - observed)
            if not t.wrap:
                continue
            bare = next(r for r in traces if (r.depth, r.filled, r.wrap) == (t.depth, t.filled, 0))
            loads = tube_mechanics.common_loads(t, bare)
            observed = first_crossing(t.branch, loads[1]) - first_crossing(t.branch, loads[0])
            reference = first_crossing(bare.branch, loads[1]) - first_crossing(bare.branch, loads[0])
            estimates = {kind: prediction(kind, t, loads) for kind in central}
            for kind, estimate in estimates.items():
                common[kind].append(estimate - observed)
            peak = tube_network.peaks(model["peak_model"], t.features)[0, 1]
            lower, upper = max(.2 * peak, .2 * bare.peak), min(.8 * peak, .8 * bare.peak)
            for p in np.linspace(lower, upper, 21)[1:]:
                actual = first_crossing(t.branch, p) - first_crossing(t.branch, lower)
                for kind in central:
                    windows[kind].append(prediction(kind, t, [lower, p]) - actual)
            row = {"configuration": t.key, "loads_kn": loads.tolist(),
                   "measured_saving_mm": reference - observed,
                   "predicted_saving_mm": {k: reference - v for k, v in estimates.items()},
                   "prediction_window_kn": [float(lower), float(upper)]}
            if "affine_amplitude" in model:
                amplitude = np.logaddexp(0., np.r_[t.features, 1.] @ model["affine_amplitude"])
                queries = np.array([loads] + [[lower, p] for p in np.linspace(lower, upper, 21)[1:]])
                predicted = amplitude * tube_mechanics.fraction(
                    model["M_T"]["shape"]["theta"], [t] * len(queries), queries,
                    np.full(len(queries), peak))
                measured = [first_crossing(t.branch, b) - first_crossing(t.branch, a) for a, b in queries]
                row["affine_errors_mm"] = (predicted - measured).tolist()
                row["affine_saving_mm"] = float(reference - predicted[0])
            rows.append(row)
        result[name] = {"central_increment_rmse_mm": {k: rmse(v) for k, v in central.items()},
                        "common_force_saving_rmse_mm": {k: rmse(v) for k, v in common.items()},
                        "predicted_window_rmse_mm": {k: rmse(v) for k, v in windows.items()},
                        "wrapping_savings": rows}
        if "affine_amplitude" in model:
            result[name]["affine_selected_rmse_mm"] = rmse([r["affine_errors_mm"][0] for r in rows])
            result[name]["affine_window_rmse_mm"] = rmse([e for r in rows for e in r["affine_errors_mm"][1:]])
    return result


def make_beam(model, names, loads, compliance, options):
    constructor = {"M_R": beam.ConditionedBearing, "M_0": beam.ConditionedCoupled,
                   "M_B": beam.ConditionedCoupled, "M_F": beam.ConditionedFiber,
                   "M_N": beam.ConditionedNeural}[model]
    return constructor(names, loads, compliance, **options)


def timing(function):
    for _ in range(2):
        function()
    times = []
    for _ in range(7):
        start = perf_counter()
        function()
        times.append(perf_counter() - start)
    return {"median_seconds": float(np.median(times)), "runs_seconds": times}


def beam_results(models=("M_R", "M_0", "M_B", "M_N"), parameter_path=None, benchmark=False):
    inputs = load_beams()
    fitted = parameters("beams", parameter_path)
    compliance = beam.observed_compliances(inputs)
    names = fitted["testing"]
    observed = np.array([inputs[n]["equal_force_mean"][-len(beam.EVALUATION_LOADS):] for n in names])
    strains = load_strains("ST65S100")
    result = {}
    for model in models:
        solution = fitted["models"][model]
        theta = torch.tensor(solution["log_parameters"])
        response = make_beam(model, names, beam.EVALUATION_LOADS, compliance, solution["prediction_options"])
        gauge = make_beam(model, ["ST65S100"], [strains["load_kn"]], compliance, solution["prediction_options"])

        def predict_strain():
            if model == "M_R":
                return beam.reinforcement_strain(gauge, theta, strains["positions_mm"])
            return gauge.strain(theta, [strains["positions_mm"]])[0]

        with torch.no_grad():
            predicted = response.predict(theta).numpy()
            strain = predict_strain().numpy()
            record = {"aggregate_rmse_mm": rmse(predicted - observed),
                      "configurations": {n: {"predicted_mm": p.tolist(), "observed_mm": o.tolist(),
                                              "rmse_mm": rmse(p - o)}
                                         for n, p, o in zip(names, predicted, observed, strict=True)},
                      "predicted_strain_microstrain": strain.tolist(),
                      "observed_strain_microstrain": strains["microstrain"]}
            if benchmark:
                record["timing_33_load_states"] = timing(lambda: response.predict(theta))
                record["timing_two_strains"] = timing(predict_strain)
            result[model] = record
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", "tubes", "beams"), default="all")
    parser.add_argument("--models", nargs="+", choices=("M_R", "M_0", "M_F", "M_B", "M_N"),
                        default=["M_R", "M_0", "M_B", "M_N"])
    parser.add_argument("--tube-parameters", type=Path)
    parser.add_argument("--beam-parameters", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/reproduction.json"))
    args = parser.parse_args()
    result = {}
    if args.section in ("all", "tubes"):
        result["tubes"] = tube_results(args.tube_parameters)
    if args.section in ("all", "beams"):
        result["beams"] = beam_results(args.models, args.beam_parameters, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if "tubes" in result:
        row = result["tubes"]["leave_wrap_2"]
        print("Tube savings RMSE (mm):", row["common_force_saving_rmse_mm"])
        print("Affine / neural window RMSE (mm):", row["affine_window_rmse_mm"],
              row["predicted_window_rmse_mm"]["M_T"])
    if "beams" in result:
        print("Beam RMSE (mm):", {k: v["aggregate_rmse_mm"] for k, v in result["beams"].items()})
    print("Results:", args.output)


if __name__ == "__main__":
    main()
