"""Refit tube networks or beam laws from the experimental inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from frp_flexure import beam, tube_mechanics, tube_network
from frp_flexure.data import load_beams, load_tubes, parameters


def peak_fit(traces, penalty, seed):
    features = np.array([t.features for t in traces])
    target = np.log([[t.table_peak, t.peak] for t in traces])
    center = target.mean(axis=0)
    target -= center
    start = tube_network.initial_parameters(3, 2, seed)
    weight = 1 / np.sqrt(target.size)
    regularizer = np.sqrt(penalty) * np.eye(len(start))[:-2]
    def residual(theta):
        value, _ = tube_network.raw_network(theta, features, 2, width=4)
        return np.r_[((value - target) * weight).ravel(), regularizer @ theta]
    def jacobian(theta):
        _, jac = tube_network.raw_network(theta, features, 2, width=4)
        return np.vstack(((jac * weight).reshape(-1, len(theta)), regularizer))
    fit = least_squares(residual, start, jac=jacobian, max_nfev=800,
                         ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return dict(theta=fit.x.tolist(), center=center.tolist(), penalty=penalty,
                seed=seed, cost=float(fit.cost), head="peaks")


def refit_tubes(model, protocol, path=None):
    fitted = parameters("tubes", path)
    record = fitted[protocol]
    traces = {t.key: t for t in load_tubes()}
    training = [traces[k] for k in record["training"]]
    seeds = tube_network.SEEDS
    if model == "peaks":
        record["peak_model"] = min(
            (peak_fit(training, record["peak_model"]["penalty"], s) for s in seeds), key=lambda r: r["cost"])
        for fold, current in record["inner_peaks"].items():
            subset = [t for t in training if record["folds"][t.key] != int(fold)]
            record["inner_peaks"][fold] = min(
                (peak_fit(subset, current["penalty"], s) for s in seeds), key=lambda r: r["cost"])
    elif model == "M_T":
        result = tube_mechanics.fit_protocol(record, traces)
        record["M_T"] = {"shape": result["shape"], "amplitude": result["amplitude"]}
        if protocol == "leave_wrap_2":
            shape = result["shape"]
            design = np.array([np.r_[t.features, 1.] for t in training])
            target = np.array(shape["amplitudes_mm"])
            weight = np.array(shape["amplitude_weights"])
            fit = least_squares(lambda x: weight * (np.logaddexp(0., design @ x) - target),
                                [0., 0., 0., np.log(np.expm1(target.mean()))])
            if not fit.success:
                raise RuntimeError(fit.message)
            record["affine_amplitude"] = fit.x.tolist()
    elif model in ("A", "S"):
        kind = {"A": "amplitude", "S": "shared"}[model]
        scores = {}
        for penalty in tube_network.PENALTIES:
            error, count = 0., 0
            for fold in range(3):
                train = [t for t in training if record["folds"][t.key] != fold]
                valid = [t for t in training if record["folds"][t.key] == fold]
                candidate = min((tube_network.fit_response(train, kind, penalty, s) for s in seeds),
                                key=lambda r: r["cost"])
                score = tube_network.validation_score(record["inner_peaks"][str(fold)], candidate, valid)
                error += score["squared_error_sum_mm2"]
                count += score["covered"]
            scores[penalty] = error / count
        penalty = min(scores, key=scores.get)
        record[model] = min((tube_network.fit_response(training, kind, penalty, s) for s in seeds),
                            key=lambda r: r["cost"])
    else:
        raise ValueError("Tube model must be peaks, A, S or M_T")
    return fitted


def refit_beams(model, path=None):
    fitted = parameters("beams", path)
    inputs = load_beams()
    names = fitted["training"]
    compliance = beam.observed_compliances(inputs)
    if model == "M_N":
        baseline = {"parameters": np.exp(fitted["models"]["M_B"]["log_parameters"])}
        start = beam.initial_neural_parameters(baseline)
        calibration = beam.JointCalibration(inputs)
        def objective(values):
            theta = torch.tensor(values, requires_grad=True)
            loss = calibration(theta)
            loss.backward()
            return loss.item(), theta.grad.numpy()
        solved = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=beam.BOUNDS,
                          options={"maxiter": 2000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 40})
        if not solved.success:
            raise RuntimeError(solved.message)
        solution = {"log_parameters": solved.x.tolist()}
    elif model == "M_R":
        solution = beam.fit_moment(names, inputs, compliance, True)
    elif model in ("M_0", "M_B", "M_F"):
        solution = beam.fit_power(names, inputs, compliance, True, separate=model == "M_0",
                                   batch_factory=beam.ConditionedFiber if model == "M_F" else None)
    else:
        raise ValueError("Beam model must be M_R, M_0, M_B, M_F or M_N")
    if not solution.get("success", True):
        raise RuntimeError(solution["message"])
    fitted["models"][model]["log_parameters"] = solution["log_parameters"]
    return fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("tubes", "beams"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", default="leave_wrap_2")
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (refit_tubes(args.model, args.protocol, args.parameters) if args.component == "tubes"
              else refit_beams(args.model, args.parameters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Fitted parameters:", args.output)


if __name__ == "__main__":
    main()
