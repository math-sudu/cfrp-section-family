"""Neural amplitudes, peak predictors and normalized response shapes."""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import expit, logsumexp
from frp_flexure.data import first_crossing
WIDTH = 4
PENALTIES = (1e-4, 1e-3, 1e-2)
SEEDS = (11, 29, 47)
MAX_NFEV = 800
DISPLACEMENT_SCALE_MM = 5.0
DOMAIN_ROUNDOFF = 8 * np.finfo(float).eps
SIZES = {"amplitude": (21, 0), "shared": (21, 12)}
NODES, WEIGHTS = leggauss(16)


def raw_network(theta, x, outputs, width=6):
    """Return values and Jacobian with shape (sample, output, parameter)."""
    n, dim = x.shape
    nw = dim * width
    nv = width * outputs
    w = theta[:nw].reshape(dim, width)
    b = theta[nw:nw + width]
    v = theta[nw + width:nw + width + nv].reshape(width, outputs)
    a = theta[-outputs:]
    h = np.tanh(x @ w + b)
    dh = 1 - h * h
    value = h @ v + a
    jac = np.zeros((n, outputs, len(theta)))
    for k in range(outputs):
        jac[:, k, :nw] = (x[:, :, None] * (dh * v[:, k])[:, None, :]).reshape(n, nw)
        jac[:, k, nw:nw + width] = dh * v[:, k]
        jac[:, k, nw + width + k:nw + width + nv:outputs] = h
        jac[:, k, -outputs + k] = 1
    return value, jac


def initial_parameters(dim, outputs, seed):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0, 0.6, dim * WIDTH), np.zeros(WIDTH),
                 rng.normal(0, 0.1, WIDTH * outputs), np.zeros(outputs)]


def domain_endpoints(values):
    """Accept the closed load domain and clip only floating-point overshoot."""
    values = np.asarray(values, dtype=float)
    accepted = (np.isfinite(values) & (values >= 0.2 - DOMAIN_ROUNDOFF)
                & (values <= 0.8 + DOMAIN_ROUNDOFF))
    return np.where(accepted, np.clip(values, 0.2, 0.8), values), accepted


def response_arrays(traces):
    features, loads, increments, weights = [], [], [], []
    for trace in traces:
        b, peak = trace.branch, trace.peak
        anchor = first_crossing(b, 0.2 * peak)
        rows = b[(b[:, 1] > 0.2 * peak) & (b[:, 1] < 0.8 * peak)]
        rows = np.vstack((rows, [first_crossing(b, 0.8 * peak), 0.8 * peak]))
        features.extend([trace.features] * len(rows))
        loads.extend(rows[:, 1] / peak)
        increments.extend(rows[:, 0] - anchor)
        weights.extend([1 / np.sqrt(len(traces) * len(rows))] * len(rows))
    return np.array(features), np.array(loads), np.array(increments), np.array(weights)


def peaks(model, features):
    features = np.asarray(features).reshape(-1, 3)
    value, _ = raw_network(np.array(model["theta"]), features, 2, width=WIDTH)
    return np.exp(value + np.array(model["center"]))


def amplitude(theta, features, kind):
    features = np.asarray(features).reshape(-1, 3)
    if kind == "common":
        value = np.full(len(features), theta[0])
        jac = np.ones((len(features), 1))
    else:
        raw, derivative = raw_network(theta, features, 1, WIDTH)
        value, jac = raw[:, 0], derivative[:, 0, :]
    return np.logaddexp(0, value), expit(value)[:, None] * jac


def shape_values(theta, features, q, kind):
    features = np.asarray(features).reshape(-1, 3)
    inputs = ((q.ravel() - 0.5) / 0.3)[:, None]
    if kind == "conditional":
        inputs = np.column_stack((inputs, np.repeat(features, q.shape[1], axis=0)))
    value, jac = raw_network(np.r_[theta, 0.0], inputs, 1, WIDTH)
    return value[:, 0].reshape(q.shape), jac[:, 0, :-1].reshape((*q.shape, len(theta)))


def shape_fraction(theta, features, lower, upper, kind, order=16):
    """Integral of rho, normalized over the full window, and its Jacobian."""
    features = np.asarray(features).reshape(-1, 3)
    lower, upper = np.broadcast_arrays(np.atleast_1d(lower), np.atleast_1d(upper))
    (lower, upper), accepted = domain_endpoints([lower, upper])
    if not np.all(accepted):
        raise ValueError("Query is outside the predicted 20--80% load domain")
    if np.any(upper < lower):
        raise ValueError("Query loads must be ordered")
    if kind == "amplitude":
        return (upper - lower) / 0.6, np.empty((len(lower), 0))
    nodes, weights = (NODES, WEIGHTS) if order == 16 else leggauss(order)
    full = np.broadcast_to(0.5 + 0.3 * nodes, (len(lower), order))
    query = (lower[:, None] + upper[:, None]) / 2 + (
        upper - lower)[:, None] * nodes / 2
    full_g, full_j = shape_values(theta, features, full, kind)
    query_g, query_j = shape_values(theta, features, query, kind)
    log_weights = full_g + np.log(0.3 * weights)
    log_z = logsumexp(log_weights, axis=1)
    denominator_gradient = np.einsum(
        "nq,nqp->np", np.exp(log_weights - log_z[:, None]), full_j)
    query_weights = np.exp(query_g - log_z[:, None]) * (
        (upper - lower)[:, None] * weights / 2)
    fraction = query_weights.sum(axis=1)
    jac = np.einsum("nq,nqp->np", query_weights,
                    query_j - denominator_gradient[:, None, :])
    return fraction, jac


def response(theta, features, lower, upper, kind, order=16):
    n_amp, _ = SIZES[kind]
    a, ja = amplitude(theta[:n_amp], features, kind)
    fraction, jf = shape_fraction(theta[n_amp:], features, lower, upper, kind, order)
    return a * fraction, np.column_stack((ja * fraction[:, None], jf * a[:, None]))


def initial(kind, seed):
    n_amp, n_shape = SIZES[kind]
    a = np.array([0.0]) if n_amp == 1 else initial_parameters(3, 1, seed)
    a[-1] = np.log(np.expm1(4.8))
    shape = (initial_parameters(4 if kind == "conditional" else 1, 1, seed)[:-1]
             if n_shape else np.empty(0))
    return np.r_[a, shape]


def fit_response(training, kind, penalty, seed):
    features, upper, target, weights = response_arrays(training)
    lower = np.full(len(upper), 0.2)
    weights /= DISPLACEMENT_SCALE_MM
    theta = initial(kind, seed)
    n_amp, _ = SIZES[kind]
    penalty_mask = np.ones(len(theta), dtype=bool)
    penalty_mask[n_amp - 1] = False
    regularizer = np.sqrt(penalty) * np.eye(len(theta))[penalty_mask]

    def residual(parameters):
        value, _ = response(parameters, features, lower, upper, kind)
        return np.r_[(value - target) * weights, regularizer @ parameters]

    def jacobian(parameters):
        _, jac = response(parameters, features, lower, upper, kind)
        return np.vstack((jac * weights[:, None], regularizer))

    start = perf_counter()
    fit = least_squares(residual, theta, jac=jacobian, max_nfev=MAX_NFEV,
                        ftol=1e-8, xtol=1e-8, gtol=1e-8)
    data_residual = residual(fit.x)[:len(target)]
    return dict(
        head=kind, theta=fit.x.tolist(), penalty=penalty, seed=seed,
        cost=float(fit.cost), data_mse=float(data_residual @ data_residual),
        nfev=int(fit.nfev), status=int(fit.status), termination=fit.message,
        optimality=float(fit.optimality), elapsed_seconds=perf_counter() - start)


def predict(peak_model, model, features, loads):
    predicted_peak = peaks(peak_model, features)[0, 1]
    (lower, upper), accepted = domain_endpoints(np.asarray(loads) / predicted_peak)
    if not np.all(accepted):
        return None
    return float(response(np.asarray(model["theta"]), features, [lower], [upper],
                          model["head"])[0][0])


def validation_score(peak_model, model, validation):
    squared_error = []
    for trace in validation:
        for fractions in QUERY_PAIRS[:3]:
            loads = np.asarray(fractions) * trace.peak
            value = predict(peak_model, model, trace.features, loads)
            if value is not None:
                target = first_crossing(trace.branch, loads[1]) - first_crossing(
                    trace.branch, loads[0])
                squared_error.append((value - target)**2)
    return dict(
        covered=len(squared_error), total=3 * len(validation),
        squared_error_sum_mm2=float(sum(squared_error)))
