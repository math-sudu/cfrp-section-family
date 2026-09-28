"""Staggered alternate-minimization solver for the lining phase-field functional.

Scheme (documented per acceptance A5):

  * One load step minimizes E_total(u, alpha) = E_el + E_d + E_visco + E_bc
    over (u, alpha) subject to  max(hist_alpha, 0) <= alpha <= 1  by ALTERNATE
    MINIMIZATION (staggered): a u-step at fixed alpha, then an alpha-step at
    fixed u, swept until BOTH block residuals hold at the SAME point:
        max |dE/du| <= u_grad_tol   and   max |projected dE/dalpha| <= alpha_pg_tol
    (block optimality = the staggered fixed point).  Irreversibility (alpha
    non-decreasing across load steps) and the [0, 1] bounds are enforced
    EXACTLY as bound constraints of the alpha block; the Deep Ritz
    irreversibility penalty term therefore evaluates to zero on every iterate
    and the minimized functional is the same.

  * u-step: Newton iteration with the CONSISTENT tangent of the fixed-alpha
    split energy (fem_core.elastic_tangent; the fixed-alpha spectral-split
    density is convex in strain, so the tangent is SPD), factored with SuperLU
    (scipy splu) plus a relative Tikhonov ridge that only conditions the
    LINEAR SOLVE (the functional and its gradient are untouched; convergence
    is measured on the true gradient), and an Armijo backtracking line search
    on the true energy.

  * alpha-step: the subproblem at fixed u is a strictly convex
    bound-constrained QP (fem_core.alpha_quadratic; strict convexity from the
    viscous term).  Solved by a primal-dual active-set method (PDAS): guess
    active bounds from the primal-dual residuals, pin them, solve the free-set
    reduced system with SuperLU, update the multipliers, iterate until the
    active sets repeat; the reported residual is the true projected gradient
    of the QP at the returned (feasible) point.  A cycling guard falls back to
    scipy L-BFGS-B; either way the exit test is the same projected-gradient
    tolerance.

Residuals and tolerances recorded per load step (acceptance A2):
  u_grad_inf      max |dE/du| at exit             <= u_grad_tol
  alpha_pg_inf    max |projected dE/dalpha| at exit <= alpha_pg_tol
  stag_dalpha_inf max |alpha move| in the last staggered sweep (diagnostic)
converged = true only when both block residuals meet tolerance; nothing
defaults, and a non-converged step is reported as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from fem_check.fem_core import FemProblem


# -------------------------- tolerances (single source) ----------------------

RTOL_U = 1.0e-7          # on max|dE/du| relative to the external load vector
ATOL_U = 1.0e-10         # floor, N/mm
RTOL_ALPHA = 1.0e-7      # on max|proj dE/dalpha| relative to the AT1 linear term
ATOL_ALPHA = 1.0e-12
MAX_NEWTON_U = 60
MAX_NEWTON_ALPHA = 60
MAX_STAGGER = 600
RIDGE_REL = 1.0e-10      # linear-solve conditioning ridge, relative to max diag


@dataclass
class StepRecord:
    step: int
    lmbda: float
    converged: bool
    u_grad_inf: float
    u_grad_tol: float
    alpha_pg_inf: float
    alpha_pg_tol: float
    stag_dalpha_inf: float
    n_stagger: int
    n_newton_u: int
    n_newton_alpha: int
    E_total: float
    E_el: float
    E_d: float
    E_hist: float
    E_visco: float
    E_trac: float
    E_spr: float
    E_ovl: float
    alpha_min: float
    alpha_max: float
    irrev_violation: float
    time_sec: float
    n_continuation: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ScaledFactor:
    lu: object
    scale: np.ndarray

    def solve(self, rhs):
        return self.scale * self.lu.solve(self.scale * rhs)


def _factor(K: sp.spmatrix):
    # Equilibrate mixed translational/rotational units before adding the
    # factorization ridge; the true gradient and tangent stay unchanged.
    diagonal = K.diagonal()
    # The imported mesh includes unused nodes with zero rows. Keep their
    # zero-load null degrees of freedom unscaled.
    scale = np.ones_like(diagonal)
    np.divide(1.0, np.sqrt(diagonal), out=scale, where=diagonal > 0.0)
    D = sp.diags(scale)
    n = K.shape[0]
    lu = splu((D @ K @ D + RIDGE_REL * sp.identity(n, format="csr")).tocsc())
    return ScaledFactor(lu, scale)


def _solve_refined(K: sp.spmatrix, b: np.ndarray, n_refine: int = 2) -> np.ndarray:
    """Solve K x = b with a ridged SuperLU factor + iterative refinement.

    The ridge conditions the factorization only; refinement against the TRUE
    matrix removes the ridge-induced residual (contraction ~ ridge/sigma_min
    per pass), so the returned solve is exact to machine precision and the
    ridge never becomes a floor on the outer residuals.
    """
    lu = _factor(K)
    x = lu.solve(b)
    for _ in range(n_refine):
        r = b - K @ x
        if not np.any(r):
            break
        x = x + lu.solve(r)
    return x


# ------------------------------- u-step -------------------------------------

class TangentCache:
    """Stale-factorization reuse for the u-Newton across staggered sweeps.

    The consistent tangent is SPD, so a factorization from a slightly earlier
    (u, alpha) state still yields descent directions (d = -P^{-1} g with P
    SPD); convergence is always measured on the TRUE gradient, so reuse can
    slow iteration but never corrupt the answer.  The cache refactors when
    progress stalls (residual reduction below CUT per iteration) or on age.
    """

    CUT = 0.2
    MAX_AGE = 12

    def __init__(self):
        self.lu = None
        self.age = 0

    def direction(self, problem: FemProblem, x, alpha, g):
        """Newton-ish direction, refactoring on demand.  Returns (d, fresh)."""
        if self.lu is not None and self.age < self.MAX_AGE:
            self.age += 1
            return self.lu.solve(-g), False
        return self.refactor(problem, x, alpha, g), True

    def refactor(self, problem: FemProblem, x, alpha, g):
        N = problem.n_nodes
        K = problem.elastic_tangent(x[:N], x[N:], alpha) + problem._K_bc
        self.lu = _factor(K)
        self.age = 0
        d = self.lu.solve(-g)
        r = -g - K @ d
        if np.any(r):
            d = d + self.lu.solve(r)   # iterative refinement against the true K
        return d


def solve_u(problem: FemProblem, u, v, alpha, lmbda, tol, cache: TangentCache | None = None):
    """Newton minimization of E(u; alpha fixed).  Returns (u, v, res, n_iter, ok)."""
    N = problem.n_nodes
    x = np.concatenate([u, v])
    if cache is None:
        cache = TangentCache()

    def energy(xv):
        E_el = problem.bulk_energies(xv[:N], xv[N:], alpha, alpha, alpha)[0]
        return E_el + sum(problem.bc_energies(xv[:N], xv[N:], lmbda))

    e_x = energy(x)
    res = np.inf
    res_prev = np.inf
    it = 0
    for it in range(1, MAX_NEWTON_U + 1):
        g = problem.grad_u(x[:N], x[N:], alpha, lmbda)
        res = float(np.max(np.abs(g)))
        if res <= tol:
            return x[:N], x[N:], res, it - 1, True
        if res > TangentCache.CUT * res_prev:
            cache.age = TangentCache.MAX_AGE   # stalling -> force fresh tangent
        res_prev = res
        d, fresh = cache.direction(problem, x, alpha, g)
        gd = float(g @ d)
        if not np.isfinite(gd) or gd >= 0.0:
            if not fresh:
                d = cache.refactor(problem, x, alpha, g)
                gd = float(g @ d)
            if not np.isfinite(gd) or gd >= 0.0:
                d = -g
                gd = float(g @ d)
        t = 1.0
        accepted = False
        for _ in range(50):
            x_try = x + t * d
            e_try = energy(x_try)
            if e_try <= e_x + 1.0e-4 * t * gd or abs(e_try - e_x) <= 1e-14 * max(1.0, abs(e_x)):
                x = x_try
                e_x = e_try
                accepted = True
                break
            t *= 0.5
        if not accepted:
            if not fresh:
                cache.age = TangentCache.MAX_AGE   # retry the iteration fresh
                continue
            break
    g = problem.grad_u(x[:N], x[N:], alpha, lmbda)
    res = float(np.max(np.abs(g)))
    return x[:N], x[N:], res, it, res <= tol


# ------------------------------ alpha-step ----------------------------------

def projected_gradient(g, a, lb, ub):
    """First-order optimality residual of min E s.t. lb <= a <= ub.

    Interior: g.  At the lower bound only: min(g, 0).  At the upper bound
    only: max(g, 0).  At BOTH (lb == ub, e.g. seeded-crack nodes pinned at 1
    by irreversibility): 0 -- a fixed variable cannot move, any gradient there
    is balanced by its bound multiplier.
    """
    pg = g.copy()
    at_lb = a <= lb + 1e-14
    at_ub = a >= ub - 1e-14
    pg[at_lb] = np.minimum(g[at_lb], 0.0)
    pg[at_ub] = np.maximum(g[at_ub], 0.0)
    pg[at_lb & at_ub] = 0.0
    return pg


def solve_alpha_qp(H, f, alpha0, lb, ub, tol):
    """Primal-dual active set (PDAS) on the strictly convex box QP.

    min 0.5 a'Ha + f'a  s.t.  lb <= a <= ub.  KKT: Ha + f = mu_lb - mu_ub with
    complementarity.  Each sweep guesses the active sets from the primal-dual
    residuals (scaling c = mean diagonal of H), pins actives to their bound,
    solves the free-set reduced system exactly, and re-derives the
    multipliers.  Terminates when the active sets repeat (exact solution of
    the QP restricted to that partition) or the true projected gradient of the
    clipped iterate meets ``tol``.  Falls back to scipy L-BFGS-B on a cycling
    guard.  Returns (alpha, proj_grad_inf, n_iter, ok).
    """
    a = np.clip(alpha0, lb, ub)
    g = H @ a + f
    pg_inf = float(np.max(np.abs(projected_gradient(g, a, lb, ub))))
    if pg_inf <= tol:
        return a, pg_inf, 0, True

    c = float(np.mean(H.diagonal()))
    mu_lb = np.where(a <= lb + 1e-14, np.maximum(g, 0.0), 0.0)
    mu_ub = np.where(a >= ub - 1e-14, np.maximum(-g, 0.0), 0.0)
    seen: set = set()
    it = 0
    for it in range(1, MAX_NEWTON_ALPHA + 1):
        act_lb = (mu_lb + c * (lb - a)) > 0.0
        act_ub = (mu_ub + c * (a - ub)) > 0.0
        act_ub &= ~act_lb
        key = (act_lb.tobytes(), act_ub.tobytes())
        if key in seen:
            break                      # cycling -> fall back below
        seen.add(key)
        free = ~(act_lb | act_ub)
        a_new = np.where(act_lb, lb, np.where(act_ub, ub, 0.0))
        if np.any(free):
            idx = np.where(free)[0]
            rhs = -(f + H @ (a_new * ~free))[idx]
            Hff = H[idx][:, idx].tocsr()
            a_new[idx] = _solve_refined(Hff, rhs)
        g_new = H @ a_new + f
        mu_lb = np.where(act_lb, np.maximum(g_new, 0.0), 0.0)
        mu_ub = np.where(act_ub, np.maximum(-g_new, 0.0), 0.0)
        a = a_new
        a_clip = np.clip(a, lb, ub)
        pg_inf = float(np.max(np.abs(projected_gradient(H @ a_clip + f, a_clip, lb, ub))))
        if pg_inf <= tol and np.array_equal(a, a_clip):
            return a_clip, pg_inf, it, True
    # cycling or iteration cap: polish with L-BFGS-B from the best clipped point
    from scipy.optimize import minimize

    a0 = np.clip(a, lb, ub)
    res = minimize(
        lambda av: (0.5 * float(av @ (H @ av)) + float(f @ av),
                    (H @ av + f)),
        a0,
        jac=True,
        method="L-BFGS-B",
        bounds=np.column_stack([lb, ub]),
        options={"maxiter": 2000, "maxfun": 4000, "ftol": 1e-18, "gtol": tol * 0.5},
    )
    a = np.clip(res.x, lb, ub)
    pg_inf = float(np.max(np.abs(projected_gradient(H @ a + f, a, lb, ub))))
    return a, pg_inf, it + int(res.nit), pg_inf <= tol


# ---------------------- Anderson-accelerated staggering ---------------------

class AndersonAccelerator:
    """Safeguarded Anderson mixing of the staggered alpha fixed-point map.

    The plain staggered sweep is the fixed-point map a -> T(a) =
    argmin_alpha E(u(a), alpha) (bounds included).  During unstable crack
    growth T advances the crack front by roughly one element ring per sweep,
    which is the observed 150-200+ sweep crawl.  Anderson(m) extrapolates the
    iterate from the last m (input, residual) pairs -- the standard
    acceleration for exactly this staggered crawl in phase-field fracture --
    and typically cuts the sweep count several-fold.

    Safeguards (the certificate never depends on them): the mixed iterate is
    clipped to [lb, ub]; the least-squares mixing is rejected in favor of the
    plain sweep when its coefficients are extreme (||theta||_1 > THETA_CAP) or
    ill-conditioned; the history restarts when the residual norm grew for two
    consecutive sweeps.  Convergence is still declared ONLY by the strict
    block-optimality test on the true fields, so acceleration can change how
    fast the fixed point is reached, never what is accepted as one.
    """

    THETA_CAP = 20.0

    def __init__(self, m: int = 5):
        self.m = int(m)
        self.a_hist: list[np.ndarray] = []
        self.r_hist: list[np.ndarray] = []
        self.prev_rnorm = np.inf
        self.grew = 0

    def reset(self):
        self.a_hist.clear()
        self.r_hist.clear()
        self.prev_rnorm = np.inf
        self.grew = 0

    def mix(self, a: np.ndarray, a_plain: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        r = a_plain - a
        rnorm = float(np.linalg.norm(r))
        if rnorm >= self.prev_rnorm:
            self.grew += 1
            if self.grew >= 2:
                self.reset()
        else:
            self.grew = 0
        self.prev_rnorm = rnorm

        self.a_hist.append(a.copy())
        self.r_hist.append(r)
        if len(self.a_hist) > self.m + 1:
            self.a_hist.pop(0)
            self.r_hist.pop(0)
        k = len(self.r_hist)
        if k < 2:
            return a_plain
        # minimize || r_k + sum_j gamma_j (r_j - r_k) || over the previous residuals
        R = np.stack([rj - self.r_hist[-1] for rj in self.r_hist[:-1]], axis=1)
        try:
            gamma, *_ = np.linalg.lstsq(R, -self.r_hist[-1], rcond=1e-10)
        except np.linalg.LinAlgError:
            return a_plain
        theta = np.zeros(k)
        theta[:-1] = gamma
        theta[-1] = 1.0 - float(np.sum(gamma))
        if not np.all(np.isfinite(theta)) or float(np.sum(np.abs(theta))) > self.THETA_CAP:
            return a_plain
        mixed = np.zeros_like(a)
        for w, aj, rj in zip(theta, self.a_hist, self.r_hist):
            mixed += w * (aj + rj)
        return np.clip(mixed, lb, ub)


# ---------------------------- one load step ---------------------------------

class AndersonAccelerator:
    """Guarded Anderson(m) mixing on the staggered alpha fixed-point map.

    The plain staggered sweep is the fixed-point iteration a_{k+1} = G(a_k):
    a u-solve at fixed alpha, then the exact alpha box-QP at that u.  During
    crack-front advance its convergence is linear with rate near one -- the
    front moves about one element per sweep -- which is the textbook regime
    for Anderson mixing.  This class stores the last M (iterate, residual)
    pairs, solves the small least-squares problem for the mixing weights, and
    proposes the mixed iterate clipped to the box constraints.

    The caller guards every proposal by energy descent and falls back to the
    plain iterate (clearing the history) when the guard rejects, so the
    acceleration can shorten the path but never redefine the target: the
    step's convergence test stays the true fixed-point optimality check at
    the fresh u.
    """

    M = 5

    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lb = lb
        self.ub = ub
        self.iterates: list[np.ndarray] = []
        self.residuals: list[np.ndarray] = []

    def reset(self) -> None:
        self.iterates.clear()
        self.residuals.clear()

    def propose(self, a: np.ndarray, a_new: np.ndarray):
        """Mixed candidate from the history including (a, a_new), or None."""
        r = a_new - a
        self.iterates.append(a.copy())
        self.residuals.append(r)
        if len(self.iterates) > self.M:
            self.iterates.pop(0)
            self.residuals.pop(0)
        m = len(self.iterates)
        if m < 2:
            return None
        R = np.stack(self.residuals, axis=1)          # (n, m)
        dR = R[:, :-1] - R[:, -1:]                    # (n, m-1)
        try:
            theta, *_ = np.linalg.lstsq(dR, -R[:, -1], rcond=None)
        except np.linalg.LinAlgError:
            return None
        w = np.empty(m)
        w[:-1] = theta
        w[-1] = 1.0 - float(theta.sum())
        G = np.stack(self.iterates, axis=1) + R       # map images g(a_j) = a_j + r_j
        cand = G @ w
        if not np.all(np.isfinite(cand)):
            return None
        return np.clip(cand, self.lb, self.ub)


def alpha_tolerance(problem: FemProblem) -> float:
    """Tolerance scale: the constant AT1 linear-term vector (per-node)."""
    f_at1 = np.zeros(problem.n_nodes)
    np.add.at(
        f_at1,
        problem.T.ravel(),
        np.repeat(problem.area * (3.0 / 8.0) * problem.Gc_mm / problem.l0 / 3.0, 3),
    )
    return max(RTOL_ALPHA * float(np.max(np.abs(f_at1))), ATOL_ALPHA)


def _stagger_at(
    problem: FemProblem,
    lmbda: float,
    u: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    alpha_prev: np.ndarray,
    a_tol: float,
    max_sweeps: int,
):
    """One staggered minimization at traction multiplier ``lmbda``.

    One tangent cache and one Anderson mixer live ACROSS the sweeps: during
    crack-front advance alpha moves by roughly one element per plain sweep,
    so a stale u-tangent stays an excellent preconditioner (refactoring
    every sweep dominated the stalled production runs' wall time), and the
    staggered map's near-unit linear rate is exactly the regime Anderson
    mixing accelerates.  Neither changes what "converged" means: the test
    below is still the true fixed point at the fresh u.

    Returns (u, v, a, converged, u_res, a_res, d_stag, n_sweeps, n_u, n_a).
    """
    u_tol = max(RTOL_U * lmbda * float(np.max(np.abs(problem._f_trac_unit))), ATOL_U)
    u_res = a_res = np.inf
    n_u_total = n_a_total = 0
    d_stag = 0.0
    converged = False
    n_sweeps = 0
    cache = TangentCache()
    mixer = AndersonAccelerator(lb, ub)
    for n_sweeps in range(1, max_sweeps + 1):
        u, v, u_res, n_u, ok_u = solve_u(problem, u, v, a, lmbda, u_tol, cache=cache)
        n_u_total += n_u
        H, f, _ = problem.alpha_quadratic(u, v, alpha_prev)
        g = H @ a + f
        a_res = float(np.max(np.abs(projected_gradient(g, a, lb, ub))))
        if ok_u and a_res <= a_tol:
            converged = True
            break
        a_new, a_res, n_a, _ok_a = solve_alpha_qp(H, f, a, lb, ub, a_tol)
        n_a_total += n_a
        d_stag = float(np.max(np.abs(a_new - a)))
        a_cand = mixer.propose(a, a_new)
        if a_cand is not None:
            # Energy guard on the alpha-dependent part of the step functional
            # at this u (the boundary terms carry no alpha): accept the mixed
            # iterate only when it descends below the plain sweep, else fall
            # back and clear the mixing history.
            e_plain = sum(problem.bulk_energies(u, v, a_new, alpha_prev, lb))
            e_mixed = sum(problem.bulk_energies(u, v, a_cand, alpha_prev, lb))
            if np.isfinite(e_mixed) and e_mixed < e_plain:
                a = a_cand
            else:
                a = a_new
                mixer.reset()
        else:
            a = a_new
    # The last alpha update can change the displacement residual. Report both
    # block residuals at the returned state, including iteration-limit exits.
    u_res = float(np.max(np.abs(problem.grad_u(u, v, a, lmbda))))
    H, f, _ = problem.alpha_quadratic(u, v, alpha_prev)
    a_res = float(np.max(np.abs(projected_gradient(H @ a + f, a, lb, ub))))
    converged = u_res <= u_tol and a_res <= a_tol
    return u, v, a, converged, u_res, a_res, d_stag, n_sweeps, n_u_total, n_a_total


N_CONTINUATION_STAGES = 4


def solve_load_step(
    problem: FemProblem,
    step_idx: int,
    lmbda: float,
    u: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    hist_alpha: np.ndarray,
    *,
    max_sweeps: int = MAX_STAGGER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StepRecord]:
    """Minimize the step functional; alpha_prev (viscous anchor) = incoming alpha."""
    t0 = time.time()
    alpha_prev = alpha.copy()
    lb = np.clip(hist_alpha, 0.0, 1.0)
    ub = np.ones_like(lb)
    a = np.clip(alpha, lb, ub)

    a_tol = alpha_tolerance(problem)

    u, v, a, converged, u_res, a_res, d_stag, n_sweeps, n_u_total, n_a_total = _stagger_at(
        problem, lmbda, u, v, a, lb, ub, alpha_prev, a_tol, max_sweeps
    )
    n_continuation = 0
    if not converged:
        # Stall retry: internal lambda-continuation.  A stalled step is one
        # whose crack-front advance is too large for the staggered map to
        # traverse in one minimization (observed: an active-set boundary
        # creep with stag_da plateauing ~1e-7).  Re-approach the SAME step
        # functional through sub-targets of the traction multiplier only --
        # alpha_prev and the viscous dt stay the step's own throughout, so
        # every intermediate stage is nothing but a warm start and the final
        # stage minimizes exactly the step functional at the full lmbda;
        # the convergence certificate is that final stage's fixed-point
        # test, unchanged.  The runner's uniform-from-zero schedule makes
        # the previous step's multiplier exactly lmbda * i / (i + 1).
        lam_prev = lmbda * step_idx / (step_idx + 1)
        for j in range(1, N_CONTINUATION_STAGES + 1):
            n_continuation = j
            lam_j = lam_prev + (lmbda - lam_prev) * j / N_CONTINUATION_STAGES
            u, v, a, converged, u_res, a_res, d_stag, ns_j, nu_j, na_j = _stagger_at(
                problem, lam_j, u, v, a, lb, ub, alpha_prev, a_tol, max_sweeps
            )
            n_sweeps += ns_j
            n_u_total += nu_j
            n_a_total += na_j
        # ``converged`` now reports the final stage at lam_j == lmbda.

    u_tol = max(RTOL_U * lmbda * float(np.max(np.abs(problem._f_trac_unit))), ATOL_U)

    E_el, E_d, E_hist, E_visco = problem.bulk_energies(u, v, a, alpha_prev, lb)
    E_trac, E_spr, E_ovl = problem.bc_energies(u, v, lmbda)
    rec = StepRecord(
        step=step_idx,
        lmbda=float(lmbda),
        converged=bool(converged),
        u_grad_inf=u_res,
        u_grad_tol=u_tol,
        alpha_pg_inf=a_res,
        alpha_pg_tol=a_tol,
        stag_dalpha_inf=d_stag,
        n_stagger=n_sweeps,
        n_newton_u=n_u_total,
        n_newton_alpha=n_a_total,
        E_total=E_el + E_d + E_hist + E_visco + E_trac + E_spr + E_ovl,
        E_el=E_el,
        E_d=E_d,
        E_hist=E_hist,
        E_visco=E_visco,
        E_trac=E_trac,
        E_spr=E_spr,
        E_ovl=E_ovl,
        alpha_min=float(a.min()),
        alpha_max=float(a.max()),
        irrev_violation=float(np.max(np.maximum(lb - a, 0.0))),
        time_sec=time.time() - t0,
        n_continuation=n_continuation,
    )
    return u, v, a, rec


# --------------------------- gradient self-check ----------------------------

def gradient_fd_check(problem: FemProblem, u, v, alpha, alpha_prev, hist, lmbda,
                      n_probe: int = 8, seed: int = 0) -> dict:
    """Central-difference check of the analytic gradients (assembly self-check).

    Probes at a deterministically PERTURBED copy of the given state: at a
    converged state the true gradient is ~0 and a finite-difference quotient
    would only compare round-off to round-off, so the probe state adds a
    seeded displacement/alpha perturbation that engages the split
    nonlinearity, the bedding springs and the overlay operators of the actual
    problem.  Guards the independent assembly against sign/scatter bugs;
    reported in the smoke report and never skipped.
    """
    rng = np.random.default_rng(seed)
    N = problem.n_nodes
    out = {}

    u = u + 0.01 * rng.standard_normal(N)
    v = v + 0.01 * rng.standard_normal(N)
    alpha = np.clip(alpha + 0.2 * rng.random(N), 0.0, 1.0)

    x = np.concatenate([u, v])
    g = problem.grad_u(u, v, alpha, lmbda)

    def e_u(xv):
        E_el = problem.bulk_energies(xv[:N], xv[N:], alpha, alpha_prev, hist)[0]
        return E_el + sum(problem.bc_energies(xv[:N], xv[N:], lmbda))

    h = 1e-6 * max(1.0, float(np.max(np.abs(x))))
    g_inf = float(np.max(np.abs(g)))
    errs = []
    for i in rng.choice(2 * N, size=n_probe, replace=False):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd = (e_u(xp) - e_u(xm)) / (2 * h)
        scale = max(abs(fd), abs(g[i]), 1e-3 * g_inf, 1e-12)
        errs.append(abs(fd - g[i]) / scale)
    out["u_grad_max_rel_err"] = float(np.max(errs))

    ga = problem.grad_alpha(u, v, alpha, alpha_prev)

    def e_a(av):
        E_el, E_d, _, E_visco = problem.bulk_energies(u, v, av, alpha_prev, hist)
        return E_el + E_d + E_visco

    # central-difference step sized for the perturbed-state energy magnitude
    # (round-off noise ~ eps*|E|/(2*ha); at ha = 1e-5 it sits orders below the
    # gradient scale while truncation stays negligible for this C^2 term)
    ha = 1e-5
    ga_inf = float(np.max(np.abs(ga)))
    errs_a = []
    for i in rng.choice(N, size=n_probe, replace=False):
        ap = alpha.copy(); ap[i] += ha
        am = alpha.copy(); am[i] -= ha
        fd = (e_a(ap) - e_a(am)) / (2 * ha)
        scale = max(abs(fd), abs(ga[i]), 1e-3 * ga_inf, 1e-12)
        errs_a.append(abs(fd - ga[i]) / scale)
    out["alpha_grad_max_rel_err"] = float(np.max(errs_a))
    return out
