"""Discrete energy assembly for the FEM reference solver (independent numpy).

The functional reproduced here is, term by term, the one the Deep Ritz lining
solver minimizes:

  * elastic energy with the Miehe spectral tension/compression split and
    degradation g(alpha) = (1 - alpha)^2 on the tensile part
    (drm_02.core.compute_energy.strain_energy_with_split, se_split='spectral',
    including its sqrt(disc + 1e-14) eigenvalue regularization);
  * AT1 crack-surface density Gc * 3/(8 l0) * (alpha + l0^2 |grad alpha|^2)
    (compute_energy uses w1/c_w * (alpha/l0 + l0 |grad alpha|^2), c_w = 8/3);
  * backward-Euler incremental dissipation
    0.5 * (eta/1e6) * (alpha-alpha_prev)^2 / dt
    (eta in Pa*s converted to MPa*s);
  * outer-boundary traction work -lambda * sum (t . u) ds
    (LiningFieldComputation.compute_pressure_energy, traction mode);
  * Winkler bedding boundary energy 0.5 * sum kn*(u.n)^2 ds + 0.5 * sum kt*(u.t)^2 ds
    (compute_spring_energy, ug = 0);
  * smeared composite-overlay boundary stiffness for a reinforced cell
    (DoubleLayerLiningFieldComputation.compute_composite_energy: roll-based
    first/second differences on the ordered outer boundary, with the same
    clamp(min=1e-9) denominators).

Discretization is IDENTICAL to the Deep Ritz element pass: linear (CST)
triangles, per-element constant strains from the standard B-matrix with the
positive element area (compute_energy.field_grads convention), and the
per-element phase field evaluated as the plain average of the three nodal
alphas (one-point rule).  All arithmetic is float64.

Units: coordinates mm, stresses/energy densities MPa (= N/mm^2), Gc in N/mm,
energies in N*mm per unit tunnel-axis length (i.e. per mm of axis).

Irreversibility and bounds are NOT penalty terms here: the solver enforces
alpha in [max(hist_alpha, 0), 1] as hard bound constraints of the minimization,
so the Deep Ritz irreversibility penalty term evaluates to exactly zero on any
feasible field.  ``hist_penalty_energy`` is still provided so the smoke report
can evaluate that term on the converged fields and show it vanishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp


SQRT_DISC_REG = 1e-14   # matches strain_energy_with_split: sqrt(disc + 1e-14)
PA_PER_MPA = 1.0e6      # matches PFFModel.viscous_term eta unit conversion
DS_CLAMP = 1e-9         # matches DoubleLayerLiningFieldComputation clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------

@dataclass
class OverlaySpec:
    """Composite-overlay boundary term data (already coverage-smeared).

    ``EA_N`` / ``kGA_N`` / ``EI_N_mm2`` are the values
    DoubleLayerLiningFieldComputation stores in ``_beam_EA`` / ``_beam_kGA`` /
    ``_beam_EI`` (params.EA_N etc., times the extra coverage argument which the
    production path leaves at 1.0).  Frames are the overlay's own
    normals/tangents/ds/curvature arrays.

    The attribute names are inherited from the member-parameter interface.
    After division by the longitudinal pitch, EA and kGA have units N/mm
    and EI has units N mm; the resulting energy is per mm of tunnel axis.
    """

    normals: np.ndarray      # (Mb, 2)
    tangents: np.ndarray     # (Mb, 2)
    ds: np.ndarray           # (Mb,)
    curvature: np.ndarray    # (Mb,)
    EA_N: float
    kGA_N: float
    EI_N_mm2: float
    layer_count: int


@dataclass
class FemProblem:
    """Everything the solver needs, in plain float64 numpy arrays."""

    # mesh
    X: np.ndarray            # (N,) mm
    Y: np.ndarray            # (N,) mm
    T: np.ndarray            # (M, 3) int
    area: np.ndarray         # (M,) mm^2, positive
    # material
    E: float
    nu: float
    Gc_mm: float             # N/mm  (= w1)
    l0: float                # mm
    tol_ir: float            # only for evaluating the (vanishing) hist penalty
    viscous_eta: float       # Pa*s
    viscous_dt: float        # s
    # outer boundary (ordered CCW, as load_lining_mesh returns it)
    bnd_idx: np.ndarray      # (Mb,) node indices
    bnd_normals: np.ndarray  # (Mb, 2) from lining_bc.compute_boundary_geometry
    bnd_tangents: np.ndarray
    bnd_ds: np.ndarray       # (Mb,)
    traction: np.ndarray     # (Mb, 2) MPa (at lambda = 1)
    kn: np.ndarray           # (Mb,) MPa/mm (crown-zero + bedding degradation applied)
    kt: np.ndarray           # (Mb,) MPa/mm
    overlay: Optional[OverlaySpec] = None
    residual_stiffness: float = 0.0

    # derived (filled by prepare())
    lam_lame: float = 0.0
    mu_lame: float = 0.0
    b: np.ndarray = field(default=None, repr=False)   # (M,3) dN/dx
    c: np.ndarray = field(default=None, repr=False)   # (M,3) dN/dy
    _dof_e: np.ndarray = field(default=None, repr=False)
    _Krow: np.ndarray = field(default=None, repr=False)
    _Kcol: np.ndarray = field(default=None, repr=False)
    _Hrow: np.ndarray = field(default=None, repr=False)
    _Hcol: np.ndarray = field(default=None, repr=False)
    _K_bc: sp.csr_matrix = field(default=None, repr=False)
    _f_trac_unit: np.ndarray = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_elem(self) -> int:
        return int(self.T.shape[0])

    def prepare(self) -> None:
        """Precompute B-matrices, scatter indices and the constant BC operators."""
        E, nu = float(self.E), float(self.nu)
        self.lam_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu_lame = E / (2.0 * (1.0 + nu))

        X, Y, T = self.X, self.Y, self.T
        x0, y0 = X[T[:, 0]], Y[T[:, 0]]
        x1, y1 = X[T[:, 1]], Y[T[:, 1]]
        x2, y2 = X[T[:, 2]], Y[T[:, 2]]
        signed2A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        n_cw = int(np.sum(signed2A <= 0.0))
        if n_cw:
            # compute_energy.field_grads divides the CCW B-formula by the
            # POSITIVE area; a clockwise triangle would silently flip its
            # strain sign there.  The gmsh meshes used here are CCW; refuse
            # anything else rather than reproduce an orientation bug.
            raise ValueError(f"{n_cw} clockwise triangles in mesh; expected all CCW")
        two_area = np.abs(signed2A)          # equals signed2A (all CCW)
        self.area = 0.5 * two_area
        # field_grads convention: grad_x = sum_i b_i f_i with
        # b = [y1-y2, y2-y0, y0-y1] / (2A), c = [x2-x1, x0-x2, x1-x0] / (2A)
        self.b = np.stack([(y1 - y2), (y2 - y0), (y0 - y1)], axis=1) / two_area[:, None]
        self.c = np.stack([(x2 - x1), (x0 - x2), (x1 - x0)], axis=1) / two_area[:, None]

        N = self.n_nodes
        # element displacement dof order [u0,u1,u2,v0,v1,v2]; global u_i -> i, v_i -> N+i
        self._dof_e = np.concatenate([T, T + N], axis=1).astype(np.int64)   # (M,6)
        self._Krow = np.repeat(self._dof_e, 6, axis=1).ravel()
        self._Kcol = np.tile(self._dof_e, (1, 6)).ravel()
        Ta = T.astype(np.int64)
        self._Hrow = np.repeat(Ta, 3, axis=1).ravel()
        self._Hcol = np.tile(Ta, (1, 3)).ravel()

        self._K_bc = self._assemble_bc_stiffness()
        self._f_trac_unit = self._assemble_traction_load_unit()

    # ------------------------------------------------------------------
    # strain / spectral-split kernels
    # ------------------------------------------------------------------
    def strains(self, u: np.ndarray, v: np.ndarray):
        """Per-element (exx, eyy, exy) with exy the tensor shear strain."""
        T = self.T
        ue = u[T]
        ve = v[T]
        exx = np.einsum("ei,ei->e", self.b, ue)
        eyy = np.einsum("ei,ei->e", self.c, ve)
        exy = 0.5 * (np.einsum("ei,ei->e", self.c, ue) + np.einsum("ei,ei->e", self.b, ve))
        return exx, eyy, exy

    def _split_state(self, exx, eyy, exy):
        tr = exx + eyy
        hd = 0.5 * (exx - eyy)
        D = np.sqrt(hd * hd + exy * exy + SQRT_DISC_REG)
        eps1 = 0.5 * tr + D
        eps2 = 0.5 * tr - D
        return tr, hd, D, eps1, eps2

    def psi_split(self, exx, eyy, exy):
        """(psi_plus, psi_minus) per element, Miehe spectral split."""
        lam, mu = self.lam_lame, self.mu_lame
        tr, _hd, _D, eps1, eps2 = self._split_state(exx, eyy, exy)
        tp = np.maximum(tr, 0.0)
        tn = np.maximum(-tr, 0.0)
        e1p = np.maximum(eps1, 0.0)
        e1n = np.maximum(-eps1, 0.0)
        e2p = np.maximum(eps2, 0.0)
        e2n = np.maximum(-eps2, 0.0)
        psi_p = 0.5 * lam * tp * tp + mu * (e1p * e1p + e2p * e2p)
        psi_n = 0.5 * lam * tn * tn + mu * (e1n * e1n + e2n * e2n)
        return psi_p, psi_n

    def _dpsi_dstrain(self, exx, eyy, exy, g_elem):
        """d(g*psi_p + psi_n)/d(exx, eyy, exy) per element."""
        lam, mu = self.lam_lame, self.mu_lame
        tr, hd, D, eps1, eps2 = self._split_state(exx, eyy, exy)
        tp = np.maximum(tr, 0.0)
        tn = np.maximum(-tr, 0.0)
        s1p = np.maximum(eps1, 0.0)
        s1n = np.maximum(-eps1, 0.0)
        s2p = np.maximum(eps2, 0.0)
        s2n = np.maximum(-eps2, 0.0)
        dpsi_dt = lam * (g_elem * tp - tn)
        dpsi_de1 = 2.0 * mu * (g_elem * s1p - s1n)
        dpsi_de2 = 2.0 * mu * (g_elem * s2p - s2n)
        a = hd / (2.0 * D)                 # d eps1/d exx = 0.5 + a ; d eps1/d eyy = 0.5 - a
        w = exy / D                        # d eps1/d exy = w ; d eps2/d exy = -w
        dpsi_dexx = dpsi_dt + dpsi_de1 * (0.5 + a) + dpsi_de2 * (0.5 - a)
        dpsi_deyy = dpsi_dt + dpsi_de1 * (0.5 - a) + dpsi_de2 * (0.5 + a)
        dpsi_dexy = (dpsi_de1 - dpsi_de2) * w
        return dpsi_dexx, dpsi_deyy, dpsi_dexy

    # ------------------------------------------------------------------
    # element-average alpha helpers
    # ------------------------------------------------------------------
    def elem_mean(self, nodal: np.ndarray) -> np.ndarray:
        T = self.T
        return (nodal[T[:, 0]] + nodal[T[:, 1]] + nodal[T[:, 2]]) / 3.0

    def alpha_grad(self, alpha: np.ndarray):
        ae = alpha[self.T]
        gax = np.einsum("ei,ei->e", self.b, ae)
        gay = np.einsum("ei,ei->e", self.c, ae)
        return gax, gay

    # ------------------------------------------------------------------
    # energies
    # ------------------------------------------------------------------
    def degradation(self, alpha):
        return self.residual_stiffness + (1.0-self.residual_stiffness)*(1.0-alpha)**2

    def bulk_energies(self, u, v, alpha, alpha_prev, hist_alpha):
        """(E_el, E_d, E_hist_penalty, E_visco) exactly as compute_energy sums them."""
        exx, eyy, exy = self.strains(u, v)
        a_bar = self.elem_mean(alpha)
        g = self.degradation(a_bar)
        psi_p, psi_n = self.psi_split(exx, eyy, exy)
        E_el = float(np.sum(self.area * (g * psi_p + psi_n)))

        gax, gay = self.alpha_grad(alpha)
        E_d = float(np.sum(
            self.area * (3.0 / 8.0) * self.Gc_mm
            * (a_bar / self.l0 + self.l0 * (gax * gax + gay * gay))
        ))

        # Deep Ritz irreversibility penalty, evaluated (vanishes on feasible fields)
        d_bar = self.elem_mean(alpha - hist_alpha)
        wp = 27.0 / 64.0 / (self.tol_ir ** 2)
        E_hist = float(np.sum(0.5 * self.Gc_mm * wp * np.maximum(-d_bar, 0.0) ** 2 * self.area))

        if self.viscous_eta > 0.0 and self.viscous_dt is not None:
            eta_mpa = self.viscous_eta / PA_PER_MPA
            dp_bar = self.elem_mean(alpha - alpha_prev)
            E_visco = float(np.sum(self.area * 0.5 * eta_mpa * dp_bar ** 2 / self.viscous_dt))
        else:
            E_visco = 0.0
        return E_el, E_d, E_hist, E_visco

    def bc_energies(self, u, v, lmbda: float):
        """(E_traction, E_spring, E_overlay) matching lining_bc term by term."""
        bi = self.bnd_idx
        ub = u[bi]
        vb = v[bi]
        tx = lmbda * self.traction[:, 0]
        ty = lmbda * self.traction[:, 1]
        E_trac = -float(np.sum((tx * ub + ty * vb) * self.bnd_ds))

        nx, ny = self.bnd_normals[:, 0], self.bnd_normals[:, 1]
        txv, tyv = self.bnd_tangents[:, 0], self.bnd_tangents[:, 1]
        un = ub * nx + vb * ny
        ut = ub * txv + vb * tyv
        E_spr = 0.5 * float(np.sum(self.kn * un * un * self.bnd_ds)) \
            + 0.5 * float(np.sum(self.kt * ut * ut * self.bnd_ds))

        E_ovl = 0.0
        if self.overlay is not None:
            axial, shear, bend = self._overlay_strains(ub, vb)
            o = self.overlay
            density = o.EA_N * axial ** 2 + o.kGA_N * shear ** 2 + o.EI_N_mm2 * bend ** 2
            E_ovl = 0.5 * float(o.layer_count) * float(np.sum(density * o.ds))
        return E_trac, E_spr, E_ovl

    def total_energy(self, u, v, alpha, alpha_prev, hist_alpha, lmbda: float) -> float:
        E_el, E_d, E_hist, E_visco = self.bulk_energies(u, v, alpha, alpha_prev, hist_alpha)
        E_trac, E_spr, E_ovl = self.bc_energies(u, v, lmbda)
        return E_el + E_d + E_hist + E_visco + E_trac + E_spr + E_ovl

    # ------------------------------------------------------------------
    # overlay kinematics (mirrors DoubleLayerLiningFieldComputation)
    # ------------------------------------------------------------------
    def _overlay_strains(self, ub, vb):
        o = self.overlay
        un = ub * o.normals[:, 0] + vb * o.normals[:, 1]
        ut = ub * o.tangents[:, 0] + vb * o.tangents[:, 1]
        den1 = np.maximum(2.0 * o.ds, DS_CLAMP)
        den2 = np.maximum(o.ds, DS_CLAMP) ** 2
        d1 = lambda f: (np.roll(f, -1) - np.roll(f, 1)) / den1
        d2 = lambda f: (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / den2
        axial = d1(ut) + o.curvature * un
        shear = d1(un) - o.curvature * ut
        bend = d2(un)
        return axial, shear, bend

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------
    def grad_u(self, u, v, alpha, lmbda: float) -> np.ndarray:
        """d(E_el + E_bc)/d[u; v]  (the other terms do not depend on u)."""
        N = self.n_nodes
        exx, eyy, exy = self.strains(u, v)
        a_bar = self.elem_mean(alpha)
        g = self.degradation(a_bar)
        dxx, dyy, dxy = self._dpsi_dstrain(exx, eyy, exy, g)
        A = self.area
        # dexx/du_i = b_i ; dexy/du_i = 0.5 c_i ; deyy/dv_i = c_i ; dexy/dv_i = 0.5 b_i
        gu_e = A[:, None] * (dxx[:, None] * self.b + 0.5 * dxy[:, None] * self.c)
        gv_e = A[:, None] * (dyy[:, None] * self.c + 0.5 * dxy[:, None] * self.b)
        grad = np.zeros(2 * N)
        np.add.at(grad, self.T.ravel(), gu_e.ravel())
        np.add.at(grad, (self.T + N).ravel(), gv_e.ravel())

        # boundary: quadratic part K_bc @ x, linear traction part -lambda * f_unit
        x = np.concatenate([u, v])
        grad += self._K_bc @ x
        grad -= lmbda * self._f_trac_unit
        return grad

    def grad_alpha(self, u, v, alpha, alpha_prev) -> np.ndarray:
        """d(E_el + E_d + E_visco)/d alpha (nodal)."""
        exx, eyy, exy = self.strains(u, v)
        psi_p, _ = self.psi_split(exx, eyy, exy)
        a_bar = self.elem_mean(alpha)
        A = self.area
        third = 1.0 / 3.0

        coef = A * (2.0 * (1.0-self.residual_stiffness) * (a_bar - 1.0) * psi_p) * third
        coef = coef + A * (3.0 / 8.0) * self.Gc_mm / self.l0 * third
        if self.viscous_eta > 0.0 and self.viscous_dt is not None:
            eta_mpa = self.viscous_eta / PA_PER_MPA
            dp_bar = self.elem_mean(alpha - alpha_prev)
            coef = coef + A * eta_mpa * dp_bar / self.viscous_dt * third

        gax, gay = self.alpha_grad(alpha)
        w = A * (3.0 / 8.0) * self.Gc_mm * self.l0 * 2.0
        ge = coef[:, None] + w[:, None] * (gax[:, None] * self.b + gay[:, None] * self.c)
        grad = np.zeros(self.n_nodes)
        np.add.at(grad, self.T.ravel(), ge.ravel())
        return grad

    # ------------------------------------------------------------------
    # tangents / quadratic forms
    # ------------------------------------------------------------------
    def elastic_tangent(self, u, v, alpha) -> sp.csr_matrix:
        """Consistent SPD tangent of the fixed-alpha split elastic energy.

        For fixed g in [0, 1] the Miehe spectral-split density
        psi = 0.5*lam*(g*<tr>+^2 + <tr>-^2) + mu*sum_i (g*<eps_i>+^2 + <eps_i>-^2)
        is CONVEX in the strain: both eigenvalue Macaulay terms are convex, and
        its exact Hessian decomposes into the projection part
        2*mu*(w1 p1 p1' + w2 p2 p2') (Heaviside weights w_i = g or 1) plus the
        eigenvector-rotation part 2*mu*(phi'(eps1)/2 - phi'(eps2)/2) * Hess(D),
        where D = sqrt(hd^2 + exy^2 + reg) is convex (Hess(D) is PSD) and
        phi'(eps1) >= phi'(eps2) because phi' is nondecreasing and
        eps1 >= eps2.  The assembled matrix is therefore PSD and the u-step
        Newton direction is a true (regularized) Newton direction.
        """
        lam, mu = self.lam_lame, self.mu_lame
        exx, eyy, exy = self.strains(u, v)
        tr, hd, D, eps1, eps2 = self._split_state(exx, eyy, exy)
        a_bar = self.elem_mean(alpha)
        g = self.degradation(a_bar)

        wt = np.where(tr > 0.0, g, 1.0)
        w1 = np.where(eps1 > 0.0, g, 1.0)
        w2 = np.where(eps2 > 0.0, g, 1.0)
        # phi'(e)/2 = g*<e>+ - <e>-  (nondecreasing in e)
        w1p = g * np.maximum(eps1, 0.0) - np.maximum(-eps1, 0.0)
        w2p = g * np.maximum(eps2, 0.0) - np.maximum(-eps2, 0.0)
        c2 = hd / D
        s2 = exy / D
        M = self.n_elem
        # projection rows in Voigt (exx, eyy, gxy)
        p1 = np.stack([0.5 * (1.0 + c2), 0.5 * (1.0 - c2), 0.5 * s2], axis=1)
        p2 = np.stack([0.5 * (1.0 - c2), 0.5 * (1.0 + c2), -0.5 * s2], axis=1)
        J = np.zeros((M, 3, 3))
        J[:, 0, 0] = J[:, 0, 1] = J[:, 1, 0] = J[:, 1, 1] = 1.0
        C = (lam * wt)[:, None, None] * J \
            + 2.0 * mu * (w1[:, None, None] * np.einsum("ei,ej->eij", p1, p1)
                          + w2[:, None, None] * np.einsum("ei,ej->eij", p2, p2))
        # rotation curvature: Hess(D) = (qq' + ss')/D - (grad D)(grad D)'/D
        # with q = d hd/d eps_v = (0.5, -0.5, 0), s = d exy/d eps_v = (0, 0, 0.5)
        gradD = np.stack([0.5 * c2, -0.5 * c2, 0.5 * s2], axis=1)   # (hd q + exy s)/D
        QS = np.zeros((3, 3))
        QS[0, 0] = QS[1, 1] = 0.25
        QS[0, 1] = QS[1, 0] = -0.25
        QS[2, 2] = 0.25
        HessD = (QS[None, :, :] - np.einsum("ei,ej->eij", gradD, gradD)) / D[:, None, None]
        C = C + 2.0 * mu * (w1p - w2p)[:, None, None] * HessD

        Bv = np.zeros((M, 3, 6))
        Bv[:, 0, 0:3] = self.b
        Bv[:, 1, 3:6] = self.c
        Bv[:, 2, 0:3] = self.c
        Bv[:, 2, 3:6] = self.b
        tmp = np.einsum("eij,ejk->eik", C, Bv)
        Ke = np.einsum("eji,ejk->eik", Bv, tmp) * self.area[:, None, None]
        n2 = 2 * self.n_nodes
        K = sp.coo_matrix((Ke.ravel(), (self._Krow, self._Kcol)), shape=(n2, n2))
        return K.tocsr()

    def _assemble_bc_stiffness(self) -> sp.csr_matrix:
        """Constant quadratic boundary operator: springs + composite overlay."""
        N = self.n_nodes
        n2 = 2 * N
        bi = self.bnd_idx
        Mb = bi.shape[0]
        nx, ny = self.bnd_normals[:, 0], self.bnd_normals[:, 1]
        tx, ty = self.bnd_tangents[:, 0], self.bnd_tangents[:, 1]
        w_n = self.kn * self.bnd_ds
        w_t = self.kt * self.bnd_ds
        rows, cols, vals = [], [], []
        for wi, dx, dy in ((w_n, nx, ny), (w_t, tx, ty)):
            for (ra, da) in ((bi, dx), (bi + N, dy)):
                for (rb, db) in ((bi, dx), (bi + N, dy)):
                    rows.append(ra)
                    cols.append(rb)
                    vals.append(wi * da * db)
        K = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n2, n2),
        ).tocsr()

        if self.overlay is not None:
            o = self.overlay
            den1 = np.maximum(2.0 * o.ds, DS_CLAMP)
            den2 = np.maximum(o.ds, DS_CLAMP) ** 2
            idx = np.arange(Mb)
            nxt = (idx + 1) % Mb
            prv = (idx - 1) % Mb
            D1 = sp.coo_matrix(
                (np.concatenate([1.0 / den1, -1.0 / den1]),
                 (np.concatenate([idx, idx]), np.concatenate([nxt, prv]))),
                shape=(Mb, Mb),
            ).tocsr()
            D2 = sp.coo_matrix(
                (np.concatenate([1.0 / den2, -2.0 / den2, 1.0 / den2]),
                 (np.concatenate([idx, idx, idx]), np.concatenate([nxt, idx, prv]))),
                shape=(Mb, Mb),
            ).tocsr()
            # selection of boundary normal / tangential displacement (in overlay frame)
            Nsel = sp.coo_matrix(
                (np.concatenate([o.normals[:, 0], o.normals[:, 1]]),
                 (np.concatenate([idx, idx]), np.concatenate([bi, bi + N]))),
                shape=(Mb, n2),
            ).tocsr()
            Tsel = sp.coo_matrix(
                (np.concatenate([o.tangents[:, 0], o.tangents[:, 1]]),
                 (np.concatenate([idx, idx]), np.concatenate([bi, bi + N]))),
                shape=(Mb, n2),
            ).tocsr()
            Kdiag = sp.diags(o.curvature)
            A_op = D1 @ Tsel + Kdiag @ Nsel
            S_op = D1 @ Nsel - Kdiag @ Tsel
            B_op = D2 @ Nsel
            L = float(o.layer_count)
            K = K + L * (
                A_op.T @ sp.diags(o.EA_N * o.ds) @ A_op
                + S_op.T @ sp.diags(o.kGA_N * o.ds) @ S_op
                + B_op.T @ sp.diags(o.EI_N_mm2 * o.ds) @ B_op
            )
        return K.tocsr()

    def _assemble_traction_load_unit(self) -> np.ndarray:
        """f such that E_traction = -lambda * f . [u; v]."""
        N = self.n_nodes
        f = np.zeros(2 * N)
        np.add.at(f, self.bnd_idx, self.traction[:, 0] * self.bnd_ds)
        np.add.at(f, self.bnd_idx + N, self.traction[:, 1] * self.bnd_ds)
        return f

    def alpha_quadratic(self, u, v, alpha_prev):
        """(H, f, const) of the alpha subproblem at fixed displacement.

        E_alpha(a) = 0.5 a^T H a + f^T a + const, assembled from
        g(a_bar)*psi_p + AT1 surface density + viscous increment; convex
        (H is PSD: psi_p >= 0, gradient term PSD, viscous coefficient >= 0).
        """
        exx, eyy, exy = self.strains(u, v)
        psi_p, _ = self.psi_split(exx, eyy, exy)
        A = self.area
        m = np.full(3, 1.0 / 3.0)
        mm = np.outer(m, m)

        quad_scalar = 2.0 * A * psi_p * (1.0-self.residual_stiffness)
        lin_scalar = -quad_scalar
        const = float(np.sum(A * psi_p))
        lin_scalar = lin_scalar + A * (3.0 / 8.0) * self.Gc_mm / self.l0
        if self.viscous_eta > 0.0 and self.viscous_dt is not None:
            eta_mpa = self.viscous_eta / PA_PER_MPA
            cvis = A * eta_mpa / self.viscous_dt
            ap_bar = self.elem_mean(alpha_prev)
            quad_scalar = quad_scalar + cvis
            lin_scalar = lin_scalar - cvis * ap_bar
            const += float(np.sum(0.5 * cvis * ap_bar ** 2))

        He = quad_scalar[:, None, None] * mm[None, :, :]
        wgrad = A * (3.0 / 8.0) * self.Gc_mm * self.l0 * 2.0
        He = He + wgrad[:, None, None] * (
            np.einsum("ei,ej->eij", self.b, self.b) + np.einsum("ei,ej->eij", self.c, self.c)
        )
        n = self.n_nodes
        H = sp.coo_matrix((He.ravel(), (self._Hrow, self._Hcol)), shape=(n, n)).tocsr()
        fvec = np.zeros(n)
        np.add.at(fvec, self.T.ravel(), np.repeat(lin_scalar / 3.0, 3))
        return H, fvec, const
