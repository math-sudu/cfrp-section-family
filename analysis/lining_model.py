"""Bond a planar Timoshenko frame to the continuum lining boundary.

Translations coincide with the concrete nodes; section rotations are independent
unknowns. For the shared staggered solver, the displacement vector is packed as
[u, v, theta], with theta appended to its second block. Material and phase-field
assembly are inherited from the shared finite-element implementation.
"""

import numpy as np
import scipy.sparse as sp

from analysis.paths import ensure_local_importable
ensure_local_importable()
from fem_check.fem_core import FemProblem


def frame_element(length, EA, kGA, EI):
    """Exact prismatic Timoshenko stiffness, [u1,v1,theta1,u2,v2,theta2]."""
    phi = 12 * EI / (kGA * length**2)
    a = EA / length
    b = 12 * EI / ((1 + phi) * length**3)
    c = 6 * EI / ((1 + phi) * length**2)
    d = (4 + phi) * EI / ((1 + phi) * length)
    e = (2 - phi) * EI / ((1 + phi) * length)
    return np.array([[a, 0, 0, -a, 0, 0],
                     [0, b, c, 0, -b, c],
                     [0, c, d, 0, -c, e],
                     [-a, 0, 0, a, 0, 0],
                     [0, -b, -c, 0, b, -c],
                     [0, c, e, 0, -c, d]])


class FrameLiningProblem(FemProblem):
    @classmethod
    def from_problem(cls, original):
        fields = {name: getattr(original, name) for name in cls.__dataclass_fields__}
        fields["residual_stiffness"] = 1e-6
        problem = cls(**fields)
        problem.prepare()
        return problem

    @property
    def n_rotations(self):
        return len(self.bnd_idx) if self.overlay else 0

    def prepare(self):
        overlay = self.overlay
        self.overlay = None
        super().prepare()
        self.overlay = overlay
        self._K_translation = self._K_bc
        self._f_translation = self._f_trac_unit
        m = self.n_rotations
        self._K_frame = sp.csr_matrix((2*self.n_nodes+m, 2*self.n_nodes+m))
        if m:
            coords = np.column_stack([self.X, self.Y])[self.bnd_idx]
            blocks, transforms, dofs, lengths = [], [], [], []
            for i in range(m):
                j = (i+1) % m
                dx = coords[j] - coords[i]
                length = np.linalg.norm(dx)
                lengths.append(length)
                c, s = dx / length
                rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
                transform = np.zeros((6, 6))
                transform[:3, :3] = transform[3:, 3:] = rotation
                k = frame_element(length, overlay.EA_N, overlay.kGA_N, overlay.EI_N_mm2)
                blocks.append(k * overlay.layer_count)
                transforms.append(transform)
                ni, nj, n = self.bnd_idx[i], self.bnd_idx[j], self.n_nodes
                dofs.append([ni, n+ni, 2*n+i, nj, n+nj, 2*n+j])
            self._frame_local = np.array(blocks)
            self._frame_transform = np.array(transforms)
            self._frame_dofs = np.array(dofs)
            self._frame_length = np.array(lengths)
            global_blocks = np.einsum('eji,ejk,ekl->eil', self._frame_transform,
                                      self._frame_local, self._frame_transform)
            self._K_frame = sp.coo_matrix(
                (global_blocks.ravel(), (np.repeat(self._frame_dofs, 6, axis=1).ravel(),
                                         np.tile(self._frame_dofs, (1, 6)).ravel())),
                shape=self._K_frame.shape).tocsr()
        self._K_bc = sp.block_diag((self._K_translation, sp.csr_matrix((m, m)))) + self._K_frame
        self._f_trac_unit = np.r_[self._f_translation, np.zeros(m)]

    def elastic_tangent(self, u, v, alpha):
        return sp.block_diag((super().elastic_tangent(u, v, alpha),
                              sp.csr_matrix((self.n_rotations, self.n_rotations))), format='csr')

    def grad_u(self, u, v, alpha, lmbda):
        n = self.n_nodes
        g = self.degradation(self.elem_mean(alpha))
        dxx, dyy, dxy = self._dpsi_dstrain(*self.strains(u, v), g)
        gu = self.area[:, None] * (dxx[:, None]*self.b + 0.5*dxy[:, None]*self.c)
        gv = self.area[:, None] * (dyy[:, None]*self.c + 0.5*dxy[:, None]*self.b)
        grad = np.zeros(2*n+self.n_rotations)
        np.add.at(grad, self.T.ravel(), gu.ravel())
        np.add.at(grad, (self.T+n).ravel(), gv.ravel())
        x = np.r_[u, v]
        grad[:2*n] += self._K_translation @ x[:2*n]
        if self.overlay:
            forces = np.einsum('eji,ej->ei', self._frame_transform, self.frame_forces(u, v))
            np.add.at(grad, self._frame_dofs.ravel(), forces.ravel())
        return grad - lmbda*self._f_trac_unit

    def bc_energies(self, u, v, lmbda):
        x = np.r_[u, v]
        translations = x[:2*self.n_nodes]
        return (-lmbda * float(x @ self._f_trac_unit),
                0.5 * float(translations @ (self._K_translation @ translations)),
                self.frame_energy(u, v) if self.overlay else 0.0)

    def _frame_deformations(self, u, v):
        dofs = np.r_[u, v][self._frame_dofs]
        local = np.einsum('eij,ej->ei', self._frame_transform, dofs)
        extension = local[:, 3] - local[:, 0]
        rotation_change = local[:, 5] - local[:, 2]
        gamma = (local[:, 4]-local[:, 1])/self._frame_length - (local[:, 2]+local[:, 5])/2
        return extension, rotation_change, gamma

    def _frame_coefficients(self):
        o, length = self.overlay, self._frame_length
        phi = 12 * o.EI_N_mm2 / (o.kGA_N * length**2)
        return (o.layer_count*o.EA_N/length, o.layer_count*o.EI_N_mm2/length,
                o.layer_count*12*o.EI_N_mm2/((1+phi)*length))

    def frame_energy(self, u, v):
        # Sum squared deformations to avoid cancellation of rigid-motion
        # components in x.T @ K @ x near a converged solution.
        return 0.5 * float(sum(np.sum(c*d*d) for c, d in
                               zip(self._frame_coefficients(), self._frame_deformations(u, v))))

    def frame_forces(self, u, v):
        """Local element end forces [N,V,M,N,V,M], smeared per axis mm."""
        axial, bending, shear_rotation = [c*d for c, d in
            zip(self._frame_coefficients(), self._frame_deformations(u, v))]
        shear = shear_rotation / self._frame_length
        return np.column_stack([-axial, -shear, -bending-shear_rotation/2,
                                axial, shear, bending-shear_rotation/2])
