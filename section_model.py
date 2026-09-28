"""Call the fitted section-law family without the parent project or a refit.

Run ``python section_model.py --help`` or see README.md for the NumPy API.
The complete coefficient matrix is read from the current fit JSON. ``Mk``
is the transition scale called ``Mcr`` in the retained fit/analysis records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from section_backbone import base_grid, logistic, solve_endpoint

VARIANT = "derived_ku_fixed_knee_mcr_frac_eiceiling"
DEFAULT_FIT = Path(__file__).resolve().parent / "data/model/coefficients.json"


class SectionFamily:
    """Fixed coefficients, one normalized shape and two configuration scales."""

    def __init__(self, fit_path=DEFAULT_FIT):
        fit = json.loads(Path(fit_path).read_text(encoding="utf-8"))
        if fit["variant"] != VARIANT:
            raise ValueError(f"Expected the shared-shape fit {VARIANT}")
        self.coefficients = np.asarray(fit["A"], dtype=float)
        if (self.coefficients.shape != (4, 8)
                or not np.all(np.isfinite(self.coefficients))
                or np.any(self.coefficients[1:3, 1:] != 0.0)):
            raise ValueError("The fit requires four eight-feature rows with shared Mk/Mu and r1")
        self.q = float(logistic(self.coefficients[1, 0]))
        self.r1 = 0.02 + 0.96 * float(logistic(self.coefficients[2, 0]))
        self.z_u, ok = solve_endpoint(1.0, self.q, self.r1, 1.0, upper_bound=np.inf)
        if not ok:
            raise ValueError("The fitted shape did not reach its strength endpoint")
        _, _, self.z, transition, plateau = base_grid(1.0, self.q, self.z_u)
        self.h = (self.r1 + (1.0 - self.r1) * transition) * plateau
        segments = 0.5 * (self.h[1:] + self.h[:-1]) * np.diff(self.z)
        self.m = np.r_[0.0, np.cumsum(segments)]

    def section(self, *, depth_mm, wrap_mm, grouted, campaign=1):
        """Evaluate descriptors; campaigns 1 and 2 use c = 0 and 1 respectively.

        The fitted records use depths 30--35 mm and nominal wraps 0--2 mm.
        Infill is 0/1; dimensions remain continuous model inputs.
        """
        d, t = float(depth_mm), float(wrap_mm)
        if not np.isfinite(d) or not np.isfinite(t) or d <= 0.0 or t < 0.0:
            raise ValueError("Depth must be positive and wrap thickness nonnegative, both finite")
        if grouted not in (False, True) or campaign not in (1, 2):
            raise ValueError("Infill must be 0/1 and campaign must be 1/2")
        g, c = float(grouted), float(campaign - 1)
        x = np.array([1.0, (d - 32.0) / 5.0, t, t*t, g, g*t, c, t*c])
        EI0 = 1.0e9 * float(np.logaddexp(0.0, self.coefficients[0] @ x))
        Mu = 5.0e5 * float(np.logaddexp(0.0, self.coefficients[3] @ x))
        if not np.isfinite(EI0) or not np.isfinite(Mu) or min(EI0, Mu) <= 0.0:
            raise ValueError("Descriptors produced a nonfinite or zero section scale")
        return SectionCurve(self, EI0, Mu)


class SectionCurve:
    """Positive-loading section relation up to the identified strength endpoint."""

    def __init__(self, family, EI0, Mu):
        self.family, self.EI0, self.Mu = family, EI0, Mu
        self.Mk, self.r1 = family.q * Mu, family.r1
        self.curvature_scale = Mu / EI0
        self.kappa_k = self.Mk / EI0
        self.kappa_u = family.z_u * self.curvature_scale

    @staticmethod
    def _within(values, endpoint, name):
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > endpoint)):
            raise ValueError(f"{name} must lie in [0, {endpoint}]")
        return values

    def moment(self, kappa):
        """M(kappa) in N mm; kappa is in mm^-1, from zero to kappa_u."""
        k = self._within(kappa, self.kappa_u, "Curvature")
        return self.Mu * np.interp(k / self.curvature_scale, self.family.z, self.family.m)

    def curvature(self, moment):
        """Inverse relation kappa(M) in mm^-1, from zero to Mu."""
        m = self._within(moment, self.Mu, "Moment")
        return self.curvature_scale * np.interp(m / self.Mu, self.family.m, self.family.z)

    def secant_stiffness(self, kappa):
        """M/kappa in N mm^2, with the tangent limit at zero curvature."""
        k = self._within(kappa, self.kappa_u, "Curvature")
        result = np.full_like(k, self.EI0 * self.family.h[0])
        return np.divide(self.moment(k), k, out=result, where=k != 0.0)

    def parameters(self):
        """Return parameters in N and mm, using the manuscript's Mk notation."""
        return {name: float(getattr(self, name)) for name in
                ("EI0", "Mk", "r1", "Mu", "kappa_k", "kappa_u")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--depth-mm", type=float, required=True)
    parser.add_argument("--wrap-mm", type=float, required=True)
    parser.add_argument("--grouted", action="store_true")
    parser.add_argument("--campaign", type=int, choices=(1, 2), default=1)
    parser.add_argument("--curve", type=Path, help="Write 201 curve samples to this CSV")
    args = parser.parse_args(argv)
    family = SectionFamily(args.fit)
    section = family.section(depth_mm=args.depth_mm, wrap_mm=args.wrap_mm,
                             grouted=args.grouted, campaign=args.campaign)
    print(json.dumps({"units": {"moment": "N mm", "stiffness": "N mm^2", "curvature": "1/mm"},
                      "parameters": section.parameters(),
                      "M_at_kappa_k": float(section.moment(section.kappa_k)),
                      "q": family.q, "z_u": family.z_u}, indent=2))
    if args.curve is not None:
        kappa = np.linspace(0.0, section.kappa_u, 201)
        np.savetxt(args.curve, np.column_stack((kappa, section.moment(kappa),
                                               section.secant_stiffness(kappa))),
                   delimiter=",", comments="",
                   header="kappa_per_mm,moment_N_mm,secant_stiffness_N_mm2")


if __name__ == "__main__":
    main()
