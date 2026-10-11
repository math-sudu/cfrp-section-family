# CFRP-wrapped tube section family

Research code and executable input data for **Flexural assessment of CFRP
composite tubes for tunnel lining reinforcement using an integral constrained
physics informed neural network**, by Pengcheng Zhu, Tielin Chen, Dingli Zhang,
Yuchen Dai and Ye Shi.

This local working version includes the 29 September curve-peak correction:
all six campaign-2 fitting peaks follow their curves, including S30-1 at
18.7595 kN. The original published v1.0.0 remains in Git history; these local
changes have not been pushed or published as a new release.

The model identifies a shared dimensionless moment-curvature shape and two
configuration-dependent scales from member bending records. Its positive
tangent stiffness integrates to the identified strength at the terminal
curvature. The repository includes the fitted model, joint identification,
leave-one-configuration-out evaluation and the section-mapped lining example.

## Installation

The tested environment is Python 3.13 with NumPy 2.4.4, SciPy 1.16.2 and
PyYAML 6.0.2. Run commands from this repository's root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Use the fitted model

```bash
python section_model.py --depth-mm 32 --wrap-mm 2 --grouted --campaign 1
python -m examples.member_response
```

The first command prints the section parameters. Add `--curve section_curve.csv`
to export its ascending moment-curvature curve. The second evaluates a simply
supported member under central loading. At a 490 mm span, the filled 32 mm
tube with a nominal 2 mm wrap has predicted peak load **48.043 kN** and peak-load
deflection **9.288 mm**.

```python
import numpy as np
from section_model import SectionFamily

family = SectionFamily()
section = family.section(depth_mm=32, wrap_mm=2, grouted=True, campaign=1)
kappa = np.linspace(0, section.kappa_u, 201)
moment = section.moment(kappa)
secant_stiffness = section.secant_stiffness(kappa)
inverse_curvature = section.curvature(0.5 * section.Mu)
```

Units are N and mm: moment in N mm, stiffness in N mm² and curvature in mm⁻¹.
The measured tube depths are 30-35 mm and nominal wraps are 0-2 mm. Campaign 1
is the carbon-steel campaign; campaign 2 is the stainless-steel campaign.
`Mk` is the transition scale, stored as `Mcr` in the identification code; the
actual moment at the transition is obtained from `section.moment(section.kappa_k)`.
The API covers the ascending branch through the strength endpoint.

## Evaluate and refit

```bash
python -m analysis.evaluate
python -m analysis.refit --n-jobs 6
python -m analysis.loco --n-jobs 6
python -m analysis.replicate_scatter
```

`evaluate` uses the bundled full-precision coefficients and generates the
18 calibration, four development-reserved and six additional capacity checks.
`refit` performs the joint coefficient fit. `loco` refits on 21 configurations
for each of 22 folds. The original four reserved responses were visible during
model development; that evaluation differs from the 22-fold calculation.
`replicate_scatter` reads the twelve reported specimen pairs.

Settings are in `configs/identification.yaml`. Six seeded restarts are used
per fit. Fit and cross-validation runtime depends on the machine; `--n-jobs`
controls concurrent restarts. `--out` selects the generated JSON destination.
For a newly fitted coefficient file, pass its path to `SectionFamily(fit_path)`
or use `section_model.py --fit path/to/fit.json ...`.

## Lining example

```bash
python -m analysis.lining_demo --case baseline --stop-step 1
python -m analysis.lining_demo --case baseline
python -m analysis.lining_demo --case bare_018 --max-sweeps 2000
python -m analysis.lining_demo --case hybrid_018 --max-sweeps 2000
```

Other reinforced cases combine `bare`, `filled`, `wrapped` or `hybrid` with
`_018` or `_036`, the two coverage fractions. The reported `bare_018` and
`hybrid_018` runs used 2000 maximum stagger sweeps per increment; the other
cases used the default 600. Twenty increments reach load factor 2. The solver
resumes its own checkpoint; `--out` selects a separate output directory.

`configs/lining.json` and `data/lining/mesh_boundary.npz` supply the resolved
mesh, loading, bedding, initial crack seed and section mappings. The mesh file
is an input, approximately 0.74 MB. The frame has independent section rotations
and is bonded to the lining's extrados translations. The time-increment energy
includes `eta * (alpha-alpha_prev)**2 / (2*dt)` and damage irreversibility.

## Included data and generated outputs

The required numerical inputs are bundled with the code under `data/`:

- Member curves and peak-load tables used by the supplied analyses.
- The fitted coefficient matrix, stored as a small model input in
  `data/model/coefficients.json`.
- The resolved lining mesh and boundary arrays.

[Data documentation](data/README.md) gives units, record roles and measurement
conditions. The digitized curves and tabulated values are the available
experimental records. Original instrument acquisition files were not retained.
The second campaign's 450 mm span is an upper estimate from the press platform;
its support-block spacing was not recorded.

Results, field histories, checkpoints and plots are generated locally and are
excluded from Git. Manuscript files, inspection photographs and the operating
line's restricted crack registry are outside this code release.

For correspondence: Tielin Chen, tlchen1@bjtu.edu.cn.
