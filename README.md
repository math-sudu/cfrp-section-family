# FRP-strengthened steel member response

Code and input data for **Mechanics-informed neural prediction of nonlinear
flexural response and load transfer in FRP-strengthened steel members**, by
Pengcheng Zhu, Tielin Chen, Dingli Zhang, Yuchen Dai and Ye Shi.

The models predict nonlinear bending response for 16 CFRP-wrapped square-tube
configurations and 13 steel-beam configurations with bolted carbon–glass FRP
strips. Tube section mechanics determines a response shape and a neural network
learns its displacement scale. The beam model learns a convex section potential
through axial–bending equilibrium and predicts deflection and local strip strain.

## Install

Use Python 3.11 or later and run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The reference environment uses Python 3.13, NumPy 2.4.4, SciPy 1.16.2,
PyTorch 2.8.0 and threadpoolctl 3.6.0. Computations use CPU float64 and one
numerical thread; no GPU is required.

## Reproduce predictions

```bash
python reproduce.py
```

This evaluates all four tube exclusions and beam models M_R, M_0, M_B and M_N
using the supplied data and fitted parameters. It writes individual predictions,
wrapping savings, errors and strip strains to `results/reproduction.json`.

Expected values, rounded as in the paper:

| Tube saving RMSE (mm) | Affine amplitude | Neural amplitude |
| --- | ---: | ---: |
| At selected forces | 0.127 | 0.071 |
| Over predicted force windows | 0.125 | 0.081 |

| Beam model | M_R | M_0 | M_F | M_B | M_N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggregate deflection RMSE (mm) | 0.436 | 0.429 | 0.361 | 0.280 | 0.269 |

M_N predicts approximately 1254 and 583 microstrain at the ST65S100 gauges
at 310 kN; the observations are 1300 and 510 microstrain.

M_F solves an additional fiber equilibrium problem and takes longer. Run it
separately or include it in the complete comparison:

```bash
python reproduce.py --section beams --models M_F --output results/fiber.json
python reproduce.py --section beams --models M_R M_0 M_F M_B M_N --output results/all_beams.json
```

Add `--benchmark` for the Table 4 timing protocol: two warmups and seven measured
repetitions of the 33-load-state prediction and two-gauge calculation. Geometry
construction and file reads are excluded; compliance conditioning is included.
Timings depend on hardware. The paper used an AMD Ryzen Threadripper 3970X.

## Refit

```bash
python refit.py --component tubes --model peaks --protocol leave_wrap_2 --output results/tubes_peaks.json
python refit.py --component tubes --model M_T --protocol leave_wrap_2 --parameters results/tubes_peaks.json --output results/tubes_refitted.json
python reproduce.py --section tubes --tube-parameters results/tubes_refitted.json --output results/tubes_predictions.json
python refit.py --component beams --model M_N --output results/beams_refitted.json
python reproduce.py --section beams --models M_N --beam-parameters results/beams_refitted.json --output results/beams_predictions.json
```

Other tube protocols are `leave_depth_30`, `leave_depth_32` and `leave_depth_35`.
A and S are the linear and shared empirical-shape comparisons. Peak refits use
the published selected penalties and three seeded starts. Response refits select
penalties using the supplied three configuration folds. M_T refits the section
shape, projected amplitudes and neural amplitude; the 2 mm exclusion also refits
the affine comparison. Beam refits accept M_R, M_0, M_B, M_F or M_N. M_N starts
from an approximation to the training-fitted M_B law, then minimizes the joint
displacement/strain loss. Beam refitting takes substantially longer than prediction.
Refit outputs are separate from the supplied fitted coefficients.

## Files and information conditions

- `data/tubes/`: configurations and digitized displacement–load curves.
- `data/beams/`: configurations, two specimen responses per design, strain
  observations and the connector law.
- `parameters/`: fitted coefficients, networks and configuration partitions.
- `frp_flexure/tube_mechanics.py`: moment sharing and shape/amplitude calibration.
- `frp_flexure/tube_network.py`: networks and empirical response comparisons.
- `frp_flexure/beam.py`: equilibrium, connectors, section models and gradients.
- [Data sources and conventions](data/README.md): provenance and units.

The four 2 mm tube configurations are excluded from their networks' training.
The three 1170 mm beam configurations supply only their 50–200 kN deflections
to set reference compliance. Their 250–350 kN deflections and ST65S100 strains
are evaluation data. The U90S100 strain observations belong to training.

The earlier tube-section-family implementation remains in the Git history.
The current files support the manuscript named above.

For correspondence: Tielin Chen, tlchen1@bjtu.edu.cn.
