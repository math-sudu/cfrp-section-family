# Experimental inputs

Column names give units: millimetres, kilonewtons and microstrain. The tube
material calculation uses N and mm internally; the beam calculation uses kN
and mm. Configuration partitions and fitted coefficients are in `../parameters/`.

## Square tubes

`tubes/configurations.csv` describes 16 configurations with 30, 32 or 35 mm
depth, empty or cement-paste-filled cores, and 0, 1 or 2 mm nominal CFRP wrapping.
The 2 mm wraps occur at 30 and 32 mm depths. The tabulated peak and maximum of
the measured loading curve are distinct observations used by the peak network's two
outputs.

`tubes/curves.csv` contains the measured displacement–load curves from the authors'
three-point bending experiments. `point_index`
preserves their displacement-ordered sequence, including small load reversals.
Queries use the first upward, piecewise-linear crossing before the curve peak.
For example, `filled_d30_t2` denotes cement-paste infill, 30 mm depth and 2 mm
nominal wrapping.

Sources:

- Yuchen Dai (2023), *Research on Anti-Flexural Performance of Small-Caliber
  CFRP–Square Section Steel Tube Concrete*, master's dissertation,
  Beijing Jiaotong University (Chinese).
- Dai, Y., Chen, T. and Wang, F. (2024), *Research on flexural performance of
  small-caliber CFRP-square section steel tube concrete*, Journal of Central
  South University (Science and Technology), 55(1), 279–290.
  [DOI: 10.11817/j.issn.1672-7207.2024.01.023](https://doi.org/10.11817/j.issn.1672-7207.2024.01.023).

## Bolted HFRP beams

`beams/configurations.csv` describes 13 reinforced steel-beam configurations.
U and ST denote uniform and staggered bolt layouts. Strip lengths are 810,
1170 and 1620 mm; thicknesses are 3.175 and 6.35 mm. Configuration codes follow
the cited experimental studies.

`beams/deflections.csv` contains 26 specimen responses from the published
four-point bending experiments at 31 common total loads from 50 to 350 kN in
10 kN steps. At each load, the response is the first upward crossing before
the first global load maximum. The loader computes the arithmetic mean of two specimens at each
force. Both individual observations are retained to represent their spread.

`beams/strains.csv` contains two U90S100 training observations and two ST65S100
evaluation observations at 310 kN. Positions are distances from midspan.
`beams/connector_law.json` gives the M6 slip/force vertices and the model's
thickness scaling rule.

Sources:

- AbouEl-Hamd, O. R., Sweedan, A. M. I. and El-Ariss, B. (2022), *Flexural
  Performance of Steel Beams Strengthened by Fastened Hybrid FRP Strips
  Utilizing Staggered Steel Bolts*, Buildings 12, 2150.
  [DOI: 10.3390/buildings12122150](https://doi.org/10.3390/buildings12122150).
- AbouEl-Hamd, O. R., Sweedan, A. M. I., El-Ariss, B. and El-Sawy, K. M.
  (2023), *Experimental and Numerical Investigation of the Behavior of Steel
  Beams Strengthened by Bolted Hybrid FRP Composites*, Buildings 13, 824.
  [DOI: 10.3390/buildings13030824](https://doi.org/10.3390/buildings13030824).
- Abouelhamd, O. R. (2023), *Upgrading the Performance of Steel Beams Using
  Fastened Advanced Composite Materials*, PhD dissertation, United Arab
  Emirates University. [DOI: 10.83048/qdmb-bz91](https://doi.org/10.83048/qdmb-bz91).
- Abou El-Hamd, O. R., Sweedan, A. M. I. and El-Sawy, K. M. (2018),
  *Experimental and numerical study of the parameters controlling the behavior
  of double-lap connections of steel plates bolted to hybrid FRP strips*,
  Thin-Walled Structures 125, 140–151.
  [DOI: 10.1016/j.tws.2018.01.018](https://doi.org/10.1016/j.tws.2018.01.018).

The two Buildings articles are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The M6 connector
force–slip relation is reported in the 2023 Buildings article and the cited
2018 connection study. Numerical inputs
retain the source authors' attribution. Source publications and photographs
are not redistributed here.
