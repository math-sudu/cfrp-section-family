"""Build the lining example from its bundled mesh, loads and section model."""
import json
import numpy as np

from analysis.paths import TUST_ROOT, ensure_local_importable

ensure_local_importable()
from fem_check.fem_core import OverlaySpec
from analysis.lining_model import FrameLiningProblem
from section_model import SectionFamily

CONFIG_PATH = TUST_ROOT / "configs/lining.json"
MESH_PATH = TUST_ROOT / "data/lining/mesh_boundary.npz"
SCHEMES = ("bare", "filled", "wrapped", "hybrid")


def inputs(case, steps, max_lambda):
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with np.load(MESH_PATH) as data:
        arrays = {k: data[k].copy() for k in data.files}
    seed = arrays.pop("seed")
    curvature = arrays.pop("overlay_curvature")
    schedule = settings["schedule"]
    dt = (max_lambda / steps * schedule["water_height_m"]
          / schedule["rise_rate_m_per_year"] * schedule["seconds_per_year"])
    section = overlay = mapping = None
    if case != "baseline":
        name, coverage = case.split("_")
        scheme = settings["schemes"][name]
        section = SectionFamily().section(**{k: scheme[k] for k in (
            "depth_mm", "wrap_mm", "grouted", "campaign")})
        pitch = scheme["depth_mm"] / (int(coverage) / 100.)
        EI = section.EI0 / pitch
        EA = EI / scheme["radius_squared_mm2"]
        kGA = scheme["shear_correction"] * EA / (2 * (1 + scheme["poisson_ratio"]))
        overlay = OverlaySpec(
            arrays["bnd_normals"], arrays["bnd_tangents"], arrays["bnd_ds"],
            curvature, EA, kGA, EI, 1)
        mapping = dict(scheme, coverage=int(coverage)/100., pitch_mm=pitch)
    problem = FrameLiningProblem(**arrays, **settings["material"],
                                viscous_dt=dt, overlay=overlay)
    problem.prepare()
    return problem, seed, section, {"double_layer": mapping}, {"bc_dict": settings["boundary"]}
