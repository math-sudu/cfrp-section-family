"""Central-load response of the filled 32 mm, 2 mm-wrap first-campaign member."""
import json
import numpy as np
from section_model import SectionFamily


def member_response(section, span_mm, load_fractions):
    fractions = np.asarray(load_fractions, dtype=float)
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("Load fractions must be between zero and one")
    peak = 4.0 * section.Mu / span_mm
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x = 0.25 * span_mm * (nodes + 1.0)
    curvature = section.curvature(fractions[:, None] * peak * x / 2.0)
    deflection = 0.25 * span_mm * np.sum(weights * x * curvature, axis=1)
    return peak * fractions, deflection


if __name__ == "__main__":
    section = SectionFamily().section(depth_mm=32, wrap_mm=2, grouted=True, campaign=1)
    fractions = [0.2, 0.5, 1.0]
    loads, deflections = member_response(section, 490.0, fractions)
    print(json.dumps({"span_mm": 490.0, "load_fractions": fractions,
                      "load_kN": (loads / 1000).tolist(),
                      "deflection_mm": deflections.tolist()}, indent=2))
