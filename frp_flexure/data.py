"""Read the experimental observations supplied with the member models."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parameters(name, path=None):
    return json.loads((Path(path) if path else ROOT / "parameters" / f"{name}.json").read_text(encoding="utf-8"))


@dataclass
class Trace:
    key: str
    depth: float
    wrap: float
    filled: int
    table_peak: float
    points: np.ndarray

    @property
    def features(self):
        return np.array([(self.depth - 32) / 5, self.wrap - 1, self.filled - .5])

    @property
    def peak(self):
        return float(self.points[:, 1].max())

    @property
    def branch(self):
        return self.points[:int(np.argmax(self.points[:, 1])) + 1]


def load_tubes():
    curves = read_csv("data/tubes/curves.csv")
    traces = []
    for row in read_csv("data/tubes/configurations.csv"):
        selected = sorted((r for r in curves if r["configuration"] == row["configuration"]),
                          key=lambda r: int(r["point_index"]))
        points = np.array([[float(r["displacement_mm"]), float(r["load_kn"])] for r in selected])
        traces.append(Trace(row["configuration"], float(row["depth_mm"]), float(row["wrap_mm"]),
                            int(row["filled"]), float(row["table_peak_kn"]), points))
    return traces


def first_crossing(points, load):
    """Interpolate the first upward crossing of the piecewise-linear trace."""
    for (ua, pa), (ub, pb) in zip(points[:-1], points[1:]):
        if pa <= load <= pb:
            return float(ua if pb == pa else ua + (ub - ua) * (load - pa) / (pb - pa))
    raise ValueError(f"No upward crossing at {load:g} kN")


def load_beams():
    rows = read_csv("data/beams/deflections.csv")
    records = {}
    for name in sorted({r["configuration"] for r in rows}):
        selected = [r for r in rows if r["configuration"] == name]
        curves = {}
        for specimen in sorted({r["specimen"] for r in selected}):
            curve = sorted((r for r in selected if r["specimen"] == specimen),
                           key=lambda r: float(r["load_kn"]))
            loads = np.array([float(r["load_kn"]) for r in curve])
            if not np.array_equal(loads, np.arange(50., 351., 10.)):
                raise ValueError(f"Expected the 50–350 kN grid for {specimen}")
            curves[specimen] = [float(r["deflection_mm"]) for r in curve]
        if len(curves) != 2:
            raise ValueError(f"Expected two specimens for {name}")
        records[name] = {"specimen_trajectories_mm": curves,
                         "equal_force_mean": np.mean(list(curves.values()), axis=0).tolist()}
    return records


def load_strains(configuration):
    rows = [r for r in read_csv("data/beams/strains.csv") if r["configuration"] == configuration]
    if not rows:
        raise KeyError(configuration)
    return {"load_kn": float(rows[0]["load_kn"]),
            "positions_mm": [float(r["distance_from_midspan_mm"]) for r in rows],
            "microstrain": [float(r["microstrain"]) for r in rows]}
