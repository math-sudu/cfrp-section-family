"""Exercise the published prediction entry point and its experimental inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from frp_flexure import beam
from frp_flexure.data import ROOT, load_beams, load_tubes, parameters
from refit import peak_fit


class ReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            subprocess.run([sys.executable, "reproduce.py", "--output", str(output)],
                           cwd=ROOT, check=True)
            cls.result = json.loads(output.read_text(encoding="utf-8"))

    def test_inputs_and_partitions(self):
        self.assertEqual(len(load_tubes()), 16)
        inputs = load_beams()
        self.assertEqual(len(inputs), 13)
        fitted = parameters("beams")
        self.assertFalse(set(fitted["training"]) & set(fitted["testing"]))
        self.assertEqual(len(fitted["training"]), 10)

    def test_tube_tables(self):
        results = self.result["tubes"]
        self.assertAlmostEqual(results["leave_depth_30"]["central_increment_rmse_mm"]["M_T"], .319, delta=.0005)
        for name, expected in [("leave_depth_32", .101), ("leave_depth_35", .256), ("leave_wrap_2", .071)]:
            self.assertAlmostEqual(results[name]["common_force_saving_rmse_mm"]["M_T"], expected, delta=.0005)
        row = results["leave_wrap_2"]
        self.assertAlmostEqual(row["affine_selected_rmse_mm"], .127, delta=.0005)
        self.assertAlmostEqual(row["affine_window_rmse_mm"], .125, delta=.0005)
        self.assertAlmostEqual(row["predicted_window_rmse_mm"]["M_T"], .081, delta=.0005)

    def test_beam_predictions(self):
        for model, expected in [("M_R", .436), ("M_0", .429), ("M_B", .280), ("M_N", .269)]:
            self.assertAlmostEqual(self.result["beams"][model]["aggregate_rmse_mm"], expected, delta=.0005)
        np.testing.assert_allclose(self.result["beams"]["M_N"]["predicted_strain_microstrain"],
                                   [1254.025000764922, 582.8542792815358], atol=1e-5)

    def test_fiber_strain(self):
        solution = parameters("beams")["models"]["M_F"]
        compliance = beam.observed_compliances(load_beams())
        gauge = beam.ConditionedFiber(["ST65S100"], [310.], compliance, **solution["prediction_options"])
        with torch.no_grad():
            strain = gauge.strain(torch.tensor(solution["log_parameters"]), [(95., 380.)])[0].numpy()
        np.testing.assert_allclose(strain, [1257.3649200697068, 584.3055802105234], atol=1e-4)

    def test_peak_refit(self):
        record = parameters("tubes")["leave_wrap_2"]
        traces = {t.key: t for t in load_tubes()}
        training = [traces[k] for k in record["training"]]
        fitted = peak_fit(training, record["peak_model"]["penalty"], record["peak_model"]["seed"])
        np.testing.assert_allclose(fitted["theta"], record["peak_model"]["theta"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
