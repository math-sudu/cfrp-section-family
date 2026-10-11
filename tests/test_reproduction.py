"""Executable checks of the released model, records and lining inputs."""
import unittest
import numpy as np

from analysis.evaluate import evaluate
from analysis.lining_inputs import inputs
from examples.member_response import member_response
from section_model import SectionFamily


class ReproductionTests(unittest.TestCase):
    def test_member_prediction_and_strength_endpoint(self):
        section = SectionFamily().section(depth_mm=32, wrap_mm=2, grouted=True, campaign=1)
        load, displacement = member_response(section, 490.0, [0.2, 0.5, 1.0])
        np.testing.assert_allclose(load[-1] / 1000.0, 48.04273594292151, rtol=1e-10)
        np.testing.assert_allclose(displacement, [1.326, 3.345, 9.288], atol=0.001)
        kappa = np.linspace(0, section.kappa_u, 301)
        moment = section.moment(kappa)
        self.assertTrue(np.all(np.diff(moment) > 0))
        np.testing.assert_allclose(moment[-1], section.Mu, rtol=1e-9)
        np.testing.assert_allclose(section.curvature(moment), kappa, atol=1e-12)

    def test_data_roles_and_reported_predictions(self):
        result = evaluate()
        self.assertEqual(len(result["calibration"]), 18)
        reserved = result["development_reserved"]
        self.assertEqual(len(reserved), 4)
        self.assertEqual(len(result["additional_capacities"]), 6)
        calibration_ids = {row["source_id"] for row in result["calibration"]}
        self.assertFalse(calibration_ids & {row["source_id"] for row in reserved})
        errors = sorted(row["curve_rel_l2"] for row in reserved)
        np.testing.assert_allclose(errors, [0.118, 0.138, 0.214, 0.219], atol=0.001)
        np.testing.assert_allclose(result["additional_S30_response"]["curve_rel_l2"],
                                   0.561, atol=0.001)

    def test_lining_geometry_and_section_transfer(self):
        baseline, seed, section, _, _ = inputs("baseline", 20, 2.0)
        self.assertEqual(baseline.n_nodes, 16519)
        self.assertEqual(baseline.T.shape, (31078, 3))
        self.assertEqual(len(baseline.bnd_idx), 1029)
        self.assertEqual(seed.shape, (16519,))
        self.assertIsNone(section)
        hybrid, _, section, mapping, _ = inputs("hybrid_018", 20, 2.0)
        pitch = mapping["double_layer"]["pitch_mm"]
        np.testing.assert_allclose(hybrid.overlay.EI_N_mm2, section.EI0 / pitch, rtol=1e-12)
        np.testing.assert_allclose(hybrid.viscous_dt, 44495302.32, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
