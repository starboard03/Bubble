import unittest

import numpy as np

from risk import cf_var, compute_risk_state


class RiskTest(unittest.TestCase):
    def test_cf_var_returns_positive_loss_estimate(self):
        returns = np.array([-0.05, -0.03, -0.02, 0.0, 0.01, 0.02, 0.03])

        result = cf_var(returns, confidence=0.99)

        self.assertGreater(result["var_daily_pct"], 0)
        self.assertIn("z_cf", result)

    def test_compute_risk_state_preserves_manual_stress_and_tipp(self):
        ret_90d = np.full(90, -0.01)
        stress_returns = np.full(365, -0.02)

        state = compute_risk_state(
            portfolio_value=10_000,
            floor=5_000,
            ret_90d=ret_90d,
            stress_returns=stress_returns,
            portfolio_peak=12_000,
            tipp_ratio=0.60,
            manual_stress=0.40,
        )

        self.assertEqual(state["binding"], "manual")
        self.assertEqual(state["tipp_floor"], 7_200)
        self.assertEqual(state["cushion"], 2_800)
        self.assertAlmostEqual(state["max_alloc"], 0.7)


if __name__ == "__main__":
    unittest.main()
