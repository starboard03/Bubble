import unittest

import numpy as np
import pandas as pd

from engine import build_features_upto, resolve_backtest_dates


class EngineTest(unittest.TestCase):
    def test_build_features_upto_returns_expected_columns(self):
        dates = pd.date_range("2021-01-01", periods=260, freq="D")
        prices = 100 * np.exp(np.linspace(0, 0.5, len(dates)))
        btc = pd.DataFrame({"close": prices}, index=dates)
        macro = pd.DataFrame(
            {
                "nasdaq": 10_000 * np.exp(np.linspace(0, 0.2, len(dates))),
                "dxy": 100 * np.exp(np.linspace(0, -0.05, len(dates))),
            },
            index=dates,
        )

        features = build_features_upto(btc, macro, dates[-1], trend_window=10)

        self.assertEqual(
            list(features.columns),
            ["btc_log_ret", "btc_vol_30d", "btc_ma200_dev", "nasdaq_log_ret", "dxy_log_ret"],
        )
        self.assertFalse(features.empty)
        self.assertFalse(features.isna().any().any())

    def test_resolve_backtest_dates_uses_latest_when_end_is_none(self):
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        btc = pd.DataFrame({"close": range(5)}, index=dates)

        resolved = resolve_backtest_dates(btc, "2022-01-03")

        self.assertEqual(resolved[0], pd.Timestamp("2022-01-03"))
        self.assertEqual(resolved[-1], pd.Timestamp("2022-01-05"))


if __name__ == "__main__":
    unittest.main()
