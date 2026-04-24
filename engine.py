"""
HMM regime engine for BTC/Cash allocation backtests.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import DATA_CONFIG


def load_backtest_data(data_dir: str = DATA_CONFIG.data_dir):
    btc_path = f"{data_dir}/btc_daily.csv"
    macro_path = f"{data_dir}/macro.csv"
    if not os.path.exists(btc_path) or not os.path.exists(macro_path):
        raise FileNotFoundError(
            "Backtest data files are missing. Run `python3 collect_data.py` first."
        )

    btc = pd.read_csv(btc_path, index_col="date", parse_dates=True)
    macro = pd.read_csv(macro_path, index_col="date", parse_dates=True)
    _validate_frame(btc, "BTC", {"close"})
    _validate_frame(macro, "Macro", {"nasdaq", "dxy"})
    return btc, macro


def _validate_frame(df: pd.DataFrame, label: str, required_columns: set[str]) -> None:
    if df.empty:
        raise RuntimeError(f"{label} backtest data is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"{label} backtest data must use a DatetimeIndex.")
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"{label} backtest data is missing columns: {sorted(missing)}")


def resolve_backtest_dates(btc: pd.DataFrame, start: str, end: Optional[str] = None):
    start_ts = pd.Timestamp(start)
    if end is None:
        dates = btc.loc[start_ts:].index
    else:
        end_ts = min(pd.Timestamp(end), btc.index.max())
        dates = btc.loc[start_ts:end_ts].index

    if len(dates) == 0:
        raise RuntimeError(
            f"No backtest dates available for requested range: start={start}, end={end or 'latest'}."
        )
    return dates


def build_features_upto(btc: pd.DataFrame, macro: pd.DataFrame, end_date, trend_window: int):
    b = btc.loc[:end_date].copy()
    raw_ret = np.log(b["close"] / b["close"].shift(1))
    b["btc_log_ret"] = raw_ret.rolling(trend_window).mean()
    b["btc_vol_30d"] = raw_ret.rolling(30).std()
    b["ma200"] = b["close"].rolling(200).mean()
    b["btc_ma200_dev"] = (b["close"] - b["ma200"]) / b["ma200"]

    m = macro.loc[:end_date].copy()
    m["nasdaq_log_ret"] = np.log(m["nasdaq"] / m["nasdaq"].shift(1)).rolling(trend_window).mean()
    m["dxy_log_ret"] = np.log(m["dxy"] / m["dxy"].shift(1)).rolling(trend_window).mean()

    features = b[["btc_log_ret", "btc_vol_30d", "btc_ma200_dev"]].join(
        m[["nasdaq_log_ret", "dxy_log_ret"]],
        how="inner",
    )
    return features.dropna()


def train_hmm_model(
    features: pd.DataFrame,
    n_components: int = 3,
    n_iter: int = 300,
    random_state: int = 42,
):
    if len(features) < 50:
        raise ValueError("train_hmm_model requires at least 50 feature rows.")

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(features.values)

    mean_ret = model.means_[:, 0]
    order = np.argsort(mean_ret)
    state_map = {
        int(order[2]): "bull",
        int(order[1]): "sideways",
        int(order[0]): "bear",
    }
    return model, state_map


def hmm_state_probs(model, state_map: dict[int, str], features: pd.DataFrame):
    _, posteriors = model.score_samples(features.values)
    today = posteriors[-1]
    return {regime: today[state] for state, regime in state_map.items()}


def hmm_allocation(model, state_map: dict[int, str], features: pd.DataFrame, weights=None):
    weights = weights or {"bull": 1.0, "sideways": 0.5, "bear": 0.0}
    probs = hmm_state_probs(model, state_map, features)
    allocation = sum(probs[regime] * weights[regime] for regime in weights)
    return probs, allocation
