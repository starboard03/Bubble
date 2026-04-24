"""
Shared runtime configuration for data collection and backtests.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    history_start: str = "2021-01-01"
    btc_symbol: str = "BTCUSDT"
    btc_interval: str = "1d"
    macro_tickers: Tuple[str, str] = ("^IXIC", "DX-Y.NYB")


@dataclass(frozen=True)
class CoreBacktestConfig:
    initial_capital: int = 10_000
    cppi_floor: int = 5_000
    tipp_ratio: float = 0.60
    hmm_retrain_days: int = 30
    hmm_min_rows: int = 200
    spot_fee: float = 0.001
    rebal_threshold: float = 0.05
    manual_stress: float = 0.0
    bt_start: str = "2022-01-01"
    bt_end: Optional[str] = None
    trend_window: int = 10
    output_path: str = "data/backtest_hmm_core.csv"


DATA_CONFIG = DataConfig()
CORE_BACKTEST_CONFIG = CoreBacktestConfig()
