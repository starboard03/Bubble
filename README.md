# Bubble

An HMM regime-aware risk model engine for BTC/Cash allocation. Bubble combines regime probabilities with CF-VaR, stress sizing, CPPI, and TIPP floor control to decide exposure under changing market conditions.

The backtest is a validation harness for the risk engine, not the main product surface. This repository is for research and model evaluation only; it is not investment advice, a trading bot, or a production portfolio management system.

## Architecture

```
Risk Engine
├── CF-VaR 99% (90d rolling + 2022 stress)
├── CPPI: Max Position = Cushion / CF-VaR
├── TIPP: Floor ratchets to 60% of portfolio peak
├── Regime Reset: Bull 80%+ for 5d + SMA50 crossover
└── Manual Stress Override (config)

Signal Layer
└── HMM Regime Detection
    5 features → 3-state Gaussian HMM → Bull/Sideways/Bear
    Features:
    - BTC trend
    - BTC volatility
    - BTC MA200 deviation
    - NASDAQ trend
    - DXY trend

Backtest Harness
└── Spot BTC / Cash allocation simulation for risk model evaluation
```

## Data Sources

| Data | Source |
|------|--------|
| BTC OHLCV | Binance REST API |
| NASDAQ / DXY | yfinance |

## Project Structure

```
├── config.py               # Shared runtime configuration
├── risk.py                 # CF-VaR, stress, CPPI, and TIPP risk engine
├── engine.py               # HMM feature, training, and regime probability engine
├── backtest.py             # Validation harness with TIPP/Regime Reset
├── collect_data.py         # BTC + macro data collection
├── tests/                  # Engine and risk smoke tests
├── LICENSE
└── requirements.txt
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Refresh historical data & run the validation harness
python3 collect_data.py
python3 backtest.py
```

## Docker

```bash
docker build -t bubble-risk-engine .

# Download data and run the validation harness with local CSV output persisted.
docker run --rm -v "$PWD/data:/app/data" bubble-risk-engine python collect_data.py
docker run --rm -v "$PWD/data:/app/data" bubble-risk-engine

# Run tests.
docker run --rm bubble-risk-engine python -m unittest
```

## Verification

```bash
python3 -m unittest
```

The full validation run downloads market data from Binance and yfinance, so it requires network access.

## Configuration

Runtime settings live in `config.py`.

- `DATA_CONFIG`: data directory, history start date, BTC symbol, macro tickers
- `CORE_BACKTEST_CONFIG`: main backtest capital, floor, retrain cadence, output path
- `CORE_BACKTEST_CONFIG.manual_stress`: manual worst-case daily drop override

`collect_data.py` now refreshes the local CSVs even when they already exist, so the default backtest path uses the latest fetched history instead of silently reusing stale files.

## Disclaimer

This project is for educational and research purposes only. Backtest results are not a guarantee of future performance. The code does not place orders, manage live positions, or include production risk controls.

## License

MIT License.

## Author

[@starboard03](https://github.com/starboard03)
