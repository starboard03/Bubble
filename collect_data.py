"""
HMM 레짐 백테스트용 히스토리 데이터 수집
─────────────────────────────────────
BTC 일봉, NASDAQ/DXY
2021-01-01 ~ 현재 (HMM 200MA 워밍업 포함)
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from config import DATA_CONFIG

DATA_DIR = DATA_CONFIG.data_dir
os.makedirs(DATA_DIR, exist_ok=True)
REQUIRED_BTC_COLUMNS = {"open", "high", "low", "close", "volume"}
REQUIRED_MACRO_COLUMNS = {"nasdaq", "dxy"}


def _validate_indexed_frame(df: pd.DataFrame, label: str, required_columns: set[str]) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError(f"{label} data is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"{label} data must use a DatetimeIndex.")
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"{label} data is missing columns: {sorted(missing)}")
    return df.sort_index()


def collect_btc():
    print("[1/2] BTC 일봉 (Binance)...")
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.fromisoformat(DATA_CONFIG.history_start).timestamp() * 1000)
    end_ms = int(datetime.utcnow().timestamp() * 1000)

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            url,
            params={
                "symbol": DATA_CONFIG.btc_symbol,
                "interval": DATA_CONFIG.btc_interval,
                "startTime": cursor,
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            raise RuntimeError(f"Binance API error while fetching BTC history: {data}")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Binance response type: {type(data).__name__}")
        if not data:
            break
        rows.extend(data)
        cursor = data[-1][6] + 1
        print(f"  {len(rows)}건...", end="\r")

    if not rows:
        raise RuntimeError("No BTC history returned from Binance.")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_vol",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df = _validate_indexed_frame(df, "BTC", REQUIRED_BTC_COLUMNS)

    path = f"{DATA_DIR}/btc_daily.csv"
    df.to_csv(path)
    print(f"  ✓ {len(df)}일  ({df.index[0].date()} ~ {df.index[-1].date()})  → {path}")
    return df


def collect_macro():
    print("[2/2] NASDAQ · DXY (yfinance)...")
    raw = yf.download(
        list(DATA_CONFIG.macro_tickers),
        start=DATA_CONFIG.history_start,
        end=datetime.now().strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty or "Close" not in raw:
        raise RuntimeError("yfinance returned no macro close data for NASDAQ/DXY.")
    close = raw["Close"]
    if not isinstance(close, pd.DataFrame):
        raise RuntimeError("Unexpected yfinance shape for macro close data.")
    close = close.rename(columns={"^IXIC": "nasdaq", "DX-Y.NYB": "dxy"})
    missing = REQUIRED_MACRO_COLUMNS - set(close.columns)
    if missing:
        raise RuntimeError(f"Missing macro columns from yfinance response: {sorted(missing)}")
    close = close[["nasdaq", "dxy"]]
    full_idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
    close = close.reindex(full_idx).ffill()
    close.index.name = "date"
    close = _validate_indexed_frame(close, "Macro", REQUIRED_MACRO_COLUMNS)

    path = f"{DATA_DIR}/macro.csv"
    close.to_csv(path)
    print(f"  ✓ {len(close)}일  ({close.index[0].date()} ~ {close.index[-1].date()})  → {path}")
    return close


def main():
    print("=" * 55)
    print("  HMM 레짐 백테스트 데이터 수집")
    print("=" * 55)
    t0 = time.time()

    print("기존 CSV가 있어도 최신 구간까지 다시 갱신합니다.")
    collect_btc()
    collect_macro()

    elapsed = time.time() - t0
    print(f"\n완료. ({elapsed:.0f}초)")
    print(f"저장 위치: {os.path.abspath(DATA_DIR)}/")


if __name__ == "__main__":
    main()
