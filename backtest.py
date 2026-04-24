"""
백테스트 — HMM 레짐 코어 모드 (Walk-Forward, 미래참조 없음)
════════════════════════════════════════════════════════
- HMM 상태별: Bull → max alloc, Sideways → 0.5, Bear → 0.0
- 리스크 캡: cushion_ratio / CF-VaR_99, 현물 max 1.0x
- TIPP floor + Regime Reset 적용
- 뉴스/파생 시그널 없음
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from engine import (
    build_features_upto,
    hmm_state_probs,
    load_backtest_data,
    resolve_backtest_dates,
    train_hmm_model,
)
from config import CORE_BACKTEST_CONFIG
from risk import compute_risk_state

warnings.filterwarnings("ignore")

INITIAL_CAPITAL = CORE_BACKTEST_CONFIG.initial_capital
CPPI_FLOOR = CORE_BACKTEST_CONFIG.cppi_floor
TIPP_RATIO = CORE_BACKTEST_CONFIG.tipp_ratio
HMM_RETRAIN_DAYS = CORE_BACKTEST_CONFIG.hmm_retrain_days
HMM_MIN_ROWS = CORE_BACKTEST_CONFIG.hmm_min_rows

SPOT_FEE = CORE_BACKTEST_CONFIG.spot_fee
REBAL_THRESHOLD = CORE_BACKTEST_CONFIG.rebal_threshold
MANUAL_STRESS = CORE_BACKTEST_CONFIG.manual_stress

BT_START = CORE_BACKTEST_CONFIG.bt_start
BT_END = CORE_BACKTEST_CONFIG.bt_end
TREND_WINDOW = CORE_BACKTEST_CONFIG.trend_window
OUTPUT_PATH = CORE_BACKTEST_CONFIG.output_path

def apply_regime_reset(today, btc, btc_price, probs, bull_streak, portfolio_value, risk_state):
    cash_locked = risk_state["cushion_ratio"] < 0.05 or (
        risk_state["binding_var"] > 0
        and risk_state["cushion_ratio"] / risk_state["binding_var"] < 0.1
    )

    bull_streak = bull_streak + 1 if probs["bull"] >= 0.80 else 0
    sma50_val = btc.loc[today, "sma50"] if today in btc.index else None
    price_above_sma50 = (
        sma50_val is not None
        and not np.isnan(sma50_val)
        and btc_price > sma50_val
    )

    if not (cash_locked and bull_streak >= 5 and price_above_sma50):
        return bull_streak, False, risk_state

    updated = dict(risk_state)
    updated["tipp_floor"] = portfolio_value * TIPP_RATIO
    updated["portfolio_peak"] = portfolio_value
    updated["cushion"] = max(portfolio_value - updated["tipp_floor"], 0)
    updated["cushion_ratio"] = updated["cushion"] / portfolio_value if portfolio_value > 0 else 0.0
    updated["max_alloc"] = (
        min(updated["cushion_ratio"] / updated["binding_var"], 1.0)
        if updated["binding_var"] > 0
        else 0.0
    )
    return bull_streak, True, updated


def print_summary(df, btc, total_fee):
    total_ret = df["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - 1
    days = (df.index[-1] - df.index[0]).days
    ann_ret = (1 + total_ret) ** (365 / days) - 1

    daily_rets = df["portfolio_value"].pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(365) if daily_rets.std() > 0 else 0
    peak = df["portfolio_value"].cummax()
    max_dd = ((df["portfolio_value"] - peak) / peak).min()

    btc_start = btc.loc[df.index[0], "close"]
    btc_end = btc.loc[df.index[-1], "close"]
    btc_bnh = btc_end / btc_start - 1

    print("\n" + "=" * 60)
    print("  백테스트 결과 (HMM 레짐 코어)")
    print("=" * 60)
    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()} ({days}일)")
    print(f"  초기 자본: ${INITIAL_CAPITAL:,}")
    print(f"  최종 자본: ${df['portfolio_value'].iloc[-1]:,.0f}")
    print(f"  총 수익률: {total_ret:+.1%}")
    print(f"  연환산 수익률: {ann_ret:+.1%}")
    print(f"  샤프 비율: {sharpe:.2f}")
    print(f"  최대 낙폭: {max_dd:.1%}")
    print(f"  Floor 이탈: {'없음' if df['portfolio_value'].min() >= CPPI_FLOOR else '있음!'}")
    print("-" * 60)
    print(f"  BTC 바이앤홀드: {btc_bnh:+.1%}")
    print(f"  BTC 대비 초과수익: {total_ret - btc_bnh:+.1%}")
    print("-" * 60)
    print(f"  평균 BTC 비중: {df['btc_alloc'].mean():.1%}")
    print(f"  평균 턴오버: {df['turnover'].mean():.2%}")
    print(f"  총 수수료: ${total_fee:,.0f}")

    resets = df["regime_reset"].sum()
    print(f"  Regime Reset: {resets}회")
    if resets > 0:
        for rd in df[df["regime_reset"]].index:
            print(f"    → {rd.date()}  PV=${df.loc[rd, 'portfolio_value']:,.0f}")
    print("=" * 60)
    print(f"  저장: {OUTPUT_PATH}")


def run_backtest():
    print("데이터 로드...")
    btc, macro = load_backtest_data()
    btc["log_ret"] = np.log(btc["close"] / btc["close"].shift(1))
    btc["sma50"] = btc["close"].rolling(50).mean()

    dates = resolve_backtest_dates(btc, BT_START, BT_END)
    print(f"백테스트: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}일)")
    print("모드: 현물 HMM 레짐 코어")

    portfolio_value = INITIAL_CAPITAL
    portfolio_peak = INITIAL_CAPITAL
    prev_alloc = 0.0
    total_fee = 0.0
    hmm_model = None
    hmm_smap = None
    last_train_date = None
    alloc_ema = 0.5
    ema_alpha = 2 / (7 + 1)
    bull_streak = 0
    results = []

    for i, today in enumerate(dates):
        btc_price = btc.loc[today, "close"]

        need_retrain = (
            hmm_model is None
            or (today - last_train_date).days >= HMM_RETRAIN_DAYS
        )
        if need_retrain:
            features = build_features_upto(
                btc,
                macro,
                today - timedelta(days=1),
                trend_window=TREND_WINDOW,
            )
            if len(features) >= HMM_MIN_ROWS:
                hmm_model, hmm_smap = train_hmm_model(features)
                last_train_date = today

        ret_90d = btc["log_ret"].loc[:today - timedelta(days=1)].dropna().values[-90:]
        stress_returns = btc["log_ret"].loc["2022"].dropna().values if today >= pd.Timestamp("2023-01-01") else None
        risk_state = compute_risk_state(
            portfolio_value=portfolio_value,
            floor=CPPI_FLOOR,
            ret_90d=ret_90d,
            stress_returns=stress_returns,
            portfolio_peak=portfolio_peak,
            tipp_ratio=TIPP_RATIO,
            manual_stress=MANUAL_STRESS,
        )
        portfolio_peak = risk_state["portfolio_peak"]

        if hmm_model is not None:
            features_now = build_features_upto(btc, macro, today, trend_window=TREND_WINDOW)
            probs = hmm_state_probs(hmm_model, hmm_smap, features_now)
        else:
            probs = {"bull": 0.0, "sideways": 1.0, "bear": 0.0}

        bull_streak, regime_reset, risk_state = apply_regime_reset(
            today,
            btc,
            btc_price,
            probs,
            bull_streak,
            portfolio_value,
            risk_state,
        )
        portfolio_peak = risk_state["portfolio_peak"]

        hmm_alloc_raw = (
            probs["bull"] * risk_state["max_alloc"]
            + probs["sideways"] * 0.5
            + probs["bear"] * 0.0
        )
        alloc_ema = ema_alpha * hmm_alloc_raw + (1 - ema_alpha) * alloc_ema
        signal_alloc = alloc_ema

        final_alloc = min(signal_alloc, risk_state["max_alloc"])
        final_alloc = max(final_alloc, 0.0)

        if abs(final_alloc - prev_alloc) < REBAL_THRESHOLD:
            final_alloc = prev_alloc

        turnover = abs(final_alloc - prev_alloc)
        fee = turnover * SPOT_FEE * portfolio_value
        portfolio_value -= fee
        total_fee += fee
        prev_alloc = final_alloc

        if i + 1 < len(dates):
            next_day = dates[i + 1]
            if next_day in btc.index and not np.isnan(btc.loc[next_day, "log_ret"]):
                btc_ret = btc.loc[next_day, "log_ret"]
                spot_ret = final_alloc * (np.exp(btc_ret) - 1)
                portfolio_value *= (1 + spot_ret)

        results.append({
            "date": today,
            "portfolio_value": portfolio_value,
            "btc_alloc": final_alloc,
            "signal_alloc": signal_alloc,
            "max_alloc": risk_state["max_alloc"],
            "turnover": turnover,
            "fee": fee,
            "p_bull": probs["bull"],
            "p_sideways": probs["sideways"],
            "p_bear": probs["bear"],
            "var_90d": risk_state["var_90d"],
            "var_2022": risk_state["var_2022"],
            "cushion": risk_state["cushion"],
            "tipp_floor": risk_state["tipp_floor"],
            "btc_price": btc_price,
            "regime_reset": regime_reset,
        })

        if (i + 1) % 100 == 0:
            print(
                f"  [{i+1}/{len(dates)}] {today.date()}  "
                f"PV=${portfolio_value:,.0f}  BTC={final_alloc:.1%}"
            )

    df = pd.DataFrame(results).set_index("date")
    df.to_csv(OUTPUT_PATH)
    print_summary(df, btc, total_fee)
    return df


if __name__ == "__main__":
    run_backtest()
