"""
Risk Module: Cornish-Fisher VaR
───────────────────────────────
Filter 1 : 최근 90일 CF-VaR (99%)
Filter 2 : 2022년 스트레스 CF-VaR (99%) → 비중 상한 캡
"""

import numpy as np
from scipy.stats import norm, skew, kurtosis
from typing import Optional


def cf_var(returns: np.ndarray, confidence: float = 0.99) -> dict:
    """
    Cornish-Fisher VaR
    - returns: 일간 로그수익률 배열
    - confidence: 신뢰도 (0.99 = 99%)
    - 반환: VaR(양수=손실), 통계치
    """
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        raise ValueError("cf_var requires at least 2 finite return values.")

    z = norm.ppf(1 - confidence)   # -2.326 for 99%
    mu = clean.mean()
    sigma = clean.std()
    if sigma <= 1e-12:
        return {
            "var_daily_pct": max(-mu, 0.0),
            "mu": mu,
            "sigma": sigma,
            "skew": 0.0,
            "excess_kurtosis": 0.0,
            "z_normal": z,
            "z_cf": z,
        }

    s = skew(clean)              # 왜도
    k = kurtosis(clean)          # 초과첨도 (scipy default: excess)
    if not np.isfinite(s):
        s = 0.0
    if not np.isfinite(k):
        k = 0.0

    # Cornish-Fisher 보정
    z_cf = (
        z
        + (z**2 - 1) / 6 * s
        + (z**3 - 3 * z) / 24 * k
        - (2 * z**3 - 5 * z) / 36 * s**2
    )

    var_daily = max(-(mu + z_cf * sigma), 0.0)  # 양수 = 일일 최대 손실률

    return {
        "var_daily_pct": var_daily,
        "mu": mu,
        "sigma": sigma,
        "skew": s,
        "excess_kurtosis": k,
        "z_normal": z,
        "z_cf": z_cf,
    }


def _fallback_var_stats(var_daily_pct: float) -> dict:
    return {
        "var_daily_pct": var_daily_pct,
        "mu": np.nan,
        "sigma": np.nan,
        "skew": np.nan,
        "excess_kurtosis": np.nan,
        "z_normal": np.nan,
        "z_cf": np.nan,
    }


def _max_alloc_from_var(cushion: float, portfolio_value: float, var_pct: float) -> float:
    if portfolio_value <= 0 or var_pct <= 0:
        return 0.0 if portfolio_value <= 0 else 1.0
    max_pos = cushion / var_pct
    return min(max_pos / portfolio_value, 1.0)


def compute_risk_state(
    portfolio_value: float,
    floor: float,
    ret_90d: np.ndarray,
    stress_returns: Optional[np.ndarray] = None,
    *,
    portfolio_peak: Optional[float] = None,
    tipp_ratio: Optional[float] = None,
    manual_stress: float = 0.0,
    confidence: float = 0.99,
    fallback_var: float = 0.15,
) -> dict:
    clean_90d = np.asarray(ret_90d, dtype=float)
    clean_90d = clean_90d[np.isfinite(clean_90d)]
    if clean_90d.size >= 30:
        var_90d_stats = cf_var(clean_90d, confidence)
    else:
        var_90d_stats = _fallback_var_stats(fallback_var)

    if stress_returns is None:
        var_stress_stats = var_90d_stats
    else:
        clean_stress = np.asarray(stress_returns, dtype=float)
        clean_stress = clean_stress[np.isfinite(clean_stress)]
        var_stress_stats = cf_var(clean_stress, confidence) if clean_stress.size >= 30 else var_90d_stats

    peak = portfolio_value if portfolio_peak is None else max(portfolio_peak, portfolio_value)
    effective_floor = floor if tipp_ratio is None else max(floor, peak * tipp_ratio)
    cushion = max(portfolio_value - effective_floor, 0.0)
    cushion_ratio = cushion / portfolio_value if portfolio_value > 0 else 0.0

    binding = "90d"
    binding_var = var_90d_stats["var_daily_pct"]
    if stress_returns is not None and var_stress_stats["var_daily_pct"] > binding_var:
        binding = "2022"
        binding_var = var_stress_stats["var_daily_pct"]
    if manual_stress > binding_var:
        binding = "manual"
        binding_var = manual_stress

    max_alloc_90d = _max_alloc_from_var(cushion, portfolio_value, var_90d_stats["var_daily_pct"])
    max_alloc_2022 = _max_alloc_from_var(cushion, portfolio_value, var_stress_stats["var_daily_pct"])
    max_alloc_manual = (
        _max_alloc_from_var(cushion, portfolio_value, manual_stress)
        if manual_stress > 0
        else 1.0
    )
    max_alloc = min(max_alloc_90d, max_alloc_2022, max_alloc_manual)

    return {
        "portfolio_value": portfolio_value,
        "base_floor": floor,
        "floor": effective_floor,
        "portfolio_peak": peak,
        "tipp_floor": effective_floor,
        "cushion": cushion,
        "cushion_ratio": cushion_ratio,
        "var_90d": var_90d_stats["var_daily_pct"],
        "var_2022": var_stress_stats["var_daily_pct"],
        "var_90d_stats": var_90d_stats,
        "var_2022_stats": var_stress_stats,
        "binding": binding,
        "binding_var": binding_var,
        "max_alloc_90d": max_alloc_90d,
        "max_alloc_2022": max_alloc_2022,
        "max_alloc_manual": max_alloc_manual,
        "max_alloc": max_alloc,
    }


def portfolio_cf_var(btc_returns: np.ndarray, btc_alloc: float,
                     confidence: float = 0.99) -> dict:
    """
    포트폴리오 CF-VaR = BTC 비중 × BTC CF-VaR
    (Cash는 무위험 가정)
    """
    btc_var = cf_var(btc_returns, confidence)
    port_var = btc_alloc * btc_var["var_daily_pct"]

    return {
        "btc_var": btc_var,
        "btc_alloc": btc_alloc,
        "portfolio_var_daily_pct": port_var,
    }


def stress_filter(btc_df, signal_alloc: float,
                  max_daily_loss: float = 0.05,
                  confidence: float = 0.99) -> dict:
    """
    2022년 스트레스 CF-VaR 기반 비중 상한 필터
    - btc_df: close 컬럼이 있는 BTC 일봉 DataFrame
    - signal_alloc: 시그널 시스템이 산출한 BTC 비중
    - max_daily_loss: 포트폴리오 일일 최대 허용 손실 (기본 5%)
    - 반환: 스트레스 VaR, 캡 비중, 최종 비중
    """
    btc_2022 = btc_df.loc["2022"]
    log_ret = np.log(btc_2022["close"] / btc_2022["close"].shift(1)).dropna()
    stress = cf_var(log_ret.values, confidence)

    # 캡: max_alloc × stress_var = max_daily_loss
    if stress["var_daily_pct"] > 0:
        max_alloc = max_daily_loss / stress["var_daily_pct"]
    else:
        max_alloc = 1.0
    max_alloc = min(max_alloc, 1.0)

    capped_alloc = min(signal_alloc, max_alloc)

    return {
        "stress_var": stress,
        "max_daily_loss": max_daily_loss,
        "max_alloc": max_alloc,
        "signal_alloc": signal_alloc,
        "capped_alloc": capped_alloc,
        "was_capped": capped_alloc < signal_alloc,
    }
