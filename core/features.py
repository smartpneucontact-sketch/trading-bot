"""Feature engineering — per-stock + macro + cross-sectional.

Three feature functions, one per model lineage:

  compute_stock_features    — v4 base (62 features)
  compute_stock_features_v6 — v4 + ~25 new signals (drawdown, momentum,
                              efficiency, ad/vol divergence, etc.)
                              Used by v5/v6/v8 models.

Macro features (compute_macro_features) and cross-sectional rank features
(_add_cross_sectional_ranks) live here too. The model selects which
feature function to apply via `_get_feature_func(feature_version)`.

Output is always a wide DataFrame keyed by date — every column is
float32 to keep model inputs lightweight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Per-stock features
# ═══════════════════════════════════════════════════════════════════════════

def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock daily features — identical to v4 training pipeline."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()

    feats: dict[str, pd.Series] = {}

    # Returns at multiple horizons
    for d in [1, 2, 3, 5, 10, 20, 60, 120]:
        feats[f"ret_{d}d"] = c.pct_change(d) * 100

    # Moving average distances
    for p in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(p, min_periods=p).mean()
        feats[f"sma_{p}"] = (c - sma) / (sma + 1e-8) * 100
    for p in [20, 50, 100]:
        sma = c.rolling(p, min_periods=p).mean()
        feats[f"sma_{p}_slope"] = sma.pct_change(5) * 100

    # EMA distances
    for p in [12, 26, 50]:
        ema = c.ewm(span=p, adjust=False).mean()
        feats[f"ema_{p}"] = (c - ema) / (ema + 1e-8) * 100

    # RSI
    for p in [5, 14, 21]:
        delta = c.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        rs = up.rolling(p).mean() / (down.rolling(p).mean() + 1e-8)
        feats[f"rsi_{p}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = (macd - macd_signal) / (c + 1e-8) * 100

    # Volatility
    for p in [5, 10, 20, 60]:
        feats[f"vol_{p}"] = ret.rolling(p).std() * np.sqrt(252)

    # Bollinger %B
    for p in [10, 20]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        feats[f"bb_{p}"] = (c - (mid - 2 * std)) / (4 * std + 1e-8)

    # ATR
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    for p in [5, 14, 20]:
        feats[f"atr_{p}"] = tr.rolling(p).mean() / (c + 1e-8) * 100

    # Volume features
    v_sma20 = v.rolling(20).mean()
    feats["vol_ratio"] = v / (v_sma20 + 1e-8)
    feats["vol_trend"] = v.rolling(5).mean() / (v_sma20 + 1e-8)

    # OBV trend
    obv = (v * ret.apply(np.sign)).cumsum()
    obv_sma = obv.rolling(20).mean()
    feats["obv_trend"] = (obv - obv_sma) / (obv_sma.abs() + 1e-8)

    # Momentum / ROC
    for p in [5, 10, 20, 60]:
        feats[f"roc_{p}"] = c.pct_change(p) * 100

    # Statistical
    for p in [20, 60]:
        feats[f"skew_{p}"] = ret.rolling(p).skew()
        feats[f"kurt_{p}"] = ret.rolling(p).kurt()

    # Sharpe
    for p in [20, 60]:
        rm = ret.rolling(p)
        feats[f"sharpe_{p}"] = (rm.mean() / (rm.std() + 1e-8)) * np.sqrt(252)

    # Channel position
    for p in [10, 20, 40, 60]:
        ch_h = h.rolling(p).max()
        ch_l = l.rolling(p).min()
        feats[f"channel_{p}"] = (c - ch_l) / (ch_h - ch_l + 1e-8)

    # Distance from highs/lows
    for p in [20, 60, 120]:
        feats[f"dist_hi_{p}"] = (c - h.rolling(p).max()) / (c + 1e-8) * 100
        feats[f"dist_lo_{p}"] = (c - l.rolling(p).min()) / (c + 1e-8) * 100

    # Intraday features from daily bar
    feats["day_range"] = (h - l) / (c + 1e-8) * 100
    feats["close_loc"] = (c - l) / (h - l + 1e-8)

    # Calendar
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        feats["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 5), index=df.index)
        feats["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 5), index=df.index)
    if hasattr(df.index, "month"):
        m = df.index.month
        feats["month_sin"] = pd.Series(np.sin(2 * np.pi * m / 12), index=df.index)
        feats["month_cos"] = pd.Series(np.cos(2 * np.pi * m / 12), index=df.index)

    result = pd.DataFrame(feats, index=df.index)
    for col in result.columns:
        result[col] = result[col].astype(np.float32)
    return result


def compute_stock_features_v6(df: pd.DataFrame) -> pd.DataFrame:
    """V6 feature set = v4 base + new mean-reversion / drawdown / efficiency signals."""
    feats_df = compute_stock_features(df)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()

    # Mean-reversion: residual momentum vs short-window trend.
    for p in [20, 60]:
        trend = c.rolling(p).mean().pct_change(5) * 100
        actual = c.pct_change(5) * 100
        feats_df[f"resid_mom_{p}"] = (actual - trend).astype(np.float32)
    feats_df["reversal_20_5"] = (
        (c.pct_change(5) * 100) - (c.pct_change(20) * 100 * 0.25)
    ).astype(np.float32)

    # Trend deviation (EMA-based)
    for p in [20, 60, 120]:
        ema = c.ewm(span=p, adjust=False).mean()
        feats_df[f"trend_dev_{p}"] = ((c - ema) / (ema + 1e-8) * 100).astype(np.float32)

    # Return z-scores
    for p in [20, 60]:
        rm = ret.rolling(p)
        feats_df[f"ret_zscore_{p}"] = (
            (ret - rm.mean()) / (rm.std() + 1e-8)
        ).astype(np.float32)

    # SMA crossovers
    sma_20 = c.rolling(20).mean()
    sma_50 = c.rolling(50).mean()
    sma_200 = c.rolling(200).mean()
    feats_df["sma_20_50_cross"] = ((sma_20 - sma_50) / (sma_50 + 1e-8) * 100).astype(np.float32)
    feats_df["sma_50_200_cross"] = ((sma_50 - sma_200) / (sma_200 + 1e-8) * 100).astype(np.float32)

    # Volatility ratios
    feats_df["vol_ratio_5_20"] = (
        ret.rolling(5).std() / (ret.rolling(20).std() + 1e-8)
    ).astype(np.float32)
    feats_df["vol_ratio_10_60"] = (
        ret.rolling(10).std() / (ret.rolling(60).std() + 1e-8)
    ).astype(np.float32)

    # Dollar volume ratio (v5 feature)
    dollar_vol = c * v
    dollar_vol_sma20 = dollar_vol.rolling(20).mean()
    feats_df["dollar_vol_ratio"] = (
        dollar_vol / (dollar_vol_sma20 + 1e-8)
    ).astype(np.float32)

    # Volume-price divergence
    v_sma20 = v.rolling(20).mean()
    feats_df["vp_divergence"] = (
        v / (v_sma20 + 1e-8) - (1 + ret.rolling(5).mean() * 10)
    ).astype(np.float32)

    # Accumulation/Distribution
    clv = ((c - lo) - (h - c)) / (h - lo + 1e-8)
    ad = (clv * v).cumsum()
    ad_sma = ad.rolling(20).mean()
    feats_df["ad_trend"] = ((ad - ad_sma) / (ad_sma.abs() + 1e-8)).astype(np.float32)

    # Gap feature
    if "open" in df.columns:
        feats_df["gap"] = (
            (df["open"].astype(float) / c.shift(1) - 1) * 100
        ).astype(np.float32)

    # Drawdown features
    rolling_max_60 = c.rolling(60).max()
    rolling_max_120 = c.rolling(120).max()
    feats_df["drawdown_60"] = (
        (c - rolling_max_60) / (rolling_max_60 + 1e-8) * 100
    ).astype(np.float32)
    feats_df["drawdown_120"] = (
        (c - rolling_max_120) / (rolling_max_120 + 1e-8) * 100
    ).astype(np.float32)
    dd_60 = (c - rolling_max_60) / (rolling_max_60 + 1e-8)
    feats_df["recovery_speed_60"] = (dd_60 - dd_60.shift(5)).astype(np.float32)

    # Consecutive up/down days
    feats_df["consec_up_5"] = (ret > 0).astype(float).rolling(5).sum().astype(np.float32)
    feats_df["consec_down_5"] = (ret < 0).astype(float).rolling(5).sum().astype(np.float32)

    # Move vs ATR
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats_df["move_vs_atr"] = (
        c.diff(5).abs() / (atr_14 * np.sqrt(5) + 1e-8)
    ).astype(np.float32)

    # Volume acceleration
    feats_df["vol_accel"] = (
        (v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8))
        - (v.rolling(20).mean() / (v.rolling(60).mean() + 1e-8))
    ).astype(np.float32)

    # Price efficiency
    for p in [5, 20]:
        directional = c.diff(p).abs()
        total_path = ret.abs().rolling(p).sum() * c + 1e-8
        feats_df[f"efficiency_{p}"] = (directional / total_path).astype(np.float32)

    return feats_df


# ═══════════════════════════════════════════════════════════════════════════
# Macro features
# ═══════════════════════════════════════════════════════════════════════════

def compute_macro_features(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute market-level features keyed by date.

    Emits each SPY-derived feature under BOTH `SP500_*` and `SPY_*` names —
    older training scripts used the SP500 prefix, newer ones SPY. They're
    identical series. The duplication keeps both eras of models loadable
    without retraining.
    """
    feats: dict[str, pd.Series] = {}

    if "VIX" in macro_data:
        vix = macro_data["VIX"]["close"]
        feats["vix"] = vix
        feats["vix_sma20"] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).mean() + 1e-8) * 100
        feats["vix_change5"] = vix.pct_change(5) * 100
        feats["vix_rank20"] = vix.rolling(60).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )

    for name in ["SP500", "SPY"]:
        if name in macro_data:
            c = macro_data[name]["close"]
            ret5 = c.pct_change(5) * 100
            ret20 = c.pct_change(20) * 100
            sma50_ = (c - c.rolling(50).mean()) / (c.rolling(50).mean() + 1e-8) * 100
            sma200 = (c - c.rolling(200).mean()) / (c.rolling(200).mean() + 1e-8) * 100
            vol20 = c.pct_change().rolling(20).std() * np.sqrt(252)
            for alias in ("SP500", "SPY"):
                feats[f"{alias}_ret5"] = ret5
                feats[f"{alias}_ret20"] = ret20
                feats[f"{alias}_sma50"] = sma50_
                feats[f"{alias}_sma200"] = sma200
                feats[f"{alias}_vol20"] = vol20
            break

    if "TLT" in macro_data and "SHY" in macro_data:
        ratio = macro_data["TLT"]["close"] / macro_data["SHY"]["close"]
        feats["yield_curve"] = (ratio - ratio.rolling(60).mean()) / (ratio.rolling(60).std() + 1e-8)
        feats["yield_curve_ret20"] = ratio.pct_change(20) * 100

    if "HYG" in macro_data:
        hyg = macro_data["HYG"]["close"]
        feats["credit_ret5"] = hyg.pct_change(5) * 100
        feats["credit_ret20"] = hyg.pct_change(20) * 100

    for name, prefix in [("UUP", "dollar"), ("GLD", "gold"), ("USO", "oil")]:
        if name in macro_data:
            c = macro_data[name]["close"]
            feats[f"{prefix}_ret5"] = c.pct_change(5) * 100
            feats[f"{prefix}_ret20"] = c.pct_change(20) * 100

    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU"]
    available_sectors = [s for s in sector_etfs if s in macro_data]
    if len(available_sectors) >= 4:
        sector_rets = pd.DataFrame()
        for s in available_sectors:
            sector_rets[s] = macro_data[s]["close"].pct_change(20) * 100
        feats["sector_spread"] = sector_rets.max(axis=1) - sector_rets.min(axis=1)
        feats["sector_dispersion"] = sector_rets.std(axis=1)

    if "IWM" in macro_data and "SPY" in macro_data:
        ratio = macro_data["IWM"]["close"] / macro_data["SPY"]["close"]
        feats["small_vs_large"] = ratio.pct_change(20) * 100

    result = pd.DataFrame(feats)
    result.index = pd.to_datetime(result.index)
    for col in result.columns:
        result[col] = result[col].astype(np.float32)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Cross-sectional ranks + feature-function dispatch
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_func(feature_version: str):
    """Return the appropriate feature computation function for a model version."""
    if feature_version in ("v5", "v6", "v8"):
        # v5 features are a subset of v6; v8 uses v6 base + sector-relative.
        return compute_stock_features_v6
    return compute_stock_features


def add_cross_sectional_ranks(
    latest_rows: dict[str, pd.Series],
    feature_cols: list[str],
) -> dict[str, pd.Series]:
    """Add cross-sectional rank features (cs_*) across all stocks.

    For each base feature that has a `cs_*` counterpart in `feature_cols`,
    compute the percentile rank across all stocks at the latest time point.
    Stocks missing a base feature get dropped (returned under the
    `__cs_dropped__` key) rather than imputed — synthesizing 0.5 for
    missing ranks would lie to the model.
    """
    cs_cols = [c for c in feature_cols if c.startswith("cs_")]
    if not cs_cols:
        return latest_rows

    cs_to_base = {cs_col: cs_col[3:] for cs_col in cs_cols}
    syms = list(latest_rows.keys())
    base_cols_needed = set(cs_to_base.values())

    cs_data: dict[str, pd.Series] = {}
    for base_col in base_cols_needed:
        vals: dict[str, float] = {}
        for sym in syms:
            row = latest_rows[sym]
            if base_col in row.index:
                vals[sym] = row[base_col]
        if vals:
            cs_data[base_col] = pd.Series(vals)

    cs_ranks: dict[str, pd.Series] = {}
    for base_col, series in cs_data.items():
        if len(series) > 1:
            cs_ranks[base_col] = series.rank(pct=True)
        else:
            cs_ranks[base_col] = series * 0 + 0.5  # neutral if only 1 stock

    result: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for sym in syms:
        row = latest_rows[sym].copy()
        ok = True
        for cs_col, base_col in cs_to_base.items():
            ranks = cs_ranks.get(base_col)
            if ranks is None or sym not in ranks.index or pd.isna(ranks.get(sym)):
                ok = False
                break
            row[cs_col] = np.float32(ranks[sym])
        if ok:
            result[sym] = row
        else:
            dropped.append(sym)

    if dropped:
        result["__cs_dropped__"] = dropped  # surfaced once to the caller
    return result
