#!/usr/bin/env python3
"""Daily ML trading pipeline for Railway deployment.

Runs once daily (triggered by cron):
1. Download latest daily bars via yfinance (stocks + macro)
2. Compute features (62 stock + 22 macro = 84 total)
3. Run LightGBM predictions → rank stocks
4. Rebalance portfolio via Alpaca API (long top 20)
5. Log results + send notification

Config: H5_LongOnly20 — 5-day horizon, long top 20, no shorts.
Rebalances every 5 trading days.

Usage:
    python pipeline.py                  # Full run
    python pipeline.py --dry-run        # Predict only, no orders
    python pipeline.py --force          # Force rebalance even if not due
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TOP_N = 20              # Number of stocks to hold long
HORIZON = 5             # Prediction horizon (trading days)
MIN_HISTORY_DAYS = 250  # Minimum days of history needed for features
LOOKBACK_DAYS = 300     # Days of history to download for feature computation

# Alpaca
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "ml_v4_model.pkl"
STATE_PATH = BASE_DIR / "state" / "pipeline_state.json"
LOG_DIR = BASE_DIR / "logs"

# Macro tickers needed for features
MACRO_TICKERS = [
    "^VIX", "SPY", "QQQ", "IWM", "TLT", "SHY", "HYG",
    "GLD", "USO", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]

MACRO_RENAME = {"^VIX": "VIX"}

# Symbols excluded from trading (IPO too recent, data quality issues)
EXCLUDED_SYMBOLS = {
    "RIVN", "LCID", "PLTR", "HOOD", "AFRM", "IONQ", "JOBY",
    "DNA", "GRAB", "NU", "RKLB", "VFS", "SMCI",
}

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    for h in handlers:
        h.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in handlers:
        logger.addHandler(h)

    return logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def get_tradeable_symbols() -> list[str]:
    """Get S&P 500 + Nasdaq 100 symbols from Wikipedia."""
    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        sp500 = []

    try:
        ndx = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
        )
        for table in ndx:
            if "Ticker" in table.columns:
                ndx_syms = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                break
            elif "Symbol" in table.columns:
                ndx_syms = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                break
        else:
            ndx_syms = []
    except Exception:
        ndx_syms = []

    all_syms = sorted(set(sp500 + ndx_syms) - EXCLUDED_SYMBOLS)
    return all_syms


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance column names (handles MultiIndex from newer versions)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def download_bars(symbols: list[str], days: int, logger) -> dict[str, pd.DataFrame]:
    """Download recent daily bars from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))  # Extra buffer for weekends/holidays

    logger.info(f"Downloading {len(symbols)} symbols ({start.date()} → {end.date()})...")

    data = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tickers_str = " ".join(batch)
        try:
            df = yf.download(
                tickers_str, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker", auto_adjust=True, progress=False,
                threads=True,
            )
            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df.copy()
                    else:
                        sym_df = df[sym].copy()
                    sym_df = _normalize_columns(sym_df)
                    sym_df = sym_df.dropna(subset=["close"])
                    if len(sym_df) >= MIN_HISTORY_DAYS:
                        data[sym] = sym_df
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")
        time.sleep(0.3)  # Rate limit

    logger.info(f"Downloaded {len(data)} symbols with >= {MIN_HISTORY_DAYS} days")
    return data


def download_macro(days: int, logger) -> dict[str, pd.DataFrame]:
    """Download macro tickers."""
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    macro = {}
    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            df = _normalize_columns(df)
            name = MACRO_RENAME.get(ticker, ticker)
            macro[name] = df.dropna(subset=["close"])
        except Exception as e:
            logger.warning(f"Macro {ticker} failed: {e}")
        time.sleep(0.1)

    logger.info(f"Downloaded {len(macro)} macro tickers")
    return macro


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as train_ml_v4.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock daily features — identical to training pipeline."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()

    feats = {}

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


def compute_macro_features(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute market-level features — identical to training pipeline."""
    feats = {}

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
            feats[f"{name}_ret5"] = c.pct_change(5) * 100
            feats[f"{name}_ret20"] = c.pct_change(20) * 100
            feats[f"{name}_sma50"] = (c - c.rolling(50).mean()) / (c.rolling(50).mean() + 1e-8) * 100
            feats[f"{name}_sma200"] = (c - c.rolling(200).mean()) / (c.rolling(200).mean() + 1e-8) * 100
            feats[f"{name}_vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
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
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_rankings(
    stock_data: dict, macro_features: pd.DataFrame,
    model, feature_cols: list, logger,
) -> list[tuple[str, float]]:
    """Generate predictions for all stocks, return sorted rankings."""
    predictions = {}

    for sym, df in stock_data.items():
        try:
            stock_feats = compute_stock_features(df)

            # Merge with macro
            if not macro_features.empty:
                merged = stock_feats.join(macro_features, how="left")
                macro_cols = macro_features.columns
                merged[macro_cols] = merged[macro_cols].ffill()
            else:
                merged = stock_feats

            # Get latest row
            latest = merged.iloc[-1:]
            avail_cols = [c for c in feature_cols if c in merged.columns]

            if len(avail_cols) < len(feature_cols) * 0.9:
                continue  # Too many missing features

            row = latest[avail_cols]
            if row.isna().any(axis=1).iloc[0]:
                # Fill remaining NaN with 0 (same as training)
                row = row.fillna(0)

            # Ensure column order matches training
            missing = [c for c in feature_cols if c not in row.columns]
            for mc in missing:
                row[mc] = 0
            row = row[feature_cols]

            pred = model.predict(row)[0]
            if np.isfinite(pred):
                predictions[sym] = float(pred)

        except Exception as e:
            logger.debug(f"Prediction failed for {sym}: {e}")

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Generated predictions for {len(ranked)} stocks")

    if ranked:
        logger.info(f"Top 5: {[(s, f'{p:.2f}') for s, p in ranked[:5]]}")
        logger.info(f"Bottom 5: {[(s, f'{p:.2f}') for s, p in ranked[-5:]]}")

    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# ALPACA TRADING
# ═══════════════════════════════════════════════════════════════════════════

def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def alpaca_request(method: str, endpoint: str, data=None):
    """Make an Alpaca API request."""
    import requests
    url = f"{ALPACA_BASE_URL}/{endpoint}"
    headers = get_alpaca_headers()

    if method == "GET":
        resp = requests.get(url, headers=headers)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unknown method: {method}")

    if resp.status_code not in (200, 204, 207):
        raise Exception(f"Alpaca {method} {endpoint} failed: {resp.status_code} {resp.text}")

    if resp.status_code == 204:
        return {}
    return resp.json()


def get_account(logger):
    """Get Alpaca account info."""
    acct = alpaca_request("GET", "v2/account")
    logger.info(f"Account: ${float(acct['portfolio_value']):,.2f} "
                f"(cash: ${float(acct['cash']):,.2f}, "
                f"buying_power: ${float(acct['buying_power']):,.2f})")
    return acct


def get_positions(logger) -> dict[str, dict]:
    """Get current positions."""
    positions = alpaca_request("GET", "v2/positions")
    pos_dict = {}
    for p in positions:
        pos_dict[p["symbol"]] = {
            "qty": float(p["qty"]),
            "market_value": float(p["market_value"]),
            "unrealized_pl": float(p["unrealized_pl"]),
            "side": p["side"],
        }
    logger.info(f"Current positions: {len(pos_dict)} stocks")
    return pos_dict


def rebalance_portfolio(
    target_symbols: list[str], logger, dry_run: bool = False,
):
    """Rebalance to equal-weight long portfolio of target symbols."""
    acct = get_account(logger)
    portfolio_value = float(acct["portfolio_value"])
    current_positions = get_positions(logger)

    target_set = set(target_symbols)
    current_set = set(current_positions.keys())

    to_sell = current_set - target_set
    to_buy = target_set - current_set
    to_hold = current_set & target_set

    target_weight = portfolio_value / len(target_symbols)
    orders = []

    # 1. Sell positions not in target
    for sym in to_sell:
        qty = current_positions[sym]["qty"]
        logger.info(f"  SELL {sym}: {qty} shares (not in target)")
        if not dry_run:
            orders.append(("sell", sym, qty))

    # 2. Rebalance existing positions
    for sym in to_hold:
        current_value = current_positions[sym]["market_value"]
        diff = target_weight - current_value
        if abs(diff) > target_weight * 0.1:  # Only rebalance if >10% off target
            logger.info(f"  REBALANCE {sym}: current ${current_value:,.0f} → target ${target_weight:,.0f}")
            if not dry_run:
                if diff > 0:
                    orders.append(("buy_notional", sym, diff))
                else:
                    orders.append(("sell_notional", sym, abs(diff)))

    # 3. Buy new positions
    for sym in to_buy:
        logger.info(f"  BUY {sym}: ${target_weight:,.0f} (new position)")
        if not dry_run:
            orders.append(("buy_notional", sym, target_weight))

    if dry_run:
        logger.info(f"DRY RUN — would execute {len(to_sell)} sells, "
                     f"{len(to_buy)} buys, {len(to_hold)} holds")
        return {"sells": len(to_sell), "buys": len(to_buy), "holds": len(to_hold)}

    # Execute orders
    executed = 0
    failed = 0

    # Execute sells first (free up capital)
    for action, sym, amount in orders:
        try:
            if action == "sell":
                alpaca_request("DELETE", f"v2/positions/{sym}")
            elif action == "buy_notional":
                alpaca_request("POST", "v2/orders", {
                    "symbol": sym, "notional": round(amount, 2),
                    "side": "buy", "type": "market", "time_in_force": "day",
                })
            elif action == "sell_notional":
                alpaca_request("POST", "v2/orders", {
                    "symbol": sym, "notional": round(amount, 2),
                    "side": "sell", "type": "market", "time_in_force": "day",
                })
            executed += 1
            time.sleep(0.1)  # Rate limit
        except Exception as e:
            logger.error(f"Order failed for {sym}: {e}")
            failed += 1

    logger.info(f"Orders: {executed} executed, {failed} failed")
    return {"executed": executed, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """Load pipeline state from JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_rebalance": None, "last_run": None, "run_count": 0, "history": []}


def save_state(state: dict):
    """Save pipeline state to JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def should_rebalance(state: dict, force: bool = False) -> bool:
    """Check if we should rebalance today."""
    if force:
        return True
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    days_since = (datetime.now() - last_date).days
    return days_since >= HORIZON


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(dry_run: bool = False, force: bool = False):
    """Execute the full daily pipeline."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"ML Trading Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: H{HORIZON}_LongOnly{TOP_N}, dry_run={dry_run}")
    logger.info("=" * 60)

    # Check API keys
    if not dry_run and (not ALPACA_API_KEY or not ALPACA_SECRET_KEY):
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    # Load model
    logger.info("\n1. Loading model...")
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    model_bundle = pickle.load(open(MODEL_PATH, "rb"))
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    logger.info(f"Model loaded: {len(feature_cols)} features, "
                f"horizon={model_bundle['horizon']}d")

    # Check state
    state = load_state()
    if not should_rebalance(state, force):
        logger.info(f"Not rebalancing today (last: {state['last_rebalance']}, "
                     f"next in {HORIZON - (datetime.now() - datetime.fromisoformat(state['last_rebalance'])).days}d)")
        state["last_run"] = datetime.now().isoformat()
        save_state(state)
        return

    # Download data
    logger.info("\n2. Downloading market data...")
    symbols = get_tradeable_symbols()
    logger.info(f"Universe: {len(symbols)} symbols")

    stock_data = download_bars(symbols, LOOKBACK_DAYS, logger)
    macro_data = download_macro(LOOKBACK_DAYS, logger)

    # Compute features
    logger.info("\n3. Computing features...")
    macro_features = compute_macro_features(macro_data)
    logger.info(f"Macro features: {len(macro_features.columns)} columns, "
                f"{len(macro_features)} days")

    # Generate predictions
    logger.info("\n4. Generating predictions...")
    rankings = predict_rankings(stock_data, macro_features, model, feature_cols, logger)

    if len(rankings) < TOP_N:
        logger.error(f"Only {len(rankings)} predictions — need at least {TOP_N}")
        sys.exit(1)

    target_symbols = [sym for sym, _ in rankings[:TOP_N]]
    logger.info(f"\nTarget portfolio ({TOP_N} stocks):")
    for i, (sym, pred) in enumerate(rankings[:TOP_N]):
        logger.info(f"  {i+1:2d}. {sym:6s}  predicted return: {pred:+.2f}%")

    # Rebalance
    logger.info("\n5. Rebalancing portfolio...")
    result = rebalance_portfolio(target_symbols, logger, dry_run=dry_run)

    # Update state
    state["last_rebalance"] = datetime.now().isoformat()
    state["last_run"] = datetime.now().isoformat()
    state["run_count"] = state.get("run_count", 0) + 1
    state["history"].append({
        "date": datetime.now().isoformat(),
        "target_symbols": target_symbols,
        "predictions": {s: p for s, p in rankings[:TOP_N]},
        "result": result,
    })
    # Keep last 100 entries
    state["history"] = state["history"][-100:]
    save_state(state)

    logger.info(f"\nPipeline complete! Run #{state['run_count']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Trading Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Predict only, no orders")
    parser.add_argument("--force", action="store_true", help="Force rebalance")
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run, force=args.force)
