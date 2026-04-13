#!/usr/bin/env python3
"""Daily ML trading pipeline for Railway deployment — Multi-Model Edition.

Runs once daily (triggered by cron):
1. Download latest daily bars via yfinance (stocks + macro)
2. Compute features (62 stock + 22 macro = 84 total for v4)
3. Run model predictions → rank stocks
4. Rebalance portfolio via Alpaca API (long top 20)
5. Log full run report + structured trade journal

Config: H5_LongOnly20 — 5-day horizon, long top 20, no shorts.
Rebalances every 5 trading days.

Multi-model: Reads MODEL_{name}_ALPACA_KEY / MODEL_{name}_ALPACA_SECRET
env vars. Each model trades its own Alpaca account independently.

Usage:
    python pipeline.py                          # Full run, all models
    python pipeline.py --dry-run                # Predict only, no orders
    python pipeline.py --force                  # Force rebalance
    python pipeline.py --model v4               # Run single model only
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf


# ═══════════════════════════════════════════════════════════════════════════
# STACKED ENSEMBLE (needed for v6 model pickle deserialization)
# ═══════════════════════════════════════════════════════════════════════════

class StackedEnsemble:
    """Stacked ensemble: base models + ridge meta-learner (v6)."""

    def __init__(self, base_models: list, meta_model, meta_scaler=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_scaler = meta_scaler

    def predict(self, X):
        base_preds = np.column_stack([
            model.predict(X) for _, model in self.base_models
        ])
        if self.meta_scaler is not None:
            base_preds = self.meta_scaler.transform(base_preds)
        return self.meta_model.predict(base_preds)


class EnsembleModel:
    """Weighted ensemble of sub-models (v5).

    Each entry in self.models is (name, model, weight).
    Prediction = weighted average of sub-model predictions.
    """

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        total_weight = sum(w for _, _, w in self.models)
        preds = np.zeros(len(X))
        for name, model, weight in self.models:
            preds += model.predict(X) * weight
        return preds / total_weight

    @property
    def feature_importances_(self):
        coefs = np.abs(self.meta_model.coef_)
        coefs = coefs / coefs.sum()
        imp = np.zeros_like(self.base_models[0][1].feature_importances_, dtype=float)
        for i, (name, model) in enumerate(self.base_models):
            if hasattr(model, 'feature_importances_'):
                imp += model.feature_importances_ * coefs[i]
        return imp

    @property
    def meta_weights(self):
        return {name: round(float(w), 4)
                for (name, _), w in zip(self.base_models, self.meta_model.coef_)}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TOP_N = 20              # Number of stocks to hold long
HORIZON = 5             # Prediction horizon (trading days)
MIN_HISTORY_DAYS = 250  # Minimum days of history needed for features
LOOKBACK_DAYS = 300     # Days of history to download for feature computation

# Persistent data dir (Railway volume)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = DATA_DIR / "logs"
TRADE_DIR = DATA_DIR / "trades"

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
# MULTI-MODEL CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a single model instance."""
    name: str                   # e.g. "v4", "v5"
    model_path: Path            # path to model.pkl
    feature_version: str        # "v4", "v5", or "v6"
    alpaca_key: str             # per-model Alpaca API key
    alpaca_secret: str          # per-model Alpaca API secret
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    state_path: Path = None     # auto-set if None
    enable_cutloss: bool = False  # V7+: enable intraday cut-loss scanner
    cutloss_hard_stop: float = -8.0    # hard stop: sell if down X% from entry
    cutloss_trailing_stop: float = -5.0  # trailing stop: sell if down X% from peak
    cutloss_portfolio_stop: float = -3.0  # portfolio drawdown: go to cash if daily loss > X%

    def __post_init__(self):
        if self.state_path is None:
            self.state_path = DATA_DIR / "state" / f"pipeline_state_{self.name}.json"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DESCRIPTIONS — displayed on dashboard
# ═══════════════════════════════════════════════════════════════════════════

MODEL_DESCRIPTIONS = {
    "v4": {
        "title": "V4 — Single LightGBM Baseline",
        "summary": "The original production model. A single LightGBM gradient-boosted "
                   "tree trained to predict 5-day forward returns.",
        "architecture": "Single LightGBM model (gradient boosting)",
        "features": "84 features: 62 stock technical indicators (momentum, volatility, "
                    "volume, moving averages, RSI, MACD, Bollinger Bands, etc.) + "
                    "22 macro features (VIX, SPY/QQQ/IWM trends, sector ETFs, "
                    "TLT/HYG credit spreads, gold, oil, dollar index)",
        "portfolio": "Equal-weight top 20 stocks. Rebalance every 5 trading days.",
        "risk": "No intraday monitoring. Market regime not considered. "
                "Holds through volatility events.",
        "training": "Trained on ~4 years of daily data across S&P 500 + Nasdaq 100 + "
                    "Russell 1000 (~1000 stocks). Walk-forward validation.",
    },
    "v5": {
        "title": "V5 — Weighted LightGBM Ensemble",
        "summary": "Three LightGBM sub-models with different hyperparameters, combined "
                   "via weighted average. Aims for more robust predictions than a single model.",
        "architecture": "Weighted ensemble of 3 LightGBM models with optimized weights. "
                        "Final prediction = weighted average of sub-model predictions.",
        "features": "Same 84+46=130 features as V6 (62 stock technicals + 22 macro + "
                    "46 cross-sectional rank features). Uses v6 feature function (superset).",
        "portfolio": "Equal-weight top 20 stocks. Rebalance every 5 trading days.",
        "risk": "No intraday monitoring. No market regime filter.",
        "training": "Trained on ~4 years of daily data. Each sub-model uses different "
                    "regularization (num_leaves, learning rate, min_child_samples) to "
                    "promote diversity.",
    },
    "v6": {
        "title": "V6 — Stacked Ensemble with Regime Filter",
        "summary": "Five diverse base models (3x LightGBM variants, XGBoost, CatBoost) "
                   "combined via Ridge regression meta-learner. Adds market regime "
                   "awareness and conviction-weighted position sizing.",
        "architecture": "Stacked ensemble: 5 base models (LightGBM main, LightGBM "
                        "regularized, LightGBM DART, XGBoost, CatBoost) → Ridge "
                        "meta-learner. Two-level prediction pipeline.",
        "features": "130 features: 62 stock technicals + 22 macro + 46 cross-sectional "
                    "rank features (cs_*). CS features rank each stock vs all peers at "
                    "each time point — captures relative momentum, value, volume.",
        "portfolio": "Conviction-weighted top 20: allocation proportional to prediction "
                     "strength, capped at 2x equal weight. Higher-conviction picks get "
                     "more capital.",
        "risk": "Market regime filter (composite score from VIX level, SPY trend, "
                "SPY momentum, HYG credit spread) → exposure multiplier 0.4–1.0. "
                "Reduces position sizes in hostile regimes. No intraday monitoring.",
        "training": "Trained on ~4 years of daily data. Base models trained independently, "
                    "meta-learner trained on out-of-fold predictions to avoid overfitting.",
    },
    "v7": {
        "title": "V7 — V6 Ensemble + Intraday Risk Management",
        "summary": "Same stacked ensemble as V6, but adds real-time portfolio monitoring "
                   "with automatic stop-loss execution. Scans positions every minute "
                   "during market hours.",
        "architecture": "Same as V6 (5 base models + Ridge meta-learner). Adds a "
                        "real-time cut-loss scanner running every 60 seconds during "
                        "market hours (9:30 AM – 4:00 PM ET).",
        "features": "Same 130 features as V6. Identical prediction pipeline.",
        "portfolio": "Conviction-weighted top 20 (same as V6). Positions actively "
                     "monitored and can be liquidated intraday if stops are hit.",
        "risk": "Three-layer risk management:\n"
                "  1. Hard stop: auto-sell if position drops 8% from entry price\n"
                "  2. Trailing stop: auto-sell if position drops 5% from its peak since entry\n"
                "  3. Portfolio stop: liquidate ALL positions if daily portfolio loss exceeds 3%\n"
                "Plus V6's market regime filter for position sizing.",
        "training": "Uses same V6 model file — no retraining needed. Risk management "
                    "is purely rule-based on live price data.",
    },
    "v8": {
        "title": "V8 — Sector-Neutral Stacked Ensemble",
        "summary": "Same stacked ensemble as V6, but forces sector diversification. "
                   "Max 3 stocks from any single GICS sector in the portfolio. "
                   "Prevents concentration risk (e.g. V6's semis overweight).",
        "architecture": "Same as V6 (5 base models + Ridge meta-learner). Adds sector "
                        "classification layer and constrained portfolio construction. "
                        "Also includes 3 sector-relative features.",
        "features": "133 features: 130 V6 features + 3 sector-relative features "
                    "(sector_rel_ret_5, sector_rel_ret_20, sector_rel_vol_20). "
                    "Sector-relative features measure stock performance vs its "
                    "sector ETF (XLK, XLF, XLV, etc.).",
        "portfolio": "Sector-constrained conviction-weighted top 20: same conviction "
                     "sizing as V6, but capped at 3 stocks per GICS sector. Skips "
                     "lower-ranked stocks from overrepresented sectors and fills "
                     "with next-best from underrepresented sectors.",
        "risk": "Market regime filter (same as V6): composite score from VIX, "
                "SPY trend, SPY momentum, HYG credit → exposure multiplier 0.4-1.0. "
                "Sector diversification itself is a risk management layer — limits "
                "blow-up from any single sector drawdown.",
        "training": "Trained on same data as V6. Sector map cached from yfinance. "
                    "Walk-forward validation with sector-neutral portfolio evaluation "
                    "at each fold (HHI concentration metric tracked).",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC MODEL CONFIGURATION (dashboard-managed)
# ═══════════════════════════════════════════════════════════════════════════

# All known models and their static properties
MODEL_REGISTRY = {
    "v4": {"feature_version": "v4", "model_dir": "v4", "fallback_model": "ml_v4_model.pkl"},
    "v5": {"feature_version": "v5", "model_dir": "v5", "fallback_model": None},
    "v6": {"feature_version": "v6", "model_dir": "v6", "fallback_model": None},
    "v7": {"feature_version": "v6", "model_dir": "v7", "fallback_model": None},  # same model as v6, with cut-loss
    "v8": {"feature_version": "v8", "model_dir": "v8", "fallback_model": None},  # sector-neutral v6 ensemble
}

CONFIG_PATH = DATA_DIR / "model_config.json"


def _default_config() -> dict:
    """Generate default config with 3 slots."""
    return {
        "slots": [
            {
                "slot_id": 1,
                "model": "v4",
                "enabled": True,
                "alpaca_key": "",
                "alpaca_secret": "",
                "enable_cutloss": False,
                "cutloss_hard_stop": -8.0,
                "cutloss_trailing_stop": -5.0,
                "cutloss_portfolio_stop": -3.0,
            },
            {
                "slot_id": 2,
                "model": "v5",
                "enabled": True,
                "alpaca_key": "",
                "alpaca_secret": "",
                "enable_cutloss": False,
                "cutloss_hard_stop": -8.0,
                "cutloss_trailing_stop": -5.0,
                "cutloss_portfolio_stop": -3.0,
            },
            {
                "slot_id": 3,
                "model": "v6",
                "enabled": True,
                "alpaca_key": "",
                "alpaca_secret": "",
                "enable_cutloss": False,
                "cutloss_hard_stop": -8.0,
                "cutloss_trailing_stop": -5.0,
                "cutloss_portfolio_stop": -3.0,
            },
        ],
        "updated_at": None,
    }


def load_model_config() -> dict:
    """Load model config from persistent storage."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return _default_config()


def save_model_config(config: dict):
    """Save model config to persistent storage."""
    config["updated_at"] = datetime.now().isoformat()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _resolve_model_path(model_name: str) -> Path:
    """Find model.pkl for a given model name."""
    reg = MODEL_REGISTRY.get(model_name, {})
    model_dir = reg.get("model_dir", model_name)

    # Primary: model/<dir>/model.pkl
    path = BASE_DIR / "model" / model_dir / "model.pkl"
    if path.exists():
        return path

    # Fallback (e.g. ml_v4_model.pkl)
    fallback = reg.get("fallback_model")
    if fallback:
        alt = BASE_DIR / "model" / fallback
        if alt.exists():
            return alt

    return path  # return primary even if missing (caller checks .exists())


def get_active_models() -> list[ModelConfig]:
    """Build active models from config file + env var fallback.

    Priority: config file slots > env vars.
    Only slots with valid API keys and existing model files are activated.
    """
    config = load_model_config()
    models = []
    config_has_keys = False

    # -- Try config file first --
    for slot in config.get("slots", []):
        if not slot.get("enabled", True):
            continue

        model_name = slot.get("model", "")
        if model_name not in MODEL_REGISTRY:
            continue

        key = slot.get("alpaca_key", "")
        secret = slot.get("alpaca_secret", "")

        if key and secret:
            config_has_keys = True

        model_path = _resolve_model_path(model_name)
        reg = MODEL_REGISTRY[model_name]

        print(f"[MODEL DETECT] slot {slot.get('slot_id')}: model={model_name}, "
              f"key={'YES' if key else 'NO'}, secret={'YES' if secret else 'NO'}, "
              f"model_file={model_path} exists={model_path.exists()}", flush=True)

        if key and secret and model_path.exists():
            models.append(ModelConfig(
                name=model_name,
                model_path=model_path,
                feature_version=reg["feature_version"],
                alpaca_key=key,
                alpaca_secret=secret,
                enable_cutloss=slot.get("enable_cutloss", False),
                cutloss_hard_stop=slot.get("cutloss_hard_stop", -8.0),
                cutloss_trailing_stop=slot.get("cutloss_trailing_stop", -5.0),
                cutloss_portfolio_stop=slot.get("cutloss_portfolio_stop", -3.0),
            ))

    # -- Fallback to env vars if config has no keys --
    if not config_has_keys:
        print("[MODEL DETECT] No keys in config file, falling back to env vars", flush=True)
        env_models = [
            ("v4", "MODEL_V4_ALPACA_KEY", "MODEL_V4_ALPACA_SECRET",
             "ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
            ("v5", "MODEL_V5_ALPACA_KEY", "MODEL_V5_ALPACA_SECRET", None, None),
            ("v6", "MODEL_V6_ALPACA_KEY", "MODEL_V6_ALPACA_SECRET", None, None),
        ]
        for name, key_env, secret_env, fallback_key, fallback_secret in env_models:
            key = os.environ.get(key_env, "")
            secret = os.environ.get(secret_env, "")
            if not key and fallback_key:
                key = os.environ.get(fallback_key, "")
            if not secret and fallback_secret:
                secret = os.environ.get(fallback_secret, "")

            model_path = _resolve_model_path(name)
            reg = MODEL_REGISTRY[name]

            print(f"[MODEL DETECT] env {name}: key={'YES' if key else 'NO'}, "
                  f"secret={'YES' if secret else 'NO'}, "
                  f"model={model_path} exists={model_path.exists()}", flush=True)

            if key and secret and model_path.exists():
                models.append(ModelConfig(
                    name=name,
                    model_path=model_path,
                    feature_version=reg["feature_version"],
                    alpaca_key=key,
                    alpaca_secret=secret,
                ))

    # Debug: list model files
    model_dir = BASE_DIR / "model"
    if model_dir.exists():
        import subprocess
        try:
            result = subprocess.run(["find", str(model_dir), "-name", "*.pkl"],
                                    capture_output=True, text=True, timeout=5)
            print(f"[MODEL DETECT] .pkl files found: {result.stdout.strip()}", flush=True)
        except Exception:
            pass

    print(f"[MODEL DETECT] Active: {[m.name for m in models]}", flush=True)
    return models


def test_alpaca_key(key: str, secret: str,
                    base_url: str = "https://paper-api.alpaca.markets") -> dict:
    """Test an Alpaca API key pair. Returns account info or error."""
    import requests
    try:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        resp = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        if resp.status_code == 200:
            acct = resp.json()
            return {
                "ok": True,
                "status": acct.get("status", "unknown"),
                "equity": float(acct.get("equity", 0)),
                "cash": float(acct.get("cash", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
                "currency": acct.get("currency", "USD"),
                "account_number": acct.get("account_number", "?"),
            }
        else:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# TRADE JOURNAL — structured, queryable trade logging
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """A single trade event — one per order submitted."""
    # Identity
    trade_id: str                   # unique: {model}_{timestamp}_{symbol}_{side}
    run_id: str                     # links all trades from the same rebalance
    model: str                      # "v4", "v5"
    timestamp: str                  # ISO-8601 UTC when order was submitted

    # Order details
    symbol: str
    side: str                       # "buy" | "sell"
    action: str                     # "new_position" | "exit_position" | "rebalance_up" | "rebalance_down"
    order_type: str                 # "market"
    time_in_force: str              # "day"

    # Amounts
    notional_usd: float             # dollar amount of the order
    shares: Optional[float] = None  # filled qty (updated post-fill)
    fill_price: Optional[float] = None  # avg fill price (updated post-fill)

    # Model context — why this trade was made
    predicted_return_pct: Optional[float] = None  # model's predicted 5d return %
    rank: Optional[int] = None                    # rank in prediction list (1 = best)
    target_weight_usd: Optional[float] = None     # target dollar allocation

    # Position context (for exits / rebalances)
    entry_price: Optional[float] = None           # avg entry price before this trade
    current_price: Optional[float] = None         # price at time of trade decision
    unrealized_pnl_usd: Optional[float] = None    # P&L at time of exit
    unrealized_pnl_pct: Optional[float] = None    # P&L % at time of exit
    holding_period_days: Optional[int] = None      # approx days held (for exits)
    position_value_before: Optional[float] = None  # position $ value before trade

    # Alpaca response
    order_id: Optional[str] = None          # Alpaca order ID
    order_status: str = "pending"           # "submitted" | "filled" | "failed"
    error_message: Optional[str] = None     # if order failed

    # Portfolio context
    portfolio_value: Optional[float] = None     # total portfolio $ at rebalance
    cash_before: Optional[float] = None         # cash $ before this order
    total_positions: Optional[int] = None       # # positions after rebalance target
    rebalance_turnover_pct: Optional[float] = None  # overall turnover this cycle


class TradeJournal:
    """Persistent trade log — appends to JSON Lines + CSV files per model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        TRADE_DIR.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = TRADE_DIR / f"trades_{model_name}.jsonl"
        self.csv_path = TRADE_DIR / f"trades_{model_name}.csv"

    def log_trade(self, record: TradeRecord):
        """Append a trade record to both JSONL and CSV."""
        data = asdict(record)

        # Append to JSONL (one JSON object per line — easy to parse)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

        # Append to CSV (for spreadsheet analysis)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(data)

    def get_trades(self, symbol: str = None, since: str = None) -> list[dict]:
        """Read trades back from the journal (for analysis endpoints)."""
        trades = []
        if not self.jsonl_path.exists():
            return trades
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if symbol and rec.get("symbol") != symbol:
                    continue
                if since and rec.get("timestamp", "") < since:
                    continue
                trades.append(rec)
        return trades


# ═══════════════════════════════════════════════════════════════════════════
# RUN REPORT (enhanced with trade journal integration)
# ═══════════════════════════════════════════════════════════════════════════

class RunReport:
    """Collects all run data for a final summary."""

    def __init__(self):
        self.start_time = datetime.now()
        self.steps = {}
        self.warnings = []
        self.errors = []
        self.timings = {}
        self.data = {}

    def start_step(self, name: str):
        self.steps[name] = {"status": "running", "start": time.time()}

    def end_step(self, name: str, status: str = "ok"):
        if name in self.steps:
            elapsed = time.time() - self.steps[name]["start"]
            self.steps[name]["status"] = status
            self.steps[name]["elapsed"] = elapsed
            self.timings[name] = elapsed

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)

    def set(self, key: str, value):
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def format_summary(self) -> str:
        total_time = (datetime.now() - self.start_time).total_seconds()
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  RUN SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  Date:         {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total time:   {total_time:.1f}s")
        lines.append("")

        # Step timings
        lines.append("  STEPS:")
        for name, info in self.steps.items():
            elapsed = info.get("elapsed", 0)
            status = info["status"]
            icon = "OK" if status == "ok" else "SKIP" if status == "skipped" else "FAIL"
            lines.append(f"    [{icon:4s}] {name:30s} {elapsed:6.1f}s")
        lines.append("")

        # Data quality
        if "universe_size" in self.data:
            lines.append("  DATA:")
            lines.append(f"    Universe requested:   {self.data.get('universe_size', '?')} symbols")
            lines.append(f"    Stocks downloaded:    {self.data.get('stocks_downloaded', '?')} symbols")
            lines.append(f"    Stocks failed:        {self.data.get('stocks_failed', '?')} symbols")
            lines.append(f"    Macro downloaded:     {self.data.get('macro_downloaded', '?')}/{len(MACRO_TICKERS)} tickers")
            lines.append(f"    Macro missing:        {', '.join(self.data.get('macro_missing', [])) or 'none'}")
            lines.append(f"    Feature columns:      {self.data.get('n_features', '?')}")
            lines.append(f"    Latest data date:     {self.data.get('latest_data_date', '?')}")
            lines.append("")

        # Predictions
        if "n_predictions" in self.data:
            lines.append("  PREDICTIONS:")
            lines.append(f"    Stocks predicted:     {self.data.get('n_predictions', '?')}")
            lines.append(f"    Prediction failures:  {self.data.get('prediction_failures', '?')}")
            pred_range = self.data.get("prediction_range", {})
            if pred_range:
                lines.append(f"    Pred range:           {pred_range.get('min', 0):.2f}% to {pred_range.get('max', 0):.2f}%")
                lines.append(f"    Pred mean:            {pred_range.get('mean', 0):.2f}%")
                lines.append(f"    Pred std:             {pred_range.get('std', 0):.2f}%")
            lines.append("")

        # Portfolio
        if "target_portfolio" in self.data:
            lines.append("  TARGET PORTFOLIO (top 20):")
            for i, (sym, pred) in enumerate(self.data["target_portfolio"]):
                lines.append(f"    {i+1:2d}. {sym:6s}  pred={pred:+6.2f}%")
            lines.append("")

        # Rebalance
        if "rebalance" in self.data:
            rb = self.data["rebalance"]
            lines.append("  REBALANCE:")
            lines.append(f"    Mode:                 {'DRY RUN' if rb.get('dry_run') else 'LIVE'}")
            lines.append(f"    Portfolio value:       ${rb.get('portfolio_value', 0):,.2f}")
            lines.append(f"    Cash available:        ${rb.get('cash', 0):,.2f}")
            lines.append(f"    Target weight/stock:   ${rb.get('target_weight', 0):,.2f}")
            lines.append(f"    Positions before:      {rb.get('positions_before', '?')}")
            lines.append(f"    Sells (exit):          {rb.get('n_sells', 0)}")
            lines.append(f"    Buys (new):            {rb.get('n_buys', 0)}")
            lines.append(f"    Rebalanced (adjust):   {rb.get('n_rebalanced', 0)}")
            lines.append(f"    Held (no change):      {rb.get('n_held', 0)}")
            lines.append(f"    Turnover:              {rb.get('turnover', 0):.0%}")
            if not rb.get("dry_run"):
                lines.append(f"    Orders submitted:      {rb.get('executed', 0)}")
                lines.append(f"    Orders failed:         {rb.get('failed', 0)}")

            if rb.get("sells_detail"):
                lines.append(f"    Sold:   {', '.join(rb['sells_detail'])}")
            if rb.get("buys_detail"):
                lines.append(f"    Bought: {', '.join(rb['buys_detail'])}")
            lines.append("")

        # Trade journal summary
        if "trade_log_summary" in self.data:
            tls = self.data["trade_log_summary"]
            lines.append("  TRADE JOURNAL:")
            lines.append(f"    Trades logged:         {tls.get('count', 0)}")
            lines.append(f"    Total notional:        ${tls.get('total_notional', 0):,.2f}")
            lines.append(f"    Buy notional:          ${tls.get('buy_notional', 0):,.2f}")
            lines.append(f"    Sell notional:         ${tls.get('sell_notional', 0):,.2f}")
            lines.append(f"    Journal file:          {tls.get('file', '?')}")
            lines.append("")

        # Warnings / Errors
        if self.warnings:
            lines.append(f"  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:20]:
                lines.append(f"    - {w}")
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings) - 20} more")
            lines.append("")

        if self.errors:
            lines.append(f"  ERRORS ({len(self.errors)}):")
            for e in self.errors[:20]:
                lines.append(f"    - {e}")
            lines.append("")

        status = "SUCCESS" if not self.errors else "COMPLETED WITH ERRORS"
        lines.append(f"  STATUS: {status}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(model_name: str = "main"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    for h in handlers:
        h.setFormatter(fmt)

    logger = logging.getLogger(f"pipeline.{model_name}")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplication on re-run
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)

    return logger, log_file


# ═══════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def _wiki_read_html(url: str, logger) -> list:
    """Read HTML tables from Wikipedia with proper User-Agent."""
    import requests as _req
    import io
    headers = {
        "User-Agent": "MLTradingBot/1.0 (educational paper trading project)"
    }
    resp = _req.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


# Symbol cache path — persists on Railway volume
SYMBOL_CACHE_PATH = DATA_DIR / "universe_cache.json"


def _load_symbol_cache(logger) -> list[str]:
    """Load cached symbol list from last successful scrape."""
    if SYMBOL_CACHE_PATH.exists():
        try:
            cache = json.loads(SYMBOL_CACHE_PATH.read_text())
            syms = cache.get("symbols", [])
            cached_at = cache.get("cached_at", "unknown")
            logger.info(f"  Loaded {len(syms)} cached symbols (from {cached_at})")
            return syms
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}")
    return []


def _save_symbol_cache(symbols: list[str], logger):
    """Save symbol list to cache for fallback."""
    try:
        SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYMBOL_CACHE_PATH.write_text(json.dumps({
            "symbols": symbols,
            "cached_at": datetime.now().isoformat(),
            "count": len(symbols),
        }, indent=2))
        logger.info(f"  Saved {len(symbols)} symbols to cache")
    except Exception as e:
        logger.warning(f"  Cache save failed: {e}")


def get_tradeable_symbols(logger, report: RunReport) -> list[str]:
    """Get S&P 500 + Nasdaq 100 + Russell 1000 symbols from Wikipedia.

    Uses proper User-Agent to avoid 403 blocks. Falls back to cached
    symbol list if all scrapes fail.
    """
    report.start_step("get_universe")
    sp500, ndx_syms, russell_syms = [], [], []

    try:
        tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", logger
        )
        sp500 = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"  S&P 500: {len(sp500)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  S&P 500 scrape failed: {e}")
        report.add_warning(f"S&P 500 scrape failed: {e}")

    try:
        ndx = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100", logger
        )
        for table in ndx:
            if "Ticker" in table.columns:
                ndx_syms = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                break
            elif "Symbol" in table.columns:
                ndx_syms = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                break
        logger.info(f"  Nasdaq 100: {len(ndx_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Nasdaq 100 scrape failed: {e}")
        report.add_warning(f"Nasdaq 100 scrape failed: {e}")

    # Russell 1000 — adds ~400-500 mid-cap stocks not in S&P 500
    try:
        r1k_tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Russell_1000_Index", logger
        )
        for table in r1k_tables:
            if "Ticker" in table.columns:
                russell_syms = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                break
            elif "Symbol" in table.columns:
                russell_syms = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                break
        logger.info(f"  Russell 1000: {len(russell_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Russell 1000 scrape failed (non-critical): {e}")
        report.add_warning(f"Russell 1000 scrape failed: {e}")

    all_syms = sorted(set(sp500 + ndx_syms + russell_syms) - EXCLUDED_SYMBOLS)

    # If scraping failed completely, fall back to cache
    if not all_syms:
        logger.warning("  All scrapes failed — falling back to cached symbol list")
        all_syms = _load_symbol_cache(logger)
    else:
        # Save successful scrape to cache
        _save_symbol_cache(all_syms, logger)

    logger.info(f"  Total universe: {len(all_syms)} unique symbols "
                f"(excluded {len(EXCLUDED_SYMBOLS)} blacklisted)")
    report.set("universe_size", len(all_syms))
    report.end_step("get_universe")
    return all_syms


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance column names (handles MultiIndex from newer versions)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def download_bars(symbols: list[str], days: int, logger,
                  report: RunReport) -> dict[str, pd.DataFrame]:
    """Download recent daily bars from Yahoo Finance."""
    report.start_step("download_stocks")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    logger.info(f"  Date range: {start.date()} -> {end.date()}")
    logger.info(f"  Symbols to download: {len(symbols)}")

    data = {}
    failed_syms = []
    batch_size = 50
    n_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(symbols), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch = symbols[batch_idx:batch_idx + batch_size]
        tickers_str = " ".join(batch)
        batch_start = time.time()

        try:
            df = yf.download(
                tickers_str, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker", auto_adjust=True, progress=False,
                threads=True,
            )
            batch_ok = 0
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
                        batch_ok += 1
                    else:
                        failed_syms.append(sym)
                except Exception:
                    failed_syms.append(sym)

            batch_time = time.time() - batch_start
            logger.info(f"  Batch {batch_num}/{n_batches}: "
                        f"{batch_ok}/{len(batch)} ok ({batch_time:.1f}s)")

        except Exception as e:
            logger.warning(f"  Batch {batch_num}/{n_batches} FAILED: {e}")
            report.add_warning(f"Stock batch {batch_num} failed: {e}")
            failed_syms.extend(batch)

        time.sleep(0.3)

    logger.info(f"  Downloaded: {len(data)} symbols, "
                f"failed: {len(failed_syms)} symbols")

    if data:
        sample_sym = next(iter(data))
        sample_df = data[sample_sym]
        logger.info(f"  Sample ({sample_sym}): {len(sample_df)} days, "
                    f"{sample_df.index[0].date()} -> {sample_df.index[-1].date()}")
        report.set("latest_data_date", str(sample_df.index[-1].date()))

    report.set("stocks_downloaded", len(data))
    report.set("stocks_failed", len(failed_syms))
    if failed_syms and len(failed_syms) <= 20:
        report.add_warning(f"Failed stocks: {', '.join(failed_syms)}")
    report.end_step("download_stocks")
    return data


def download_macro(days: int, logger, report: RunReport) -> dict[str, pd.DataFrame]:
    """Download macro tickers."""
    report.start_step("download_macro")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    macro = {}
    missing = []

    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            df = _normalize_columns(df)
            name = MACRO_RENAME.get(ticker, ticker)
            df_clean = df.dropna(subset=["close"])
            if len(df_clean) > 0:
                macro[name] = df_clean
                logger.info(f"  {name:6s}: {len(df_clean)} days, "
                            f"latest close: {df_clean['close'].iloc[-1]:.2f}")
            else:
                missing.append(ticker)
                logger.warning(f"  {ticker}: no data returned")
        except Exception as e:
            missing.append(ticker)
            logger.warning(f"  {ticker}: FAILED - {e}")
            report.add_warning(f"Macro {ticker} failed: {e}")
        time.sleep(0.1)

    logger.info(f"  Macro: {len(macro)}/{len(MACRO_TICKERS)} downloaded")
    if missing:
        logger.warning(f"  Missing: {', '.join(missing)}")

    report.set("macro_downloaded", len(macro))
    report.set("macro_missing", missing)
    report.end_step("download_macro")
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
# V6 FEATURES — extends v5 with drawdown, momentum, efficiency
# ═══════════════════════════════════════════════════════════════════════════

def compute_stock_features_v6(df: pd.DataFrame) -> pd.DataFrame:
    """V6 features = v4 base features + new v6 signals."""
    # Start with v4 features as base
    feats_df = compute_stock_features(df)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    lo = df['low'].astype(float)
    v = df['volume'].astype(float)
    ret = c.pct_change()

    # Mean-reversion signals
    for p in [20, 60]:
        trend = c.rolling(p).mean().pct_change(5) * 100
        actual = c.pct_change(5) * 100
        feats_df[f'resid_mom_{p}'] = (actual - trend).astype(np.float32)

    feats_df['reversal_20_5'] = ((c.pct_change(5) * 100) - (c.pct_change(20) * 100 * 0.25)).astype(np.float32)

    for p in [20, 60, 120]:
        ema = c.ewm(span=p, adjust=False).mean()
        feats_df[f'trend_dev_{p}'] = ((c - ema) / (ema + 1e-8) * 100).astype(np.float32)

    for p in [20, 60]:
        rm = ret.rolling(p)
        feats_df[f'ret_zscore_{p}'] = ((ret - rm.mean()) / (rm.std() + 1e-8)).astype(np.float32)

    # SMA crossovers
    sma_20 = c.rolling(20).mean()
    sma_50 = c.rolling(50).mean()
    sma_200 = c.rolling(200).mean()
    feats_df['sma_20_50_cross'] = ((sma_20 - sma_50) / (sma_50 + 1e-8) * 100).astype(np.float32)
    feats_df['sma_50_200_cross'] = ((sma_50 - sma_200) / (sma_200 + 1e-8) * 100).astype(np.float32)

    # Volatility ratios
    feats_df['vol_ratio_5_20'] = (ret.rolling(5).std() / (ret.rolling(20).std() + 1e-8)).astype(np.float32)
    feats_df['vol_ratio_10_60'] = (ret.rolling(10).std() / (ret.rolling(60).std() + 1e-8)).astype(np.float32)

    # Dollar volume ratio (v5 feature)
    dollar_vol = c * v
    dollar_vol_sma20 = dollar_vol.rolling(20).mean()
    feats_df['dollar_vol_ratio'] = (dollar_vol / (dollar_vol_sma20 + 1e-8)).astype(np.float32)

    # Volume-price divergence
    v_sma20 = v.rolling(20).mean()
    feats_df['vp_divergence'] = (v / (v_sma20 + 1e-8) - (1 + ret.rolling(5).mean() * 10)).astype(np.float32)

    # Accumulation/Distribution
    clv = ((c - lo) - (h - c)) / (h - lo + 1e-8)
    ad = (clv * v).cumsum()
    ad_sma = ad.rolling(20).mean()
    feats_df['ad_trend'] = ((ad - ad_sma) / (ad_sma.abs() + 1e-8)).astype(np.float32)

    # Gap feature
    if 'open' in df.columns:
        feats_df['gap'] = ((df['open'].astype(float) / c.shift(1) - 1) * 100).astype(np.float32)

    # Drawdown features
    rolling_max_60 = c.rolling(60).max()
    rolling_max_120 = c.rolling(120).max()
    feats_df['drawdown_60'] = ((c - rolling_max_60) / (rolling_max_60 + 1e-8) * 100).astype(np.float32)
    feats_df['drawdown_120'] = ((c - rolling_max_120) / (rolling_max_120 + 1e-8) * 100).astype(np.float32)

    dd_60 = (c - rolling_max_60) / (rolling_max_60 + 1e-8)
    feats_df['recovery_speed_60'] = (dd_60 - dd_60.shift(5)).astype(np.float32)

    # Consecutive up/down days
    feats_df['consec_up_5'] = (ret > 0).astype(float).rolling(5).sum().astype(np.float32)
    feats_df['consec_down_5'] = (ret < 0).astype(float).rolling(5).sum().astype(np.float32)

    # Move vs ATR
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats_df['move_vs_atr'] = (c.diff(5).abs() / (atr_14 * np.sqrt(5) + 1e-8)).astype(np.float32)

    # Volume acceleration
    feats_df['vol_accel'] = ((v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)) -
                             (v.rolling(20).mean() / (v.rolling(60).mean() + 1e-8))).astype(np.float32)

    # Price efficiency
    for p in [5, 20]:
        directional = c.diff(p).abs()
        total_path = ret.abs().rolling(p).sum() * c + 1e-8
        feats_df[f'efficiency_{p}'] = (directional / total_path).astype(np.float32)

    return feats_df


# ═══════════════════════════════════════════════════════════════════════════
# V6 REGIME DETECTION + CONVICTION SIZING (for live trading)
# ═══════════════════════════════════════════════════════════════════════════

def compute_live_regime_score(macro_data: dict[str, pd.DataFrame]) -> float:
    """Compute current market regime score for live portfolio construction.

    Returns float in [-1, 1]: -1 = hostile, +1 = favorable.
    """
    signals = []

    # VIX signal
    if 'VIX' in macro_data:
        vix = macro_data['VIX']['close']
        if len(vix) >= 120:
            vix_pct = vix.rank(pct=True).iloc[-1]
            signals.append(('vix', 0.30, 1 - 2 * vix_pct))

    # SPY trend signal
    spy_data = macro_data.get('SPY')
    if spy_data is not None:
        spy_close = spy_data['close']
        if len(spy_close) >= 200:
            sma_200 = spy_close.rolling(200).mean().iloc[-1]
            dist_pct = (spy_close.iloc[-1] - sma_200) / (sma_200 + 1e-8) * 100
            signals.append(('trend', 0.30, max(-1, min(1, dist_pct / 5))))

            # Momentum
            if len(spy_close) >= 20:
                ret_20 = (spy_close.iloc[-1] / spy_close.iloc[-20] - 1) * 100
                signals.append(('momentum', 0.20, max(-1, min(1, ret_20 / 5))))

    # Credit signal
    if 'HYG' in macro_data:
        hyg = macro_data['HYG']['close']
        if len(hyg) >= 50:
            hyg_sma50 = hyg.rolling(50).mean().iloc[-1]
            hyg_dist = (hyg.iloc[-1] - hyg_sma50) / (hyg_sma50 + 1e-8) * 100
            signals.append(('credit', 0.20, max(-1, min(1, hyg_dist / 2))))

    if not signals:
        return 0.0  # neutral if no data

    total_weight = sum(w for _, w, _ in signals)
    score = sum(w * s for _, w, s in signals) / total_weight
    return max(-1.0, min(1.0, score))


def regime_to_exposure(regime_score: float) -> float:
    """Map regime score to portfolio exposure multiplier.

    -1.0 -> 0.4 (40% exposure)
     0.0 -> 0.85 (85% exposure)
    +0.3 -> 1.0 (full exposure)
    """
    if regime_score >= 0.3:
        return 1.0
    elif regime_score >= 0.0:
        return 0.85 + (regime_score / 0.3) * 0.15
    elif regime_score >= -0.3:
        return 0.7 + ((regime_score + 0.3) / 0.3) * 0.15
    else:
        return max(0.4, 0.7 + (regime_score + 0.3) / 0.7 * 0.3)


def conviction_weights(
    predictions: list[tuple[str, float]],
    top_n: int = 20,
    max_weight_multiple: float = 2.0,
    regime_exposure: float = 1.0,
) -> dict[str, float]:
    """Convert ranked predictions into conviction-weighted allocations.

    Returns {symbol: weight} where weights sum to regime_exposure.
    """
    top = predictions[:top_n]
    if not top:
        return {}

    preds = np.array([p for _, p in top])
    preds_shifted = preds - preds.min() + 0.01
    raw_weights = preds_shifted / preds_shifted.sum()

    equal_weight = 1.0 / top_n
    max_weight = equal_weight * max_weight_multiple
    capped = np.minimum(raw_weights, max_weight)
    capped = capped / capped.sum() * regime_exposure

    return {sym: round(float(w), 6) for (sym, _), w in zip(top, capped)}


def sector_neutral_weights(
    predictions: list[tuple[str, float]],
    sector_map: dict[str, str],
    top_n: int = 20,
    max_per_sector: int = 3,
    max_weight_multiple: float = 2.0,
    regime_exposure: float = 1.0,
) -> dict[str, float]:
    """Sector-constrained conviction-weighted portfolio (V8).

    Same as conviction_weights but caps at max_per_sector stocks from any
    single GICS sector. Skips lower-ranked stocks from overrepresented
    sectors and fills with next-best from underrepresented sectors.
    """
    if not predictions:
        return {}

    selected = []
    sector_counts = {}

    for sym, pred in predictions:
        if len(selected) >= top_n:
            break
        sector = sector_map.get(sym, "Unknown")
        count = sector_counts.get(sector, 0)
        if count < max_per_sector:
            selected.append((sym, pred))
            sector_counts[sector] = count + 1

    if not selected:
        return {}

    preds = np.array([p for _, p in selected])
    preds_shifted = preds - preds.min() + 0.01
    raw_weights = preds_shifted / preds_shifted.sum()

    equal_weight = 1.0 / len(selected)
    max_weight = equal_weight * max_weight_multiple
    capped = np.minimum(raw_weights, max_weight)
    capped = capped / capped.sum() * regime_exposure

    return {sym: round(float(w), 6) for (sym, _), w in zip(selected, capped)}


def _load_sector_map_for_pipeline() -> dict:
    """Load cached sector map for pipeline use. Returns empty dict if not found."""
    sector_cache = DATA_DIR / "sector_map.json"
    if sector_cache.exists():
        try:
            with open(sector_cache) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_feature_func(feature_version: str):
    """Return the appropriate feature computation function for a model version."""
    if feature_version in ("v5", "v6", "v8"):
        # v5 features are a subset of v6; v8 uses v6 base + sector-relative
        return compute_stock_features_v6
    else:
        return compute_stock_features


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def _add_cross_sectional_ranks(
    latest_rows: dict[str, pd.Series],
    feature_cols: list[str],
) -> dict[str, pd.Series]:
    """Add cross-sectional rank features (cs_*) across all stocks.

    For each base feature that has a cs_ counterpart in feature_cols,
    compute the percentile rank across all stocks for the latest time point.
    """
    # Find which cs_ features are needed
    cs_cols = [c for c in feature_cols if c.startswith("cs_")]
    if not cs_cols:
        return latest_rows

    # Map cs_ col -> base col (e.g. cs_ret_5d -> ret_5d)
    cs_to_base = {}
    for cs_col in cs_cols:
        base_col = cs_col[3:]  # strip "cs_"
        cs_to_base[cs_col] = base_col

    # Collect base feature values across all stocks
    syms = list(latest_rows.keys())
    base_cols_needed = set(cs_to_base.values())

    # Build cross-sectional DataFrame: rows=stocks, cols=base features
    cs_data = {}
    for base_col in base_cols_needed:
        vals = {}
        for sym in syms:
            row = latest_rows[sym]
            if base_col in row.index:
                vals[sym] = row[base_col]
        if vals:
            cs_data[base_col] = pd.Series(vals)

    # Compute percentile ranks
    cs_ranks = {}
    for base_col, series in cs_data.items():
        if len(series) > 1:
            cs_ranks[base_col] = series.rank(pct=True)
        else:
            cs_ranks[base_col] = series * 0 + 0.5  # neutral if only 1 stock

    # Add cs_ features to each stock's row
    result = {}
    for sym in syms:
        row = latest_rows[sym].copy()
        for cs_col, base_col in cs_to_base.items():
            if base_col in cs_ranks and sym in cs_ranks[base_col].index:
                row[cs_col] = np.float32(cs_ranks[base_col][sym])
            else:
                row[cs_col] = np.float32(0.5)  # neutral default
        result[sym] = row

    return result


def predict_rankings(
    stock_data: dict, macro_features: pd.DataFrame,
    model, feature_cols: list, logger, report: RunReport,
    feature_version: str = "v4",
) -> list[tuple[str, float]]:
    """Generate predictions for all stocks, return sorted rankings."""
    report.start_step("predict")
    predictions = {}
    failures = []
    feat_func = _get_feature_func(feature_version)

    # Check if cross-sectional features are needed
    has_cs_features = any(c.startswith("cs_") for c in feature_cols)

    # Step 1: Compute per-stock features and merge with macro
    latest_rows = {}
    for sym, df in stock_data.items():
        try:
            stock_feats = feat_func(df)

            # Merge with macro
            if not macro_features.empty:
                merged = stock_feats.join(macro_features, how="left")
                macro_cols = macro_features.columns
                merged[macro_cols] = merged[macro_cols].ffill()
            else:
                merged = stock_feats

            # Get latest row
            latest_rows[sym] = merged.iloc[-1]

        except Exception as e:
            failures.append((sym, str(e)))

    logger.info(f"  Features computed: {len(latest_rows)} ok, "
                f"{len(failures)} failed")

    # Step 2: Add cross-sectional ranks if needed
    if has_cs_features and latest_rows:
        logger.info(f"  Adding cross-sectional ranks across "
                    f"{len(latest_rows)} stocks...")
        latest_rows = _add_cross_sectional_ranks(latest_rows, feature_cols)

    # Step 3: Generate predictions
    for sym, row_series in latest_rows.items():
        try:
            row = row_series.to_frame().T
            avail_cols = [c for c in feature_cols if c in row.columns]

            if len(avail_cols) < len(feature_cols) * 0.9:
                missing_count = len(feature_cols) - len(avail_cols)
                failures.append((sym, f"too many missing features "
                               f"({missing_count}/{len(feature_cols)})"))
                continue

            row = row[avail_cols]
            if row.isna().any(axis=1).iloc[0]:
                row = row.fillna(0)

            # Ensure column order matches training
            missing = [c for c in feature_cols if c not in row.columns]
            for mc in missing:
                row[mc] = 0
            row = row[feature_cols]

            pred = model.predict(row)[0]
            if np.isfinite(pred):
                predictions[sym] = float(pred)
            else:
                failures.append((sym, "non-finite prediction"))

        except Exception as e:
            failures.append((sym, str(e)))

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"  Predictions: {len(ranked)} ok, {len(failures)} failed")
    if failures:
        # Group failures by reason for summary
        from collections import Counter
        reason_counts = Counter(reason for _, reason in failures)
        for reason, count in reason_counts.most_common(5):
            logger.info(f"    Failure: {count}x — {reason}")
        if len(failures) <= 10:
            for sym, reason in failures:
                logger.info(f"    SKIP {sym}: {reason}")

    # Prediction distribution stats
    if predictions:
        preds_arr = np.array(list(predictions.values()))
        report.set("prediction_range", {
            "min": float(np.min(preds_arr)),
            "max": float(np.max(preds_arr)),
            "mean": float(np.mean(preds_arr)),
            "std": float(np.std(preds_arr)),
            "median": float(np.median(preds_arr)),
        })
        logger.info(f"  Prediction range: {np.min(preds_arr):.2f}% to "
                    f"{np.max(preds_arr):.2f}% "
                    f"(mean: {np.mean(preds_arr):.2f}%, "
                    f"std: {np.std(preds_arr):.2f}%)")

    report.set("n_predictions", len(ranked))
    report.set("prediction_failures", len(failures))

    if ranked:
        logger.info(f"  Top 5:    {[(s, f'{p:+.2f}%') for s, p in ranked[:5]]}")
        logger.info(f"  Bottom 5: {[(s, f'{p:+.2f}%') for s, p in ranked[-5:]]}")

    report.end_step("predict")
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# ALPACA TRADING (with structured trade logging)
# ═══════════════════════════════════════════════════════════════════════════

def _make_alpaca_headers(mc: ModelConfig) -> dict:
    """Build Alpaca auth headers for a specific model's credentials."""
    return {
        "APCA-API-KEY-ID": mc.alpaca_key,
        "APCA-API-SECRET-KEY": mc.alpaca_secret,
        "Content-Type": "application/json",
    }


def alpaca_request(method: str, endpoint: str, mc: ModelConfig,
                   data=None, logger=None):
    """Make an Alpaca API request with per-model credentials."""
    import requests
    url = f"{mc.alpaca_base_url}/{endpoint}"
    headers = _make_alpaca_headers(mc)

    if logger:
        logger.info(f"    API [{mc.name}]: {method} {endpoint}")

    if method == "GET":
        resp = requests.get(url, headers=headers)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unknown method: {method}")

    if resp.status_code not in (200, 204, 207):
        err_msg = f"Alpaca {method} {endpoint}: {resp.status_code} {resp.text}"
        if logger:
            logger.error(f"    {err_msg}")
        raise Exception(err_msg)

    if resp.status_code == 204:
        return {}
    return resp.json()


def get_account(mc: ModelConfig, logger, report: RunReport):
    """Get Alpaca account info for a model."""
    acct = alpaca_request("GET", "v2/account", mc, logger=logger)
    pv = float(acct["portfolio_value"])
    cash = float(acct["cash"])
    bp = float(acct["buying_power"])
    logger.info(f"  Account [{mc.name}]: portfolio=${pv:,.2f}, "
                f"cash=${cash:,.2f}, buying_power=${bp:,.2f}")
    report.data.setdefault("rebalance", {}).update({
        "portfolio_value": pv, "cash": cash, "buying_power": bp,
    })
    return acct


def get_positions(mc: ModelConfig, logger) -> dict[str, dict]:
    """Get current positions with details for a model."""
    positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
    pos_dict = {}
    total_value = 0
    total_pl = 0

    for p in positions:
        mv = float(p["market_value"])
        pl = float(p["unrealized_pl"])
        pos_dict[p["symbol"]] = {
            "qty": float(p["qty"]),
            "market_value": mv,
            "unrealized_pl": pl,
            "unrealized_pl_pct": float(p.get("unrealized_plpc", 0)) * 100,
            "avg_entry": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "side": p["side"],
        }
        total_value += mv
        total_pl += pl

    logger.info(f"  Positions [{mc.name}]: {len(pos_dict)} stocks, "
                f"value=${total_value:,.2f}, unrealized P&L=${total_pl:,.2f}")

    if pos_dict:
        for sym, info in sorted(pos_dict.items()):
            logger.info(f"    {sym:6s}: {info['qty']:>8.2f} shares @ "
                        f"${info['avg_entry']:>8.2f} -> ${info['current_price']:>8.2f}, "
                        f"val=${info['market_value']:>10,.2f}, "
                        f"P&L=${info['unrealized_pl']:>+8,.2f} "
                        f"({info['unrealized_pl_pct']:>+5.1f}%)")

    return pos_dict


def rebalance_portfolio(
    target_symbols: list[str],
    rankings: list[tuple[str, float]],
    mc: ModelConfig,
    journal: TradeJournal,
    logger,
    report: RunReport,
    dry_run: bool = False,
    target_weights: dict[str, float] = None,
):
    """Rebalance portfolio with full trade logging.

    If target_weights is provided (v6 conviction sizing), use those allocations.
    Otherwise falls back to equal-weight across target_symbols.
    """
    report.start_step("rebalance")
    rb_data = report.data.setdefault("rebalance", {})
    rb_data["dry_run"] = dry_run

    # Build prediction lookup: symbol -> (predicted_return, rank)
    pred_lookup = {}
    for rank_idx, (sym, pred) in enumerate(rankings):
        pred_lookup[sym] = (pred, rank_idx + 1)

    # Generate a unique run_id for this rebalance cycle
    run_id = f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    acct = get_account(mc, logger, report)
    portfolio_value = float(acct["portfolio_value"])
    cash_available = float(acct["cash"])
    current_positions = get_positions(mc, logger)
    rb_data["positions_before"] = len(current_positions)

    target_set = set(target_symbols)
    current_set = set(current_positions.keys())

    to_sell = sorted(current_set - target_set)
    to_buy = sorted(target_set - current_set)
    to_hold = sorted(current_set & target_set)

    # Per-symbol target dollar allocations
    if target_weights:
        # Conviction-weighted (v6): each symbol has a custom weight
        sym_allocations = {sym: portfolio_value * w for sym, w in target_weights.items()}
        avg_weight = portfolio_value / len(target_symbols)
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "conviction"
        logger.info(f"  Sizing: CONVICTION-WEIGHTED (exposure={sum(target_weights.values()):.0%})")
    else:
        # Equal-weight (v4/v5): every stock gets the same allocation
        avg_weight = portfolio_value / len(target_symbols)
        sym_allocations = {sym: avg_weight for sym in target_symbols}
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "equal"

    # Turnover calculation
    total_positions = max(len(target_set) + len(current_set), 1)
    turnover = (len(to_sell) + len(to_buy)) / total_positions

    orders = []
    n_rebalanced = 0
    n_held_unchanged = 0

    # -- 1. Sell positions not in target --
    logger.info(f"\n  SELLS - {len(to_sell)} positions to exit:")
    for sym in to_sell:
        pos = current_positions[sym]
        qty = pos["qty"]
        mv = pos["market_value"]
        pl = pos["unrealized_pl"]
        pl_pct = pos["unrealized_pl_pct"]
        entry = pos["avg_entry"]
        price = pos["current_price"]

        logger.info(f"    EXIT  {sym:6s}: {qty:>8.2f} shares, "
                    f"entry=${entry:.2f} -> now=${price:.2f}, "
                    f"val=${mv:,.2f}, P&L=${pl:+,.2f} ({pl_pct:+.1f}%)")

        if not dry_run:
            orders.append({
                "action": "sell", "symbol": sym, "qty": qty,
                "notional": mv, "side": "sell",
                "trade_action": "exit_position",
                "entry_price": entry, "current_price": price,
                "position_value_before": mv,
                "unrealized_pnl_usd": pl, "unrealized_pnl_pct": pl_pct,
            })

    # -- 2. Check held positions — rebalance if needed --
    logger.info(f"\n  HOLDS - {len(to_hold)} positions to check:")
    for sym in to_hold:
        pos = current_positions[sym]
        current_value = pos["market_value"]
        sym_target = sym_allocations.get(sym, avg_weight)
        diff = sym_target - current_value
        drift_pct = abs(diff) / (sym_target + 1e-8) * 100
        pred_ret, rank = pred_lookup.get(sym, (None, None))
        pred_str = f", pred={pred_ret:+.2f}% rank=#{rank}" if pred_ret is not None else ""

        if abs(diff) > sym_target * 0.1:
            if diff > 0:
                direction = "BUY more"
                trade_action = "rebalance_up"
            else:
                direction = "TRIM"
                trade_action = "rebalance_down"

            logger.info(f"    REBAL {sym:6s}: ${current_value:,.0f} -> ${sym_target:,.0f} "
                        f"(drift {drift_pct:.0f}%, {direction} ${abs(diff):,.0f}{pred_str})")
            n_rebalanced += 1

            if not dry_run:
                side = "buy" if diff > 0 else "sell"
                orders.append({
                    "action": f"{side}_notional", "symbol": sym,
                    "notional": abs(diff), "side": side,
                    "trade_action": trade_action,
                    "entry_price": pos["avg_entry"],
                    "current_price": pos["current_price"],
                    "position_value_before": current_value,
                    "predicted_return": pred_ret, "rank": rank,
                })
        else:
            logger.info(f"    HOLD  {sym:6s}: ${current_value:,.0f} "
                        f"(drift {drift_pct:.0f}% < 10%, no action{pred_str})")
            n_held_unchanged += 1

    # -- 3. Buy new positions --
    logger.info(f"\n  BUYS - {len(to_buy)} new positions:")
    for sym in to_buy:
        sym_target = sym_allocations.get(sym, avg_weight)
        pred_ret, rank = pred_lookup.get(sym, (None, None))
        pred_str = f"pred={pred_ret:+.2f}%, rank=#{rank}" if pred_ret is not None else ""
        logger.info(f"    NEW   {sym:6s}: ${sym_target:,.0f} ({pred_str})")

        if not dry_run:
            orders.append({
                "action": "buy_notional", "symbol": sym,
                "notional": sym_target, "side": "buy",
                "trade_action": "new_position",
                "predicted_return": pred_ret, "rank": rank,
            })

    rb_data.update({
        "n_sells": len(to_sell), "n_buys": len(to_buy),
        "n_rebalanced": n_rebalanced, "n_held": n_held_unchanged,
        "sells_detail": to_sell, "buys_detail": to_buy,
        "turnover": turnover,
    })

    logger.info(f"\n  Summary: {len(to_sell)} exits, {len(to_buy)} new buys, "
                f"{n_rebalanced} rebalanced, {n_held_unchanged} held, "
                f"turnover: {turnover:.0%}")

    if dry_run:
        logger.info("  MODE: DRY RUN - no orders sent, no trades logged")
        report.end_step("rebalance")
        return rb_data

    # -- Execute orders + log each trade --
    logger.info(f"\n  Executing {len(orders)} orders...")
    executed = 0
    failed = 0
    trade_count = 0
    total_notional = 0.0
    buy_notional = 0.0
    sell_notional = 0.0

    for order in orders:
        sym = order["symbol"]
        trade_ts = datetime.now(timezone.utc).isoformat()
        trade_id = f"{mc.name}_{trade_ts.replace(':', '').replace('-', '')}_{sym}_{order['side']}"

        # Build trade record
        trade = TradeRecord(
            trade_id=trade_id,
            run_id=run_id,
            model=mc.name,
            timestamp=trade_ts,
            symbol=sym,
            side=order["side"],
            action=order["trade_action"],
            order_type="market",
            time_in_force="day",
            notional_usd=round(order["notional"], 2),
            predicted_return_pct=order.get("predicted_return"),
            rank=order.get("rank"),
            target_weight_usd=round(sym_allocations.get(sym, avg_weight), 2),
            entry_price=order.get("entry_price"),
            current_price=order.get("current_price"),
            unrealized_pnl_usd=order.get("unrealized_pnl_usd"),
            unrealized_pnl_pct=order.get("unrealized_pnl_pct"),
            position_value_before=order.get("position_value_before"),
            portfolio_value=round(portfolio_value, 2),
            cash_before=round(cash_available, 2),
            total_positions=len(target_symbols),
            rebalance_turnover_pct=round(turnover * 100, 1),
        )

        try:
            if order["action"] == "sell":
                # Close entire position
                alpaca_request("DELETE", f"v2/positions/{sym}", mc, logger=logger)
                trade.order_status = "submitted"
                trade.shares = order["qty"]
                logger.info(f"    OK  EXIT  {sym:6s}: closed {order['qty']:.2f} shares, "
                            f"P&L=${order.get('unrealized_pnl_usd', 0):+,.2f} "
                            f"({order.get('unrealized_pnl_pct', 0):+.1f}%)")

            elif order["action"] in ("buy_notional", "sell_notional"):
                # Fractional notional order
                resp = alpaca_request("POST", "v2/orders", mc, {
                    "symbol": sym,
                    "notional": round(order["notional"], 2),
                    "side": order["side"],
                    "type": "market",
                    "time_in_force": "day",
                }, logger=logger)
                trade.order_id = resp.get("id")
                trade.order_status = resp.get("status", "submitted")
                logger.info(
                    f"    OK  {order['trade_action'].upper():16s} {sym:6s}: "
                    f"{order['side']} ${order['notional']:,.2f}, "
                    f"order_id={resp.get('id', '?')}, "
                    f"status={resp.get('status', '?')}"
                )

            executed += 1
            total_notional += order["notional"]
            if order["side"] == "buy":
                buy_notional += order["notional"]
            else:
                sell_notional += order["notional"]

        except Exception as e:
            trade.order_status = "failed"
            trade.error_message = str(e)
            logger.error(f"    FAIL {order['trade_action'].upper():16s} {sym:6s}: {e}")
            report.add_error(f"Order failed: {order['trade_action']} {sym} - {e}")
            failed += 1

        # Log trade to journal regardless of success/failure
        journal.log_trade(trade)
        trade_count += 1
        time.sleep(0.1)

    rb_data.update({"executed": executed, "failed": failed})
    report.set("trade_log_summary", {
        "count": trade_count,
        "total_notional": round(total_notional, 2),
        "buy_notional": round(buy_notional, 2),
        "sell_notional": round(sell_notional, 2),
        "file": str(journal.jsonl_path),
    })

    logger.info(f"\n  Orders done: {executed} submitted, {failed} failed")
    logger.info(f"  Trade journal: {trade_count} trades logged -> {journal.jsonl_path}")
    logger.info(f"  Notional: ${total_notional:,.2f} total "
                f"(${buy_notional:,.2f} buys, ${sell_notional:,.2f} sells)")

    report.end_step("rebalance")
    return rb_data


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT (per-model)
# ═══════════════════════════════════════════════════════════════════════════

def load_state(mc: ModelConfig) -> dict:
    """Load pipeline state from JSON for a specific model."""
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    if mc.state_path.exists():
        return json.loads(mc.state_path.read_text())
    return {"last_rebalance": None, "last_run": None, "run_count": 0, "history": []}


def save_state(state: dict, mc: ModelConfig):
    """Save pipeline state to JSON for a specific model."""
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    mc.state_path.write_text(json.dumps(state, indent=2, default=str))


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
# MAIN PIPELINE — runs per model
# ═══════════════════════════════════════════════════════════════════════════

def run_single_model(
    mc: ModelConfig,
    stock_data: dict,
    macro_features: pd.DataFrame,
    macro_data: dict = None,
    dry_run: bool = False,
    force: bool = False,
):
    """Execute the pipeline for one model."""
    logger, log_file = setup_logging(mc.name)
    report = RunReport()
    journal = TradeJournal(mc.name)

    logger.info("=" * 70)
    logger.info(f"  ML TRADING PIPELINE - Model: {mc.name.upper()}")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Config: H{HORIZON}_LongOnly{TOP_N}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
    logger.info(f"  Force rebalance: {force}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"  Trade journal: {journal.jsonl_path}")
    logger.info("=" * 70)

    # -- Step 1: Load model --
    logger.info("\n[1/5] LOADING MODEL")
    report.start_step("load_model")
    if not mc.model_path.exists():
        logger.error(f"Model not found: {mc.model_path}")
        report.add_error(f"Model not found: {mc.model_path}")
        logger.info(report.format_summary())
        return

    # Custom unpickler to resolve classes saved from training scripts
    # Models may be pickled with module="__main__" (direct script run) or
    # module="scripts.train_ml_v6" etc. (run via python -m). Map all to
    # the pipeline-local class definitions so deserialization works on Railway.
    class _PipelineUnpickler(pickle.Unpickler):
        _class_map = {
            "StackedEnsemble": StackedEnsemble,
            "EnsembleModel": EnsembleModel,
        }
        _known_training_modules = {
            "__main__",
            "scripts.train_ml_v4",
            "scripts.train_ml_v5",
            "scripts.train_ml_v6",
            "scripts.train_ml_v7",
            "scripts.train_ml_v8",
            "train_ml_v4",
            "train_ml_v5",
            "train_ml_v6",
            "train_ml_v7",
            "train_ml_v8",
        }

        def find_class(self, module, name):
            if name in self._class_map and module in self._known_training_modules:
                return self._class_map[name]
            return super().find_class(module, name)

    model_bundle = _PipelineUnpickler(open(mc.model_path, "rb")).load()
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    logger.info(f"  Model: {len(feature_cols)} features, "
                f"horizon={model_bundle.get('horizon', '?')}d, "
                f"task={model_bundle.get('task', '?')}")
    logger.info(f"  Trained: {model_bundle.get('saved_at', '?')}")
    report.set("n_features", len(feature_cols))
    report.end_step("load_model")

    # -- Step 2: Check state --
    state = load_state(mc)
    logger.info(f"\n  State: run #{state.get('run_count', 0) + 1}, "
                f"last rebalance: {state.get('last_rebalance', 'never')}")

    if not should_rebalance(state, force):
        last = datetime.fromisoformat(state["last_rebalance"])
        days_since = (datetime.now() - last).days
        days_until = HORIZON - days_since
        logger.info(f"  Skipping - last rebalance {days_since}d ago, "
                     f"next in {days_until}d")
        state["last_run"] = datetime.now().isoformat()
        save_state(state, mc)
        logger.info(report.format_summary())
        return

    # -- Step 3: Compute features (data already downloaded) --
    logger.info("\n[3/5] COMPUTING FEATURES")
    report.start_step("compute_features")
    logger.info(f"  Using {len(stock_data)} stocks, "
                f"{len(macro_features.columns)} macro features")
    report.set("stocks_downloaded", len(stock_data))
    report.end_step("compute_features")

    # -- Step 4: Generate predictions --
    logger.info("\n[4/5] GENERATING PREDICTIONS")
    logger.info(f"  Feature version: {mc.feature_version}")
    rankings = predict_rankings(
        stock_data, macro_features, model, feature_cols, logger, report,
        feature_version=mc.feature_version,
    )

    if len(rankings) < TOP_N:
        logger.error(f"Only {len(rankings)} predictions - need at least {TOP_N}")
        report.add_error(f"Insufficient predictions: {len(rankings)} < {TOP_N}")
        logger.info(report.format_summary())
        return

    target_symbols = [sym for sym, _ in rankings[:TOP_N]]
    report.set("target_portfolio", rankings[:TOP_N])

    logger.info(f"\n  Target portfolio ({TOP_N} stocks):")
    for i, (sym, pred) in enumerate(rankings[:TOP_N]):
        logger.info(f"    {i+1:2d}. {sym:6s}  pred={pred:+6.2f}%")

    # -- V6/V8: Compute regime score & conviction weights --
    tw = None  # target_weights: None = equal-weight (v4/v5 default)
    regime_exposure = 1.0

    if mc.feature_version in ("v6", "v8") and macro_data is not None:
        vtag = mc.feature_version.upper()
        logger.info(f"\n  [{vtag}] Computing market regime & conviction weights...")
        try:
            regime_score = compute_live_regime_score(macro_data)
            regime_exposure = regime_to_exposure(regime_score)
            regime_label = (
                "FAVORABLE" if regime_score > 0.3
                else "HOSTILE" if regime_score < -0.3
                else "NEUTRAL"
            )
            logger.info(f"  [{vtag}] Regime score: {regime_score:+.3f} ({regime_label})")
            logger.info(f"  [{vtag}] Exposure multiplier: {regime_exposure:.2f}")

            if mc.feature_version == "v8":
                # V8: sector-neutral conviction weights
                sector_map = _load_sector_map_for_pipeline()
                if not sector_map:
                    # Try loading from model bundle
                    try:
                        model_path = _resolve_model_path(mc.name)
                        with open(model_path, "rb") as f:
                            bundle = pickle.load(f)
                        sector_map = bundle.get("sector_map", {})
                    except Exception:
                        pass

                max_per_sector = 3
                # Check model bundle for sector_config
                try:
                    model_path = _resolve_model_path(mc.name)
                    with open(model_path, "rb") as f:
                        bundle = pickle.load(f)
                    sc = bundle.get("sector_config", {})
                    max_per_sector = sc.get("max_per_sector", 3)
                except Exception:
                    pass

                tw = sector_neutral_weights(
                    rankings, sector_map,
                    top_n=TOP_N,
                    max_per_sector=max_per_sector,
                    max_weight_multiple=2.0,
                    regime_exposure=regime_exposure,
                )
                # Log sector distribution
                sector_counts = {}
                for sym_s in tw:
                    sec = sector_map.get(sym_s, "Unknown")
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                logger.info(f"  [V8] Sector distribution: {sector_counts}")
            else:
                tw = conviction_weights(
                    rankings, top_n=TOP_N,
                    max_weight_multiple=2.0,
                    regime_exposure=regime_exposure,
                )

            total_alloc = sum(tw.values())
            max_w = max(tw.values()) if tw else 0
            min_w = min(tw.values()) if tw else 0
            logger.info(f"  [{vtag}] Conviction weights: total_alloc={total_alloc:.4f}, "
                        f"max={max_w:.4f}, min={min_w:.4f}")
            for sym_w, w in sorted(tw.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {sym_w:6s}  weight={w:.4f}")
            if len(tw) > 5:
                logger.info(f"    ... and {len(tw) - 5} more")

            # Use conviction-weighted target symbols
            target_symbols = list(tw.keys())

            report.set("v6_regime", {
                "regime_score": round(regime_score, 4),
                "regime_label": regime_label,
                "exposure_multiplier": round(regime_exposure, 4),
                "conviction_weights": {s: round(w, 6) for s, w in tw.items()},
            })
        except Exception as e:
            logger.warning(f"  [{vtag}] Regime/conviction failed, falling back to "
                          f"equal-weight: {e}")
            report.add_error(f"{vtag} regime computation failed: {e}")
            tw = None
    elif mc.feature_version in ("v6", "v8"):
        logger.warning(f"  [{mc.feature_version.upper()}] No macro_data available for regime - using equal-weight")

    # -- Step 5: Rebalance --
    logger.info("\n[5/5] REBALANCING PORTFOLIO")
    result = rebalance_portfolio(
        target_symbols, rankings, mc, journal, logger, report,
        dry_run=dry_run, target_weights=tw,
    )

    # -- Save state --
    state["last_rebalance"] = datetime.now().isoformat()
    state["last_run"] = datetime.now().isoformat()
    state["run_count"] = state.get("run_count", 0) + 1
    history_entry = {
        "date": datetime.now().isoformat(),
        "run_id": f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "target_symbols": target_symbols,
        "predictions": {s: round(p, 4) for s, p in rankings[:TOP_N]},
        "result": {k: v for k, v in result.items()
                   if k not in ("sells_detail", "buys_detail")},
    }
    if tw is not None:
        history_entry["regime_exposure"] = round(regime_exposure, 4)
        history_entry["conviction_weights"] = {s: round(w, 6) for s, w in tw.items()}
    state["history"].append(history_entry)
    state["history"] = state["history"][-100:]
    save_state(state, mc)

    # -- Final summary --
    summary = report.format_summary()
    logger.info(summary)
    logger.info(f"Log saved to: {log_file}")


# ═══════════════════════════════════════════════════════════════════════════
# CUT-LOSS SCANNER — runs every minute during market hours for V7+
# ═══════════════════════════════════════════════════════════════════════════

# Track peak prices per (model, symbol) for trailing stop
_peak_prices: dict[tuple[str, str], float] = {}
_peak_lock = threading.Lock()

# Track daily portfolio start values for portfolio-level stop
_daily_portfolio_start: dict[str, float] = {}


def _is_market_open() -> bool:
    """Check if US stock market is currently open (simple time check)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def cutloss_scan():
    """Scan all cutloss-enabled models and execute stops if triggered.

    Called every 60 seconds by the scheduler during market hours.
    Three stop types:
      1. Hard stop: sell if position is down X% from avg entry price
      2. Trailing stop: sell if position is down X% from peak price since entry
      3. Portfolio stop: liquidate ALL positions if daily portfolio drawdown > X%
    """
    if not _is_market_open():
        return

    logger = logging.getLogger("cutloss")

    models = get_active_models()
    cutloss_models = [mc for mc in models if mc.enable_cutloss]

    if not cutloss_models:
        return

    for mc in cutloss_models:
        try:
            _cutloss_scan_model(mc, logger)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: scan error: {e}")


def _cutloss_scan_model(mc: ModelConfig, logger):
    """Run cut-loss checks for a single model."""
    # Fetch positions
    try:
        positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
    except Exception as e:
        logger.warning(f"[CUTLOSS] {mc.name}: failed to fetch positions: {e}")
        return

    if not positions:
        return

    # Fetch account for portfolio-level stop
    try:
        acct = alpaca_request("GET", "v2/account", mc, logger=logger)
        current_equity = float(acct.get("equity", 0))
        last_equity = float(acct.get("last_equity", 0))
    except Exception:
        current_equity = last_equity = 0

    # Initialize daily start if needed (first scan of the day)
    today = datetime.now().strftime("%Y-%m-%d")
    daily_key = f"{mc.name}_{today}"
    if daily_key not in _daily_portfolio_start and last_equity > 0:
        _daily_portfolio_start[daily_key] = last_equity

    start_equity = _daily_portfolio_start.get(daily_key, last_equity)

    # ── Portfolio-level stop check ───────────────────────────
    if start_equity > 0 and current_equity > 0:
        daily_drawdown_pct = (current_equity / start_equity - 1) * 100
        if daily_drawdown_pct <= mc.cutloss_portfolio_stop:
            logger.warning(
                f"[CUTLOSS] {mc.name}: PORTFOLIO STOP triggered! "
                f"Daily drawdown: {daily_drawdown_pct:.2f}% <= {mc.cutloss_portfolio_stop}%. "
                f"Liquidating ALL positions."
            )
            _liquidate_all(mc, positions, "portfolio_stop", logger)
            return

    # ── Per-position stop checks ─────────────────────────────
    symbols_to_sell = []

    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("qty", 0))
        avg_entry = float(p.get("avg_entry_price", 0))
        current_price = float(p.get("current_price", 0))

        if avg_entry <= 0 or current_price <= 0 or qty <= 0:
            continue

        # Update peak price for trailing stop
        peak_key = (mc.name, sym)
        with _peak_lock:
            prev_peak = _peak_prices.get(peak_key, avg_entry)
            current_peak = max(prev_peak, current_price)
            _peak_prices[peak_key] = current_peak

        # Hard stop: down X% from entry
        pct_from_entry = (current_price / avg_entry - 1) * 100
        if pct_from_entry <= mc.cutloss_hard_stop:
            logger.warning(
                f"[CUTLOSS] {mc.name}: HARD STOP on {sym}! "
                f"{pct_from_entry:.2f}% from entry (threshold: {mc.cutloss_hard_stop}%)"
            )
            symbols_to_sell.append((sym, qty, "hard_stop", pct_from_entry))
            continue

        # Trailing stop: down X% from peak
        pct_from_peak = (current_price / current_peak - 1) * 100
        if pct_from_peak <= mc.cutloss_trailing_stop:
            logger.warning(
                f"[CUTLOSS] {mc.name}: TRAILING STOP on {sym}! "
                f"{pct_from_peak:.2f}% from peak ${current_peak:.2f} "
                f"(threshold: {mc.cutloss_trailing_stop}%)"
            )
            symbols_to_sell.append((sym, qty, "trailing_stop", pct_from_peak))
            continue

    # Execute sells
    for sym, qty, reason, pct in symbols_to_sell:
        try:
            _execute_cutloss_sell(mc, sym, qty, reason, pct, logger)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: failed to sell {sym}: {e}")


def _execute_cutloss_sell(mc: ModelConfig, symbol: str, qty: float,
                          reason: str, pct: float, logger):
    """Execute a market sell order for a cut-loss trigger."""
    logger.info(f"[CUTLOSS] {mc.name}: SELLING {symbol} qty={qty:.2f} "
                f"reason={reason} ({pct:.2f}%)")

    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    }

    try:
        result = alpaca_request("POST", "v2/orders", mc, data=order_data, logger=logger)
        order_id = result.get("id", "?")
        logger.info(f"[CUTLOSS] {mc.name}: {symbol} sell order placed: {order_id}")

        # Log to trade journal
        journal = TradeJournal(mc.name)
        record = TradeRecord(
            trade_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="sell",
            action=reason,
            order_type="market",
            notional_target=0,
            order_status="submitted",
            order_id=order_id,
            shares=qty,
        )
        journal.log(record)

        # Clear peak tracking for this symbol
        peak_key = (mc.name, symbol)
        _peak_prices.pop(peak_key, None)

    except Exception as e:
        logger.error(f"[CUTLOSS] {mc.name}: order failed for {symbol}: {e}")
        raise


def _liquidate_all(mc: ModelConfig, positions: list, reason: str, logger):
    """Liquidate all positions for a model (portfolio-level stop)."""
    logger.warning(f"[CUTLOSS] {mc.name}: LIQUIDATING ALL {len(positions)} positions ({reason})")

    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("qty", 0))
        if qty > 0:
            try:
                _execute_cutloss_sell(mc, sym, qty, reason, 0.0, logger)
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to liquidate {sym}: {e}")

    # Clear all peak tracking for this model
    keys_to_remove = [k for k in _peak_prices if k[0] == mc.name]
    for k in keys_to_remove:
        _peak_prices.pop(k, None)


def run_pipeline(dry_run: bool = False, force: bool = False,
                 model_filter: str = None):
    """Execute the full daily pipeline for all active models."""
    logger, log_file = setup_logging("main")

    # -- Discover models --
    models = get_active_models()
    if model_filter:
        models = [m for m in models if m.name == model_filter]

    if not models:
        logger.error("No active models found. Set MODEL_V4_ALPACA_KEY/SECRET "
                     "(or ALPACA_API_KEY/SECRET) and ensure model files exist.")
        sys.exit(1)

    logger.info(f"Active models: {[m.name for m in models]}")

    # -- Download data once (shared across models) --
    report = RunReport()

    logger.info("\n[1/2] DOWNLOADING MARKET DATA (shared)")
    symbols = get_tradeable_symbols(logger, report)
    if not symbols:
        logger.error("No symbols found - cannot proceed")
        sys.exit(1)

    stock_data = download_bars(symbols, LOOKBACK_DAYS, logger, report)
    macro_data = download_macro(LOOKBACK_DAYS, logger, report)

    if not stock_data:
        logger.error("No stock data downloaded - cannot proceed")
        sys.exit(1)

    logger.info("\n[2/2] COMPUTING MACRO FEATURES (shared)")
    macro_features = compute_macro_features(macro_data)
    logger.info(f"  Macro features: {len(macro_features.columns)} columns, "
                f"{len(macro_features)} days")

    # -- Run each model --
    for mc in models:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  Running model: {mc.name.upper()}")
        logger.info(f"{'=' * 70}")
        try:
            run_single_model(mc, stock_data, macro_features,
                             macro_data=macro_data,
                             dry_run=dry_run, force=force)
        except Exception as e:
            logger.error(f"Model {mc.name} FAILED: {e}\n{traceback.format_exc()}")

    logger.info("\nAll models complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Trading Pipeline (Multi-Model)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Predict only, no orders")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model only (e.g. v4, v5)")
    args = parser.parse_args()

    try:
        run_pipeline(dry_run=args.dry_run, force=args.force,
                     model_filter=args.model)
    except Exception as e:
        logging.getLogger("pipeline").error(
            f"FATAL: {e}\n{traceback.format_exc()}"
        )
        sys.exit(1)
