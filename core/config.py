"""Per-slot model configuration — what runs, where its keys live, what
risk overlays are enabled.

A "slot" is one Alpaca account paired with one model file. The dashboard
edits a JSON config (`model_config.json` on the Railway volume) with up
to N slots; this module reads that config and falls back to env vars if
no slot in the config file resolves to a usable model + key pair.

`ModelConfig` is the per-slot runtime dataclass. It carries the model
file path, Alpaca creds, and the cutloss thresholds — everything the
runner needs to operate one slot.

`MODEL_REGISTRY` lists every model name we *can* run (v4..v8) and where
to find its pickled artifact. `MODEL_DESCRIPTIONS` is the human-facing
metadata that the dashboard renders.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# DATA_DIR is Railway's persistent volume; BASE_DIR is the repo root
# (deploy/). Computed at import time so resolving paths doesn't depend on
# any other module being imported first.
_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
_BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = _DATA_DIR / "model_config.json"


# ═══════════════════════════════════════════════════════════════════════════
# ModelConfig — runtime per-slot info
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a single model instance (one Alpaca account)."""
    name: str                   # e.g. "v4", "v5"
    model_path: Path            # path to model.pkl
    feature_version: str        # "v4", "v5", or "v6"
    alpaca_key: str             # per-slot Alpaca API key
    alpaca_secret: str          # per-slot Alpaca API secret
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    state_path: Path = None     # auto-set if None
    enable_cutloss: bool = False
    cutloss_hard_stop: float = -8.0
    cutloss_trailing_stop: float = -5.0
    cutloss_portfolio_stop: float = -3.0

    def __post_init__(self):
        if self.state_path is None:
            self.state_path = _DATA_DIR / "state" / f"pipeline_state_{self.name}.json"


# ═══════════════════════════════════════════════════════════════════════════
# Model registry + dashboard descriptions
# ═══════════════════════════════════════════════════════════════════════════

# All known models and where to find their pickled artifact.
MODEL_REGISTRY: dict[str, dict] = {
    "v4": {"feature_version": "v4", "model_dir": "v4", "fallback_model": "ml_v4_model.pkl"},
    "v5": {"feature_version": "v5", "model_dir": "v5", "fallback_model": None},
    "v6": {"feature_version": "v6", "model_dir": "v6", "fallback_model": None},
    "v7": {"feature_version": "v6", "model_dir": "v7", "fallback_model": None},
    "v8": {"feature_version": "v8", "model_dir": "v8", "fallback_model": None},
}


# Human-facing model documentation shown on the dashboard's About panel.
# Keep these in sync with the actual model behavior — drift is misleading.
MODEL_DESCRIPTIONS: dict[str, dict] = {
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
                "  3. Portfolio stop: tiered soft scaling — Tier 1 (-3%) → 60% exposure, "
                "Tier 2 (-5%) → 30%, Tier 3 (-7%) → full liquidate + trip flag.\n"
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
# Dashboard-managed config (model_config.json)
# ═══════════════════════════════════════════════════════════════════════════

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
    """Load slot config from the persistent JSON file.

    Falls back to a fresh default config (all-empty keys) if the file is
    missing or unparseable. The dashboard never sees a hard failure.
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return _default_config()


def save_model_config(config: dict) -> None:
    """Save slot config to the persistent JSON file. Sets `updated_at`."""
    config["updated_at"] = datetime.now().isoformat()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _resolve_model_path(model_name: str) -> Path:
    """Find model.pkl for a given model name.

    Looks first at `model/<dir>/model.pkl`, then at the registry's
    `fallback_model` filename (used by v4's legacy `ml_v4_model.pkl`).
    Returns the primary path even when missing — the caller checks
    `.exists()` to handle absence.
    """
    reg = MODEL_REGISTRY.get(model_name, {})
    model_dir = reg.get("model_dir", model_name)
    path = _BASE_DIR / "model" / model_dir / "model.pkl"
    if path.exists():
        return path
    fallback = reg.get("fallback_model")
    if fallback:
        alt = _BASE_DIR / "model" / fallback
        if alt.exists():
            return alt
    return path


def get_active_models() -> list[ModelConfig]:
    """Build active models from config file, falling back to env vars.

    Priority: dashboard-edited config file slots → env vars. Only slots
    with valid API keys AND an existing model file are activated. When no
    slot in the config resolves, env vars get a chance — that's how the
    bot bootstraps on a fresh container before the dashboard saves
    anything.
    """
    config = load_model_config()
    models: list[ModelConfig] = []
    activated_via_config = False

    # -- Try config file first --
    for slot in config.get("slots", []):
        if not slot.get("enabled", True):
            continue
        model_name = slot.get("model", "")
        if model_name not in MODEL_REGISTRY:
            continue

        key = slot.get("alpaca_key", "")
        secret = slot.get("alpaca_secret", "")
        model_path = _resolve_model_path(model_name)
        reg = MODEL_REGISTRY[model_name]

        print(f"[MODEL DETECT] slot {slot.get('slot_id')}: model={model_name}, "
              f"key={'YES' if key else 'NO'}, secret={'YES' if secret else 'NO'}, "
              f"model_file={model_path} exists={model_path.exists()}", flush=True)

        if key and secret and model_path.exists():
            activated_via_config = True
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

    # -- Fallback to env vars if no slot in the config actually activated.
    # Keys-but-no-pickle is treated as "config didn't work" so env vars
    # still get a chance.
    if not activated_via_config:
        print("[MODEL DETECT] No models activated from config; trying env vars", flush=True)
        for name in MODEL_REGISTRY:
            key_env = f"MODEL_{name.upper()}_ALPACA_KEY"
            secret_env = f"MODEL_{name.upper()}_ALPACA_SECRET"
            # Legacy ALPACA_API_KEY env vars are honored only for v4
            # (the original single-model setup).
            fallback_key = "ALPACA_API_KEY" if name == "v4" else None
            fallback_secret = "ALPACA_SECRET_KEY" if name == "v4" else None

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

    # Debug: list pkl files actually present, for diagnosing missing-model logs.
    model_dir = _BASE_DIR / "model"
    if model_dir.exists():
        try:
            result = subprocess.run(
                ["find", str(model_dir), "-name", "*.pkl"],
                capture_output=True, text=True, timeout=5,
            )
            print(f"[MODEL DETECT] .pkl files found: {result.stdout.strip()}", flush=True)
        except Exception:
            pass

    print(f"[MODEL DETECT] Active: {[m.name for m in models]}", flush=True)
    return models


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca key sanity check (dashboard "Test key" button)
# ═══════════════════════════════════════════════════════════════════════════

def test_alpaca_key(key: str, secret: str,
                    base_url: str = "https://paper-api.alpaca.markets") -> dict:
    """Test an Alpaca API key pair. Returns account info or {ok: False, error: ...}."""
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
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
