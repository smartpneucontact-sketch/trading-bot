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
# Ensemble wrappers — see core/ensemble.py. Re-exported here because the
# pickle unpickler (further down) maps any training-script module path to
# the local module's classes; these names must resolve in this namespace.
# ═══════════════════════════════════════════════════════════════════════════
from core.ensemble import StackedEnsemble, EnsembleModel  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TOP_N = 20              # Number of stocks to hold long
HORIZON = 5             # Prediction horizon (trading days)
LOOKBACK_DAYS = 300     # Days of history to download for feature computation

# Persistent data dir (Railway volume)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = DATA_DIR / "logs"
TRADE_DIR = DATA_DIR / "trades"

# Data download — see core/data.py + core/universe.py.
from core.data import (  # noqa: E402
    MACRO_TICKERS, MACRO_RENAME, MIN_HISTORY_DAYS,
    _normalize_columns, download_bars, download_macro,
)
from core.universe import (  # noqa: E402
    EXCLUDED_SYMBOLS, SYMBOL_CACHE_PATH,
    _wiki_read_html, _load_symbol_cache, _save_symbol_cache,
    get_tradeable_symbols,
)


# Multi-model config (ModelConfig + slot loaders + Alpaca key test) — see core/config.py.
from core.config import (  # noqa: E402
    ModelConfig,
    MODEL_REGISTRY, MODEL_DESCRIPTIONS, CONFIG_PATH,
    _default_config, load_model_config, save_model_config,
    _resolve_model_path, get_active_models, test_alpaca_key,
)


# MODEL_DESCRIPTIONS, MODEL_REGISTRY, CONFIG_PATH, config loaders, key tester
# all live in core/config.py (imported above).


# ═══════════════════════════════════════════════════════════════════════════
# TRADE JOURNAL — structured, queryable trade logging
# ═══════════════════════════════════════════════════════════════════════════

# Trade journal classes — see core/journal.py.
from core.journal import TradeRecord, TradeJournal  # noqa: E402, F401


# Run report — see core/run_report.py.
from core.run_report import RunReport  # noqa: E402, F401


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Logging setup — see core/logging_setup.py.
from core.logging_setup import setup_logging, get_cutloss_logger as _get_cutloss_logger  # noqa: E402


# Universe scraping (Wikipedia) + bar/macro download via yfinance now live
# in core/universe.py and core/data.py (imported above).


# Feature engineering — see core/features.py.
from core.features import (  # noqa: E402
    compute_stock_features, compute_stock_features_v6, compute_macro_features,
    get_feature_func as _get_feature_func,
    add_cross_sectional_ranks as _add_cross_sectional_ranks,
)

# Portfolio construction (regime + conviction + sector-neutral) — see core/portfolio.py.
from core.portfolio import (  # noqa: E402
    compute_live_regime_score, regime_to_exposure,
    conviction_weights, sector_neutral_weights,
    load_sector_map_for_pipeline as _load_sector_map_for_pipeline,
)

# Inference orchestrator — see core/inference.py.
from core.inference import predict_rankings  # noqa: E402


# Feature engineering, regime/portfolio sizing, and inference all live
# in core/{features,portfolio,inference}.py (imported above).



# ═══════════════════════════════════════════════════════════════════════════
# ALPACA TRADING (with structured trade logging)
# ═══════════════════════════════════════════════════════════════════════════

# Alpaca primitives — see core/alpaca.py.
from core.alpaca import (  # noqa: E402
    _to_alpaca_symbol, _make_alpaca_headers, alpaca_request,
    get_account, get_positions,
)

# Order placement, asset checking, status polling, full rebalance flow
# — see core/orders.py.
from core.orders import (  # noqa: E402
    fetch_inactive_assets as _core_fetch_inactive_assets,
    poll_order_status as _poll_order_status,
    rebalance_portfolio,
)


def _fetch_inactive_assets(symbols, mc, logger):
    """Thin wrapper that injects the pipeline-level TOP_N constant."""
    return _core_fetch_inactive_assets(symbols, mc, logger, top_n=TOP_N)


# rebalance_portfolio is now in core/orders.py (imported above).

# State management — see core/state.py.
from core.state import (  # noqa: E402
    load_state, save_state,
    trading_days_between as _trading_days_between,
    should_rebalance as _core_should_rebalance,
)


def should_rebalance(state: dict, force: bool = False) -> bool:
    """Thin wrapper that injects the pipeline-level HORIZON constant."""
    return _core_should_rebalance(state, horizon_days=HORIZON, force=force)


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

    with open(mc.model_path, "rb") as _model_fh:
        model_bundle = _PipelineUnpickler(_model_fh).load()
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

    # Cut-loss circuit breaker: if portfolio_stop fired today, sit in cash
    # for the rest of the session. Otherwise the rebalance would buy fresh
    # positions that the next 60s cutloss tick would immediately liquidate
    # again (the daily anchor is yesterday's close — see _cutloss_scan_model).
    today_iso = datetime.now().strftime("%Y-%m-%d")
    if state.get("portfolio_stop_tripped_date") == today_iso:
        logger.warning(
            f"  Skipping rebalance — portfolio_stop circuit breaker tripped "
            f"earlier today ({today_iso}). Staying in cash until next session."
        )
        state["last_run"] = datetime.now().isoformat()
        save_state(state, mc)
        logger.info(report.format_summary())
        return

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

    # SAFEGUARD: Refuse to run with incomplete data.
    # Full universe is ~1000 stocks. If we have <500, yfinance download was
    # partial (e.g. rate-limited) and predictions will be unreliable.
    MIN_STOCKS_REQUIRED = 500
    MIN_MACRO_FEATURES = 15  # normally 22
    if len(stock_data) < MIN_STOCKS_REQUIRED:
        msg = (f"Insufficient stock data: {len(stock_data)} < {MIN_STOCKS_REQUIRED} required. "
               f"Download was likely rate-limited. Aborting to avoid bad trades.")
        logger.error(msg)
        report.add_error(msg)
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return
    if len(macro_features.columns) < MIN_MACRO_FEATURES:
        msg = (f"Insufficient macro data: {len(macro_features.columns)} < {MIN_MACRO_FEATURES} required. "
               f"Download was likely rate-limited. Aborting to avoid bad trades.")
        logger.error(msg)
        report.add_error(msg)
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return

    report.end_step("compute_features")

    # -- Step 4: Generate predictions --
    logger.info("\n[4/5] GENERATING PREDICTIONS")
    logger.info(f"  Feature version: {mc.feature_version}")
    rankings = predict_rankings(
        stock_data, macro_features, model, feature_cols, logger, report,
        feature_version=mc.feature_version,
    )

    # Filter out inactive/untradeable assets (e.g. HOLX after delisting).
    # Walk further down the ranking until we have at least TOP_N tradeable
    # candidates so that a few delistings don't abort the rebalance.
    if not dry_run and mc.alpaca_key:
        all_inactive = set()
        check_window = TOP_N * 2
        max_window = min(len(rankings), TOP_N * 5)  # never check more than top 100
        while True:
            window_syms = [sym for sym, _ in rankings[:check_window]]
            inactive = _fetch_inactive_assets(window_syms, mc, logger)
            all_inactive.update(inactive)
            tradeable_in_window = [s for s in window_syms if s not in all_inactive]
            if len(tradeable_in_window) >= TOP_N or check_window >= max_window:
                break
            check_window = min(check_window + TOP_N, max_window)
            logger.info(f"  Only {len(tradeable_in_window)}/{TOP_N} tradeable in top "
                        f"{check_window - TOP_N}; extending check to top {check_window}")
        if all_inactive:
            rankings = [(sym, pred) for sym, pred in rankings if sym not in all_inactive]
            report.set("inactive_assets_filtered", sorted(all_inactive))

    if len(rankings) < TOP_N:
        logger.error(f"Only {len(rankings)} predictions - need at least {TOP_N}")
        report.add_error(f"Insufficient predictions: {len(rankings)} < {TOP_N}")
        logger.info(report.format_summary())
        return

    # Final guard: re-apply EXCLUDED_SYMBOLS in case anything slipped past
    # the universe filter (e.g. a cached universe pre-dating an exclusion).
    rankings = [(sym, p) for sym, p in rankings if sym not in EXCLUDED_SYMBOLS]

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
                # Priority: 1) cached file on disk, 2) embedded in model bundle
                sector_map = _load_sector_map_for_pipeline()
                if not sector_map:
                    sector_map = model_bundle.get("sector_map", {})
                if not sector_map:
                    logger.warning(f"  [{vtag}] No sector map found! Falling back to conviction_weights (no sector constraint).")
                    tw = conviction_weights(
                        rankings, top_n=TOP_N,
                        max_weight_multiple=2.0,
                        regime_exposure=regime_exposure,
                    )
                else:
                    sc = model_bundle.get("sector_config", {})
                    max_per_sector = sc.get("max_per_sector", 3)
                    logger.info(f"  [{vtag}] Sector map: {len(sector_map)} stocks mapped, max {max_per_sector}/sector")

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

    # Avoid submitting `time_in_force=day` orders after market close — they
    # would expire unfilled and leave the portfolio in a half-rebalanced
    # state. Skip the trading step on closed days; predictions and history
    # are still saved so the dashboard reflects the day's run.
    if not dry_run and not _is_market_open():
        logger.warning(
            "  Market is closed (weekend, holiday, or outside 9:30–16:00 ET); "
            "skipping order submission. Predictions saved; rebalance will "
            "retry on the next scheduled run during market hours."
        )
        report.add_warning("Rebalance skipped: market closed")
        result = {"dry_run": False, "skipped_market_closed": True}
    else:
        result = rebalance_portfolio(
            target_symbols, rankings, mc, journal, logger, report,
            dry_run=dry_run, target_weights=tw,
        )

    # -- Save state --
    # Only update last_rebalance if a rebalance actually happened. Skipping
    # the order-submission step (market closed, dry_run) means no trades
    # were placed; advancing last_rebalance would lock the model out of its
    # next opportunity by making should_rebalance() think it just rebalanced.
    rebalanced = not (
        dry_run
        or (isinstance(result, dict) and result.get("skipped_market_closed"))
    )
    if rebalanced:
        state["last_rebalance"] = datetime.now().isoformat()
    state["last_run"] = datetime.now().isoformat()
    state["run_count"] = state.get("run_count", 0) + 1
    history_entry = {
        "date": datetime.now().isoformat(),
        "run_id": f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
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
# CUT-LOSS SCANNER — see core/risk.py.
# ═══════════════════════════════════════════════════════════════════════════

# The full cutloss scanner (soft tiered portfolio stop, hard + trailing
# per-position stops, redistribute into model next-best picks) lives in
# core/risk.py. pipeline.py re-exports the public symbol `cutloss_scan`
# (called by the dashboard scheduler every 60s) and helper symbols that
# external code historically referenced.
from core.risk import (  # noqa: E402
    cutloss_scan,
    _cutloss_scan_model, _redistribute_after_cutloss,
    _place_redistribute_buy, _execute_cutloss_sell,
    _liquidate_all, _soft_scale_portfolio,
    _cutloss_state_lock,
)




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
        return

    logger.info(f"Active models: {[m.name for m in models]}")

    # -- Download data once (shared across models) --
    report = RunReport()

    logger.info("\n[1/2] DOWNLOADING MARKET DATA (shared)")
    symbols = get_tradeable_symbols(logger, report)
    if not symbols:
        logger.error("No symbols found - cannot proceed")
        return

    stock_data = download_bars(symbols, LOOKBACK_DAYS, logger, report)
    macro_data = download_macro(LOOKBACK_DAYS, logger, report)

    if not stock_data:
        logger.error("No stock data downloaded - cannot proceed")
        return

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
