#!/usr/bin/env python3
"""ML trading pipeline — multi-model orchestrator (top-level entry point).

The implementation now lives in the `core/` package; this module is a
thin re-export shim that:

  1. Re-exports the public symbols `dashboard.py` (and any other caller)
     historically imported from `pipeline.*` — so the dashboard keeps
     working without modification.
  2. Provides the CLI entry point that calls `core.runner.run_pipeline`.

Daily flow (driven by the dashboard's APScheduler):

  1. Download latest daily bars via yfinance (stocks + macro)
  2. Compute features (62 stock + 22 macro = 84 for v4; more for v6/v8)
  3. Run model predictions → rank stocks
  4. Rebalance portfolio via Alpaca API (long top 20, conviction-weighted)
  5. Log full run report + structured trade journal

Config: H5_LongOnly20 — 5-day horizon, long top 20, no shorts.
Rebalances every 5 trading days.

Multi-model: each slot has its own Alpaca account (`MODEL_<name>_ALPACA_KEY`
/ `MODEL_<name>_ALPACA_SECRET` env vars, or per-slot keys in the
dashboard-managed config file).

Usage:
    python pipeline.py                  # full run, all models
    python pipeline.py --dry-run        # predict only, no orders
    python pipeline.py --force          # force rebalance
    python pipeline.py --model v6       # run a single model only
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback


# ═══════════════════════════════════════════════════════════════════════════
# Public symbol re-exports — preserved for dashboard.py + any external code
# that historically did `import pipeline; pipeline.<thing>`. The actual
# implementations live in the `core/` package; modify them there.
# ═══════════════════════════════════════════════════════════════════════════

# Pipeline-level config constants
TOP_N = 20
HORIZON = 5
LOOKBACK_DAYS = 300

# Path constants — used by some dashboard endpoints that build absolute paths
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
BASE_DIR = Path(__file__).parent
LOG_DIR = DATA_DIR / "logs"
TRADE_DIR = DATA_DIR / "trades"

# Ensemble pickle-deserialization classes. Kept importable at the
# pipeline.* namespace for backwards compat with any caller that did
# `from pipeline import StackedEnsemble`.
from core.ensemble import EnsembleModel, StackedEnsemble  # noqa: E402, F401

# Per-slot config + model registry + Alpaca key tester
from core.config import (  # noqa: E402, F401
    CONFIG_PATH, MODEL_DESCRIPTIONS, MODEL_REGISTRY, ModelConfig,
    _default_config, _resolve_model_path,
    get_active_models, load_model_config, save_model_config,
    test_alpaca_key,
)

# Universe + market data
from core.universe import (  # noqa: E402, F401
    EXCLUDED_SYMBOLS, SYMBOL_CACHE_PATH,
    _load_symbol_cache, _save_symbol_cache, _wiki_read_html,
    get_tradeable_symbols,
)
from core.data import (  # noqa: E402, F401
    MACRO_RENAME, MACRO_TICKERS, MIN_HISTORY_DAYS,
    _normalize_columns, download_bars, download_macro,
)

# Logging + run-state primitives
from core.logging_setup import (  # noqa: E402, F401
    setup_logging, get_cutloss_logger as _get_cutloss_logger,
)
from core.run_report import RunReport  # noqa: E402, F401
from core.journal import TradeJournal, TradeRecord  # noqa: E402, F401
from core.state import (  # noqa: E402
    load_state, save_state,
    trading_days_between as _trading_days_between,
    should_rebalance as _core_should_rebalance,
)

# Features, portfolio sizing, inference
from core.features import (  # noqa: E402, F401
    compute_macro_features, compute_stock_features, compute_stock_features_v6,
    add_cross_sectional_ranks as _add_cross_sectional_ranks,
    get_feature_func as _get_feature_func,
)
from core.portfolio import (  # noqa: E402, F401
    compute_live_regime_score, conviction_weights, regime_to_exposure,
    sector_neutral_weights,
    load_sector_map_for_pipeline as _load_sector_map_for_pipeline,
)
from core.inference import predict_rankings  # noqa: E402, F401

# Alpaca primitives + orders
from core.alpaca import (  # noqa: E402, F401
    _make_alpaca_headers, _to_alpaca_symbol, alpaca_request,
    get_account, get_positions,
)
from core.orders import (  # noqa: E402, F401
    fetch_inactive_assets as _core_fetch_inactive_assets,
    poll_order_status as _poll_order_status,
    rebalance_portfolio,
)

# Cutloss scanner (with the soft tiered portfolio stop + Phase 1 fixes)
from core.risk import (  # noqa: E402, F401
    cutloss_scan,
    _cutloss_scan_model, _redistribute_after_cutloss,
    _place_redistribute_buy, _execute_cutloss_sell,
    _liquidate_all, _soft_scale_portfolio,
    _cutloss_state_lock,
)

# Daily-pipeline orchestrator
from core.runner import run_pipeline, run_single_model  # noqa: E402, F401


# ═══════════════════════════════════════════════════════════════════════════
# Thin wrappers that inject pipeline-level constants. These exist so the
# old call signatures keep working — every caller (including dashboard.py)
# was written before TOP_N / HORIZON were parameterised.
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_inactive_assets(symbols, mc, logger):
    """Injects the pipeline-level TOP_N constant."""
    return _core_fetch_inactive_assets(symbols, mc, logger, top_n=TOP_N)


def should_rebalance(state: dict, force: bool = False) -> bool:
    """Injects the pipeline-level HORIZON constant."""
    return _core_should_rebalance(state, horizon_days=HORIZON, force=force)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ML Trading Pipeline (Multi-Model)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Predict only, no orders")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model only (e.g. v4, v5)")
    args = parser.parse_args()
    try:
        run_pipeline(
            dry_run=args.dry_run, force=args.force, model_filter=args.model,
            top_n=TOP_N, horizon=HORIZON, lookback_days=LOOKBACK_DAYS,
        )
    except Exception as e:
        logging.getLogger("pipeline").error(
            f"FATAL: {e}\n{traceback.format_exc()}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
