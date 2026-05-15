"""Daily pipeline orchestrator.

`run_pipeline()` is the top-level entry point hit by the daily cron. It:

  1. Discovers active model slots (from config file or env vars)
  2. Downloads bars + macro **once**, shared across all slots
  3. Computes macro features once
  4. Hands off to `run_single_model()` per slot, which:
     - loads the slot's pickled model
     - generates predictions
     - filters inactive Alpaca assets
     - applies V6/V8 regime + conviction-weighted sizing
     - rebalances via `core.orders.rebalance_portfolio`
     - updates persisted state + history

`load_model_bundle()` handles the custom unpickler that maps any
training-script module path (`__main__`, `scripts.train_ml_v6`, …) to
the in-tree `StackedEnsemble` / `EnsembleModel` classes so model
artifacts deserialize cleanly regardless of how they were trained.
"""

from __future__ import annotations

import pickle
import traceback
from datetime import datetime, timezone

from core.config import ModelConfig, get_active_models
from core.data import download_bars, download_macro
from core.ensemble import EnsembleModel, StackedEnsemble
from core.features import compute_macro_features
from core.inference import predict_rankings
from core.journal import TradeJournal
from core.logging_setup import setup_logging
from core.market import is_market_open
from core.orders import fetch_inactive_assets, rebalance_portfolio
from core.portfolio import (
    compute_live_regime_score, conviction_weights,
    load_sector_map_for_pipeline, regime_to_exposure,
    sector_neutral_weights,
)
from core.run_report import RunReport
from core.state import load_state, save_state, should_rebalance
from core.universe import EXCLUDED_SYMBOLS, get_tradeable_symbols


# Default config. Equivalent to the legacy pipeline-level constants. Callers
# can override per-run via kwargs.
DEFAULT_TOP_N = 20
DEFAULT_HORIZON = 5
DEFAULT_LOOKBACK_DAYS = 300

# Data-quality safeguards: refuse to rebalance with partial yfinance pulls.
# Full universe ≈ 1000 stocks; <500 means rate-limiting hit. Macro is normally
# 22 features; <15 means partial download.
MIN_STOCKS_REQUIRED = 500
MIN_MACRO_FEATURES = 15


def load_model_bundle(model_path) -> dict:
    """Load a pickled model bundle, remapping training-script module paths.

    Training scripts pickle `StackedEnsemble` / `EnsembleModel` under their
    own module names (`__main__`, `scripts.train_ml_v6`, ...). On Railway
    those modules don't exist, so we install a custom unpickler that maps
    any known training-script path to the in-tree class.
    """
    class _PipelineUnpickler(pickle.Unpickler):
        _class_map = {
            "StackedEnsemble": StackedEnsemble,
            "EnsembleModel": EnsembleModel,
        }
        _known_training_modules = {
            "__main__",
            "scripts.train_ml_v4", "scripts.train_ml_v5",
            "scripts.train_ml_v6", "scripts.train_ml_v7",
            "scripts.train_ml_v8",
            "train_ml_v4", "train_ml_v5",
            "train_ml_v6", "train_ml_v7", "train_ml_v8",
        }

        def find_class(self, module, name):
            if name in self._class_map and module in self._known_training_modules:
                return self._class_map[name]
            return super().find_class(module, name)

    with open(model_path, "rb") as fh:
        return _PipelineUnpickler(fh).load()


def run_single_model(
    mc: ModelConfig,
    stock_data: dict,
    macro_features,
    macro_data: dict | None = None,
    dry_run: bool = False,
    force: bool = False,
    top_n: int = DEFAULT_TOP_N,
    horizon: int = DEFAULT_HORIZON,
) -> None:
    """Execute the pipeline for one model slot."""
    logger, log_file = setup_logging(mc.name)
    report = RunReport()
    journal = TradeJournal(mc.name)

    logger.info("=" * 70)
    logger.info(f"  ML TRADING PIPELINE - Model: {mc.name.upper()}")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Config: H{horizon}_LongOnly{top_n}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
    logger.info(f"  Force rebalance: {force}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"  Trade journal: {journal.jsonl_path}")
    logger.info("=" * 70)

    # -- Step 1: Load model -------------------------------------------------
    logger.info("\n[1/5] LOADING MODEL")
    report.start_step("load_model")
    if not mc.model_path.exists():
        logger.error(f"Model not found: {mc.model_path}")
        report.add_error(f"Model not found: {mc.model_path}")
        logger.info(report.format_summary())
        return

    model_bundle = load_model_bundle(mc.model_path)
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    logger.info(
        f"  Model: {len(feature_cols)} features, "
        f"horizon={model_bundle.get('horizon', '?')}d, "
        f"task={model_bundle.get('task', '?')}"
    )
    logger.info(f"  Trained: {model_bundle.get('saved_at', '?')}")
    report.set("n_features", len(feature_cols))
    report.end_step("load_model")

    # -- Step 2: Check state ------------------------------------------------
    state = load_state(mc)
    logger.info(
        f"\n  State: run #{state.get('run_count', 0) + 1}, "
        f"last rebalance: {state.get('last_rebalance', 'never')}"
    )

    # Cut-loss circuit breaker: if portfolio_stop tripped today, sit in cash
    # for the rest of the session. Otherwise rebalance would buy fresh
    # positions that the next 60s cutloss tick would immediately liquidate
    # again (the daily anchor is yesterday's close — see core/risk.py).
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

    if not should_rebalance(state, horizon_days=horizon, force=force):
        last = datetime.fromisoformat(state["last_rebalance"])
        days_since = (datetime.now() - last).days
        days_until = horizon - days_since
        logger.info(
            f"  Skipping - last rebalance {days_since}d ago, "
            f"next in {days_until}d"
        )
        state["last_run"] = datetime.now().isoformat()
        save_state(state, mc)
        logger.info(report.format_summary())
        return

    # -- Step 3: Verify shared data is sufficient --------------------------
    logger.info("\n[3/5] COMPUTING FEATURES")
    report.start_step("compute_features")
    logger.info(
        f"  Using {len(stock_data)} stocks, "
        f"{len(macro_features.columns)} macro features"
    )
    report.set("stocks_downloaded", len(stock_data))

    if len(stock_data) < MIN_STOCKS_REQUIRED:
        msg = (
            f"Insufficient stock data: {len(stock_data)} < {MIN_STOCKS_REQUIRED} required. "
            f"Download was likely rate-limited. Aborting to avoid bad trades."
        )
        logger.error(msg)
        report.add_error(msg)
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return
    if len(macro_features.columns) < MIN_MACRO_FEATURES:
        msg = (
            f"Insufficient macro data: {len(macro_features.columns)} < "
            f"{MIN_MACRO_FEATURES} required. Download was likely rate-limited. "
            f"Aborting to avoid bad trades."
        )
        logger.error(msg)
        report.add_error(msg)
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return

    report.end_step("compute_features")

    # -- Step 4: Generate predictions ---------------------------------------
    logger.info("\n[4/5] GENERATING PREDICTIONS")
    logger.info(f"  Feature version: {mc.feature_version}")
    rankings = predict_rankings(
        stock_data, macro_features, model, feature_cols, logger, report,
        feature_version=mc.feature_version,
    )

    # Filter inactive/untradeable assets (e.g. HOLX after delisting).
    # Walk further down the ranking until we have at least `top_n` tradeable
    # candidates so a few delistings don't abort the rebalance.
    if not dry_run and mc.alpaca_key:
        all_inactive: set[str] = set()
        check_window = top_n * 2
        max_window = min(len(rankings), top_n * 5)  # never check more than top 100
        while True:
            window_syms = [sym for sym, _ in rankings[:check_window]]
            inactive = fetch_inactive_assets(window_syms, mc, logger, top_n=top_n)
            all_inactive.update(inactive)
            tradeable_in_window = [s for s in window_syms if s not in all_inactive]
            if len(tradeable_in_window) >= top_n or check_window >= max_window:
                break
            check_window = min(check_window + top_n, max_window)
            logger.info(
                f"  Only {len(tradeable_in_window)}/{top_n} tradeable in top "
                f"{check_window - top_n}; extending check to top {check_window}"
            )
        if all_inactive:
            rankings = [(sym, pred) for sym, pred in rankings if sym not in all_inactive]
            report.set("inactive_assets_filtered", sorted(all_inactive))

    if len(rankings) < top_n:
        logger.error(f"Only {len(rankings)} predictions - need at least {top_n}")
        report.add_error(f"Insufficient predictions: {len(rankings)} < {top_n}")
        logger.info(report.format_summary())
        return

    # Final guard: re-apply EXCLUDED_SYMBOLS in case a cached universe
    # pre-dates an exclusion change.
    rankings = [(sym, p) for sym, p in rankings if sym not in EXCLUDED_SYMBOLS]

    target_symbols = [sym for sym, _ in rankings[:top_n]]
    report.set("target_portfolio", rankings[:top_n])

    logger.info(f"\n  Target portfolio ({top_n} stocks):")
    for i, (sym, pred) in enumerate(rankings[:top_n]):
        logger.info(f"    {i+1:2d}. {sym:6s}  pred={pred:+6.2f}%")

    # -- V6/V8: regime score → conviction-weighted (optionally sector-neutral)
    target_weights: dict[str, float] | None = None
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
                # Priority: cached sector map > embedded in model bundle > fallback.
                sector_map = load_sector_map_for_pipeline()
                if not sector_map:
                    sector_map = model_bundle.get("sector_map", {})
                if not sector_map:
                    logger.warning(
                        f"  [{vtag}] No sector map found! "
                        f"Falling back to conviction_weights (no sector constraint)."
                    )
                    target_weights = conviction_weights(
                        rankings, top_n=top_n,
                        max_weight_multiple=2.0,
                        regime_exposure=regime_exposure,
                    )
                else:
                    sc = model_bundle.get("sector_config", {})
                    max_per_sector = sc.get("max_per_sector", 3)
                    logger.info(
                        f"  [{vtag}] Sector map: {len(sector_map)} stocks mapped, "
                        f"max {max_per_sector}/sector"
                    )
                    target_weights = sector_neutral_weights(
                        rankings, sector_map,
                        top_n=top_n,
                        max_per_sector=max_per_sector,
                        max_weight_multiple=2.0,
                        regime_exposure=regime_exposure,
                    )
                # Log sector distribution
                sector_counts: dict[str, int] = {}
                for sym_s in target_weights:
                    sec = sector_map.get(sym_s, "Unknown")
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                logger.info(f"  [V8] Sector distribution: {sector_counts}")
            else:
                target_weights = conviction_weights(
                    rankings, top_n=top_n,
                    max_weight_multiple=2.0,
                    regime_exposure=regime_exposure,
                )

            total_alloc = sum(target_weights.values())
            max_w = max(target_weights.values()) if target_weights else 0
            min_w = min(target_weights.values()) if target_weights else 0
            logger.info(
                f"  [{vtag}] Conviction weights: total_alloc={total_alloc:.4f}, "
                f"max={max_w:.4f}, min={min_w:.4f}"
            )
            for sym_w, w in sorted(target_weights.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {sym_w:6s}  weight={w:.4f}")
            if len(target_weights) > 5:
                logger.info(f"    ... and {len(target_weights) - 5} more")

            target_symbols = list(target_weights.keys())

            report.set("v6_regime", {
                "regime_score": round(regime_score, 4),
                "regime_label": regime_label,
                "exposure_multiplier": round(regime_exposure, 4),
                "conviction_weights": {s: round(w, 6) for s, w in target_weights.items()},
            })
        except Exception as e:
            logger.warning(
                f"  [{vtag}] Regime/conviction failed, falling back to equal-weight: {e}"
            )
            report.add_error(f"{vtag} regime computation failed: {e}")
            target_weights = None
    elif mc.feature_version in ("v6", "v8"):
        logger.warning(
            f"  [{mc.feature_version.upper()}] No macro_data available "
            f"for regime - using equal-weight"
        )

    # -- Step 5: Rebalance --------------------------------------------------
    logger.info("\n[5/5] REBALANCING PORTFOLIO")

    # Avoid submitting `time_in_force=day` orders after market close — they
    # would expire unfilled. Skip the trading step on closed days; the
    # predictions and history are still saved.
    if not dry_run and not is_market_open():
        logger.warning(
            "  Market is closed (weekend, holiday, or outside 9:30–16:00 ET); "
            "skipping order submission. Predictions saved; rebalance will "
            "retry on the next scheduled run during market hours."
        )
        report.add_warning("Rebalance skipped: market closed")
        result: dict = {"dry_run": False, "skipped_market_closed": True}
    else:
        result = rebalance_portfolio(
            target_symbols, rankings, mc, journal, logger, report,
            dry_run=dry_run, target_weights=target_weights,
        )

    # -- Save state ---------------------------------------------------------
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
    history_entry: dict = {
        "date": datetime.now().isoformat(),
        "run_id": f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "target_symbols": target_symbols,
        "predictions": {s: round(p, 4) for s, p in rankings[:top_n]},
        "result": {k: v for k, v in result.items()
                   if k not in ("sells_detail", "buys_detail")},
    }
    if target_weights is not None:
        history_entry["regime_exposure"] = round(regime_exposure, 4)
        history_entry["conviction_weights"] = {
            s: round(w, 6) for s, w in target_weights.items()
        }
    state["history"].append(history_entry)
    state["history"] = state["history"][-100:]
    save_state(state, mc)

    summary = report.format_summary()
    logger.info(summary)
    logger.info(f"Log saved to: {log_file}")


def run_pipeline(dry_run: bool = False, force: bool = False,
                 model_filter: str | None = None,
                 top_n: int = DEFAULT_TOP_N,
                 horizon: int = DEFAULT_HORIZON,
                 lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
    """Execute the full daily pipeline for all active models.

    Downloads data once (universe + bars + macro) and reuses it across
    every active slot. Each slot's exceptions are caught individually so
    one slot's failure doesn't bring down the others.
    """
    logger, _log_file = setup_logging("main")

    # -- Discover models ----------------------------------------------------
    models = get_active_models()
    if model_filter:
        models = [m for m in models if m.name == model_filter]

    if not models:
        logger.error(
            "No active models found. Set MODEL_V4_ALPACA_KEY/SECRET "
            "(or ALPACA_API_KEY/SECRET) and ensure model files exist."
        )
        return

    logger.info(f"Active models: {[m.name for m in models]}")

    # -- Download data once (shared across models) --------------------------
    report = RunReport()

    logger.info("\n[1/2] DOWNLOADING MARKET DATA (shared)")
    symbols = get_tradeable_symbols(logger, report)
    if not symbols:
        logger.error("No symbols found - cannot proceed")
        return

    stock_data = download_bars(symbols, lookback_days, logger, report)
    macro_data = download_macro(lookback_days, logger, report)

    if not stock_data:
        logger.error("No stock data downloaded - cannot proceed")
        return

    logger.info("\n[2/2] COMPUTING MACRO FEATURES (shared)")
    macro_features = compute_macro_features(macro_data)
    logger.info(
        f"  Macro features: {len(macro_features.columns)} columns, "
        f"{len(macro_features)} days"
    )

    # -- Run each model -----------------------------------------------------
    for mc in models:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  Running model: {mc.name.upper()}")
        logger.info(f"{'=' * 70}")
        try:
            run_single_model(
                mc, stock_data, macro_features,
                macro_data=macro_data,
                dry_run=dry_run, force=force,
                top_n=top_n, horizon=horizon,
            )
        except Exception as e:
            logger.error(f"Model {mc.name} FAILED: {e}\n{traceback.format_exc()}")

    logger.info("\nAll models complete.")
