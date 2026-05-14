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

def _fetch_inactive_assets(symbols: list[str], mc: ModelConfig,
                           logger) -> set[str]:
    """Check which symbols are inactive/untradeable on Alpaca.

    Queries Alpaca's asset endpoint for each candidate symbol. Returns a set
    of symbols that are NOT active or NOT tradeable. We only check the top
    candidates (2x portfolio size) to keep API calls reasonable.
    """
    import requests
    inactive = set()
    headers = _make_alpaca_headers(mc)
    check_count = min(len(symbols), TOP_N * 2)  # check top 40 candidates
    checked = 0
    for sym in symbols[:check_count]:
        try:
            url = f"{mc.alpaca_base_url}/v2/assets/{_to_alpaca_symbol(sym)}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                asset = resp.json()
                if asset.get("status") != "active" or not asset.get("tradable", True):
                    inactive.add(sym)
                    logger.info(f"    Filtered {sym}: status={asset.get('status')}, "
                                f"tradable={asset.get('tradable')}")
            elif resp.status_code == 404:
                inactive.add(sym)
                logger.info(f"    Filtered {sym}: not found on Alpaca")
            # else: API error, keep the symbol (fail later at order time with proper handling)
            checked += 1
        except Exception as e:
            logger.warning(f"    Asset check failed for {sym}: {e}")
            # On timeout/error, keep the symbol rather than blocking the pipeline
            checked += 1
    if inactive:
        logger.info(f"  Inactive asset filter: removed {len(inactive)} of {checked} checked "
                     f"({', '.join(sorted(inactive))})")
    else:
        logger.info(f"  Inactive asset filter: all {checked} checked symbols are active")
    return inactive


def _poll_order_status(order_id: str, mc: ModelConfig, logger,
                       max_wait: float = 8.0, interval: float = 0.5) -> dict:
    """Poll Alpaca for final order status (filled/rejected/etc).

    Most market orders fill in <1s but secondary-venue routing during
    volatile minutes can take several seconds. We wait up to 8s before
    giving up so the trade journal records real fill prices/quantities.
    Polling backs off after the first 4 attempts to limit API calls.
    """
    import time as _time
    terminal_states = {"filled", "canceled", "expired", "rejected",
                       "suspended", "replaced"}
    elapsed = 0.0
    attempts = 0
    while elapsed < max_wait:
        try:
            order = alpaca_request("GET", f"v2/orders/{order_id}", mc)
            status = order.get("status", "")
            if status in terminal_states:
                return order
        except Exception:
            break
        attempts += 1
        # 0.5s × 4 = 2s of fast polling, then 1s intervals up to 8s total
        sleep_for = interval if attempts < 4 else max(interval, 1.0)
        _time.sleep(sleep_for)
        elapsed += sleep_for
    # Return whatever we have (may still be pending)
    try:
        return alpaca_request("GET", f"v2/orders/{order_id}", mc)
    except Exception:
        return {"status": "unknown", "id": order_id}


def _to_alpaca_symbol(sym: str) -> str:
    """Convert a yfinance-style class-share ticker (BRK-B) to Alpaca form (BRK.B).

    The universe scrape stores symbols with dashes for yfinance compatibility;
    Alpaca's trading and data APIs expect a dot. Pattern: ends with `-X`
    where X is a single uppercase ASCII letter.
    """
    if (
        len(sym) >= 3
        and sym[-2] == "-"
        and sym[-1].isascii()
        and sym[-1].isalpha()
        and sym[-1].isupper()
    ):
        return sym[:-2] + "." + sym[-1]
    return sym


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
        resp = requests.get(url, headers=headers, timeout=15)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data, timeout=15)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=15)
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

    # Generate a unique run_id (microsecond precision avoids collisions when
    # two clicks land in the same second, e.g. cron + manual trigger).
    run_id = f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    acct = get_account(mc, logger, report)
    portfolio_value = float(acct["portfolio_value"])
    cash_available = float(acct["cash"])
    current_positions = get_positions(mc, logger)
    rb_data["positions_before"] = len(current_positions)

    if not target_symbols:
        logger.error(f"  [REBALANCE] {mc.name}: target_symbols is empty, aborting rebalance")
        report.add_error("Rebalance cancelled: no target symbols")
        report.end_step("rebalance")
        return rb_data

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
            order_ok = False
            alpaca_sym = _to_alpaca_symbol(sym)
            if order["action"] == "sell":
                # Close entire position
                resp = alpaca_request("DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger)
                # DELETE returns the closing order object (or empty on 204)
                order_status = resp.get("status", "accepted") if resp else "accepted"
                trade.order_id = resp.get("id") if resp else None
                trade.order_status = order_status
                trade.shares = order["qty"]
                if order_status in ("rejected", "canceled", "expired"):
                    logger.error(f"    REJECTED EXIT {sym:6s}: status={order_status}, "
                                 f"reason: {resp.get('reject_reason', 'unknown')}")
                    trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                    report.add_error(f"Order {order_status}: EXIT {sym} - {resp.get('reject_reason', '')}")
                    failed += 1
                else:
                    order_ok = True
                    logger.info(f"    OK  EXIT  {sym:6s}: closed {order['qty']:.2f} shares, "
                                f"P&L=${order.get('unrealized_pnl_usd', 0):+,.2f} "
                                f"({order.get('unrealized_pnl_pct', 0):+.1f}%)")

            elif order["action"] in ("buy_notional", "sell_notional"):
                # Fractional notional order
                resp = alpaca_request("POST", "v2/orders", mc, {
                    "symbol": alpaca_sym,
                    "notional": round(order["notional"], 2),
                    "side": order["side"],
                    "type": "market",
                    "time_in_force": "day",
                }, logger=logger)
                order_status = resp.get("status", "submitted")
                trade.order_id = resp.get("id")
                trade.order_status = order_status

                if order_status in ("rejected", "canceled", "expired"):
                    logger.error(
                        f"    REJECTED {order['trade_action'].upper():16s} {sym:6s}: "
                        f"{order['side']} ${order['notional']:,.2f}, "
                        f"status={order_status}, "
                        f"reason: {resp.get('reject_reason', 'unknown')}"
                    )
                    trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                    report.add_error(f"Order {order_status}: {order['trade_action']} {sym} - {resp.get('reject_reason', '')}")
                    failed += 1
                else:
                    order_ok = True
                    filled_qty = resp.get("filled_qty")
                    filled_str = f", filled_qty={filled_qty}" if filled_qty and filled_qty != "0" else ""
                    logger.info(
                        f"    OK  {order['trade_action'].upper():16s} {sym:6s}: "
                        f"{order['side']} ${order['notional']:,.2f}, "
                        f"order_id={resp.get('id', '?')}, "
                        f"status={order_status}{filled_str}"
                    )

            if order_ok:
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

        # Poll for final order status (fills, price, qty)
        if trade.order_id and trade.order_status not in ("failed", "rejected", "canceled", "expired"):
            try:
                final = _poll_order_status(trade.order_id, mc, logger)
                trade.order_status = final.get("status", trade.order_status)
                filled_qty = final.get("filled_qty")
                filled_price = final.get("filled_avg_price")
                if filled_qty and filled_qty != "0":
                    trade.shares = float(filled_qty)
                if filled_price and filled_price != "0":
                    trade.fill_price = float(filled_price)
            except Exception:
                pass  # keep whatever status we already have

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
# CUT-LOSS SCANNER — runs every minute during market hours for V7+
# ═══════════════════════════════════════════════════════════════════════════

# Cut-loss state (peak prices, daily portfolio anchor, portfolio-stop trip flag)
# is persisted per-model in mc.state_path. A single lock serialises read-modify-
# write windows inside cutloss_scan so back-to-back scheduler ticks don't race.
_cutloss_state_lock = threading.Lock()


# Market hours / holiday calendar — see core/market.py.
from core.market import US_MARKET_HOLIDAYS, is_market_open as _is_market_open  # noqa: E402


def _get_cutloss_logger():
    """Get or create a cutloss logger with file handler.

    Uses a single rotating log file (cutloss_YYYYMMDD.log) per day so that
    cutloss events are visible via the /api/logs endpoint.
    """
    logger = logging.getLogger("cutloss")
    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Daily log file — accessible via dashboard log viewer
        log_file = LOG_DIR / f"cutloss_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
    else:
        # Rotate file handler if the day changed
        today_suffix = datetime.now().strftime('%Y%m%d')
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if today_suffix not in str(h.baseFilename):
                    logger.removeHandler(h)
                    h.close()
                    log_file = LOG_DIR / f"cutloss_{today_suffix}.log"
                    fmt = logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    new_fh = logging.FileHandler(log_file)
                    new_fh.setFormatter(fmt)
                    logger.addHandler(new_fh)
                break
    return logger


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

    logger = _get_cutloss_logger()

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
    """Run cut-loss checks for a single model.

    Cut-loss state (trip flag, daily anchor, peak prices) is persisted to
    mc.state_path so it survives Railway redeploys and can be inspected /
    reasoned about. Once portfolio_stop fires, the trip flag blocks any
    further scans (and any rebalance) for the rest of the day.
    """
    today_iso = datetime.now().strftime("%Y-%m-%d")

    with _cutloss_state_lock:
        state = load_state(mc)

        # Trip-flag short-circuit: portfolio_stop already fired today → cash mode.
        if state.get("portfolio_stop_tripped_date") == today_iso:
            return

        # Fetch positions
        try:
            positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
        except Exception as e:
            logger.warning(f"[CUTLOSS] {mc.name}: failed to fetch positions: {e}")
            return

        if not positions:
            return

        n_positions = len(positions)

        # Fetch account for portfolio-level stop
        try:
            acct = alpaca_request("GET", "v2/account", mc, logger=logger)
            current_equity = float(acct.get("equity", 0))
            last_equity = float(acct.get("last_equity", 0))
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: failed to fetch account: {e}")
            return

        # Initialise / refresh the daily anchor on date change. last_equity
        # is yesterday's close from Alpaca and is stable all day, so any
        # intraday restart still measures drawdown from the same baseline.
        state_dirty = False
        if state.get("daily_portfolio_start_date") != today_iso:
            anchor = last_equity if last_equity > 0 else current_equity
            if anchor > 0:
                state["daily_portfolio_start"] = anchor
                state["daily_portfolio_start_date"] = today_iso
                state_dirty = True
        start_equity = state.get("daily_portfolio_start", last_equity)

        # ── Portfolio-level soft tiered scaler ───────────────────
        # Three tiers anchored to the configured portfolio_stop:
        #   Tier 1 (DD ≤ pstop):       scale to 60% exposure
        #   Tier 2 (DD ≤ pstop·5/3):   scale to 30% exposure
        #   Tier 3 (DD ≤ pstop·7/3):   liquidate to 0% + trip flag (hard stop)
        # Replaces the prior single hard threshold that triggered repeatedly
        # (2026-05-01, 2026-05-07) and forced sells at the worst intraday
        # point. Tier 1/2 scale down pro-rata without setting the trip flag,
        # so the model can re-enter on the next scheduled rebalance.
        if start_equity > 0 and current_equity > 0:
            daily_drawdown_pct = (current_equity / start_equity - 1) * 100
            pstop = float(mc.cutloss_portfolio_stop)  # negative, e.g. -3.0

            if daily_drawdown_pct <= pstop * (7.0 / 3.0):
                # Tier 3 — hard liquidation, preserves original behavior.
                logger.warning(
                    f"[CUTLOSS] {mc.name}: PORTFOLIO STOP TIER 3! "
                    f"Daily drawdown: {daily_drawdown_pct:.2f}% <= "
                    f"{pstop * 7.0 / 3.0:.2f}%. "
                    f"Liquidating ALL {n_positions} positions "
                    f"(equity=${current_equity:,.2f})."
                )
                state["portfolio_stop_tripped_date"] = today_iso
                state["peak_prices"] = {}
                state["last_rebalance"] = None
                save_state(state, mc)
                _liquidate_all(mc, positions, "portfolio_stop", logger)
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SUMMARY — liquidated all "
                    f"{n_positions} positions, 0/{n_positions} remaining, "
                    f"equity=${current_equity:,.2f}"
                )
                return

            elif daily_drawdown_pct <= pstop * (5.0 / 3.0):
                # Tier 2 — scale to 30% gross exposure. No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, current_equity,
                    target_exposure=0.30,
                    tier="Tier2", dd_pct=daily_drawdown_pct, logger=logger,
                )
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 2 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop * 5.0 / 3.0:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 30% exposure. "
                    f"Trip flag NOT set; trailing stops still active."
                )
                # Fall through — remaining trailing stops still apply.

            elif daily_drawdown_pct <= pstop:
                # Tier 1 — scale to 60% gross exposure. No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, current_equity,
                    target_exposure=0.60,
                    tier="Tier1", dd_pct=daily_drawdown_pct, logger=logger,
                )
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 1 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 60% exposure. "
                    f"Trip flag NOT set."
                )
                # Fall through — remaining trailing stops still apply.

        # ── Per-position stop checks ─────────────────────────────
        peak_prices = state.setdefault("peak_prices", {})
        held_symbols = {p["symbol"] for p in positions}
        # Prune peaks for positions no longer held.
        for sym in list(peak_prices.keys()):
            if sym not in held_symbols:
                del peak_prices[sym]
                state_dirty = True

        symbols_to_sell = []

        for p in positions:
            sym = p["symbol"]
            qty = float(p.get("qty", 0))
            avg_entry = float(p.get("avg_entry_price", 0))
            current_price = float(p.get("current_price", 0))

            if avg_entry <= 0 or current_price <= 0 or qty <= 0:
                continue

            prev_peak = peak_prices.get(sym, avg_entry)
            current_peak = max(prev_peak, current_price)
            if peak_prices.get(sym) != current_peak:
                peak_prices[sym] = current_peak
                state_dirty = True

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
        sold_symbols = []
        for sym, qty, reason, pct in symbols_to_sell:
            try:
                _execute_cutloss_sell(mc, sym, qty, reason, pct, logger)
                sold_symbols.append(sym)
                if peak_prices.pop(sym, None) is not None:
                    state_dirty = True
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to sell {sym}: {e}")

        if state_dirty:
            save_state(state, mc)

    # Summary after sells (outside the lock — pure logging + redistribute)
    if sold_symbols:
        remaining = n_positions - len(sold_symbols)
        logger.info(
            f"[CUTLOSS] {mc.name}: SUMMARY — sold {len(sold_symbols)} "
            f"({', '.join(sold_symbols)}), {remaining}/{n_positions} positions remaining, "
            f"equity=${current_equity:,.2f}"
        )

        # Redistribute freed cash into remaining positions (hard/trailing stops only)
        try:
            _redistribute_after_cutloss(mc, sold_symbols, logger)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: redistribution failed: {e}")


def _redistribute_after_cutloss(mc: ModelConfig, sold_symbols: list[str],
                                 logger):
    """Replace sold positions with next-best stocks from the model's latest predictions.

    After a hard/trailing stop sells a position:
    1. Look up the model's last predictions from state
    2. Pick replacement stocks (ranked candidates not currently held)
    3. Buy replacements with equal share of freed cash
    4. Distribute any leftover proportionally across remaining positions
    """
    import time as _time
    # Wait briefly for sells to settle
    _time.sleep(2)

    # Fetch fresh positions and account
    try:
        positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
        account = alpaca_request("GET", "v2/account", mc, logger=logger)
    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to fetch positions/account: {e}")
        return

    # Skip-guard: if today's drawdown is already within 1.5 percentage points
    # of the portfolio_stop trip threshold, redistributing more cash into the
    # sinking portfolio is counterproductive — the next 60s scan will likely
    # liquidate it anyway. Widened from 1.0pp to 1.5pp after the
    # 2026-05-01 / 2026-05-07 cascades showed the brake firing too late.
    try:
        current_eq = float(account.get("equity", 0))
        last_eq = float(account.get("last_equity", 0))
    except (TypeError, ValueError):
        current_eq = last_eq = 0
    if current_eq > 0 and last_eq > 0:
        drawdown_pct = (current_eq / last_eq - 1) * 100
        skip_threshold = mc.cutloss_portfolio_stop + 1.5  # e.g. -3.0 + 1.5 = -1.5
        if drawdown_pct <= skip_threshold:
            logger.warning(
                f"[REDISTRIBUTE] {mc.name}: SKIPPED — daily drawdown {drawdown_pct:+.2f}% "
                f"is within 1.5pp of portfolio_stop ({mc.cutloss_portfolio_stop:+.1f}%); "
                f"funnelling cash into a sinking portfolio is counterproductive."
            )
            return

    # Calculate available cash to redistribute
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    available = min(cash, buying_power)

    # Keep a small buffer (1% of equity) to avoid over-allocating
    equity = float(account.get("equity", 0))
    buffer = equity * 0.01
    available = max(0, available - buffer)

    if available < 50:
        logger.info(f"[REDISTRIBUTE] {mc.name}: only ${available:.2f} available, skipping")
        return

    # Current held symbols (after sells)
    held_symbols = set()
    for p in positions:
        sym = p["symbol"]
        if sym not in sold_symbols:
            held_symbols.add(sym)

    n_held = len(held_symbols)
    n_to_replace = len(sold_symbols)

    # Load last predictions from model state to find replacement candidates
    state = load_state(mc)
    history = state.get("history", [])
    replacements = []

    if history:
        latest = history[-1]
        predictions = latest.get("predictions", {})
        # predictions is {symbol: score}, sorted by score descending
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Also check a wider set: target_symbols from last rebalance
        target_syms = set(latest.get("target_symbols", []))

        # Find candidates: in predictions, not currently held, not just sold
        sold_set = set(sold_symbols)
        for sym, score in ranked:
            if sym in held_symbols or sym in sold_set:
                continue
            replacements.append((sym, score))
            if len(replacements) >= n_to_replace:
                break

        # If not enough from top predictions, look beyond TOP_N in target_symbols
        if len(replacements) < n_to_replace:
            for sym in target_syms:
                if sym in held_symbols or sym in sold_set:
                    continue
                if sym not in [r[0] for r in replacements]:
                    replacements.append((sym, 0))
                    if len(replacements) >= n_to_replace:
                        break

    if replacements:
        logger.info(f"[REDISTRIBUTE] {mc.name}: replacing {n_to_replace} sold position(s) "
                    f"with {len(replacements)} new: {', '.join(r[0] for r in replacements)}")
    else:
        logger.info(f"[REDISTRIBUTE] {mc.name}: no replacement candidates found in predictions, "
                    f"distributing into {n_held} existing positions")

    # Allocate cash: equal-weight for replacements, remainder spread across existing
    journal = TradeJournal(mc.name)
    n_bought = 0

    if replacements:
        # Give each replacement an equal share of the freed cash
        n_targets = len(replacements) + n_held
        replacement_alloc = available / max(n_targets, 1)

        for sym, score in replacements:
            alloc = round(replacement_alloc, 2)
            if alloc < 10:
                continue
            n_bought += _place_redistribute_buy(
                mc, sym, alloc, "replacement", score, journal, logger
            )

        # Remaining cash goes proportionally to existing positions
        used = replacement_alloc * len(replacements)
        leftover = available - used
    else:
        leftover = available

    # NOTE: the prior implementation topped up existing positions with any
    # leftover cash here. That created a "redistribute death spiral" on
    # falling days — sold position frees cash → top up survivors at higher
    # cost basis → next survivor stops out → repeat — and triggered the
    # PORTFOLIO STOP on v7+v8 multiple times (2026-05-01, 2026-05-07).
    # Leftover cash is now deliberately held until the next scheduled
    # rebalance. Replacement (above) is preserved because it's model-driven.
    deployed = available - leftover
    if leftover > 50:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: ${leftover:.2f} held as cash until next "
            f"rebalance (topup-existing disabled to prevent procyclical averaging "
            f"down on losers)."
        )

    logger.info(f"[REDISTRIBUTE] {mc.name}: completed — {n_bought} buys placed, "
                f"${deployed:.2f} deployed (${leftover:.2f} held cash), "
                f"{n_held + len(replacements)}/{TOP_N} target positions")


def _place_redistribute_buy(mc: ModelConfig, symbol: str, alloc: float,
                             action: str, score: float,
                             journal, logger) -> int:
    """Place a single redistribution buy order. Returns 1 on success, 0 on failure."""
    order_data = {
        "symbol": _to_alpaca_symbol(symbol),
        "notional": str(alloc),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    try:
        result = alpaca_request("POST", "v2/orders", mc, data=order_data, logger=logger)
        order_id = result.get("id", "?")
        order_status = result.get("status", "submitted")
        label = f"NEW {symbol}" if action == "replacement" else f"TOP-UP {symbol}"
        logger.info(f"[REDISTRIBUTE] {mc.name}: {label} ${alloc:.2f}"
                    f"{f' (score={score:.4f})' if score else ''} → {order_status}")

        # Poll for fill
        fill_price = None
        filled_qty = 0
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = _poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
            except Exception:
                pass

        record = TradeRecord(
            trade_id=f"{mc.name}_redist_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="buy",
            action=f"redistribute_{action}",
            order_type="market",
            time_in_force="day",
            notional_usd=alloc,
            order_status=order_status,
            order_id=order_id,
            shares=filled_qty,
            fill_price=fill_price,
        )
        journal.log_trade(record)
        return 1

    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to buy {symbol}: {e}")
        return 0


def _execute_cutloss_sell(mc: ModelConfig, symbol: str, qty: float,
                          reason: str, pct: float, logger):
    """Execute a market sell to close a position triggered by a cut-loss rule.

    Uses `DELETE /v2/positions/<symbol>` which closes the entire position
    atomically — handles fractional shares cleanly and never half-fills.
    """
    logger.info(f"[CUTLOSS] {mc.name}: SELLING {symbol} qty={qty:.2f} "
                f"reason={reason} ({pct:.2f}%)")

    alpaca_sym = _to_alpaca_symbol(symbol)
    try:
        result = alpaca_request(
            "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger,
        ) or {}
        order_id = result.get("id", "?")
        order_status = result.get("status", "accepted")

        if order_status in ("rejected", "canceled", "expired"):
            logger.error(f"[CUTLOSS] {mc.name}: {symbol} close order {order_status}: "
                         f"{result.get('reject_reason', 'unknown')}")
        else:
            logger.info(f"[CUTLOSS] {mc.name}: {symbol} close order placed: {order_id}, status={order_status}")

        # Poll for final order status before journaling
        fill_price = None
        filled_qty = qty
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = _poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
                logger.info(f"[CUTLOSS] {mc.name}: {symbol} final status={order_status}, "
                            f"filled_qty={filled_qty}, fill_price={fill_price}")
            except Exception:
                pass

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
            time_in_force="day",
            notional_usd=0,
            order_status=order_status,
            order_id=order_id,
            error_message=result.get("reject_reason") if order_status in ("rejected", "canceled", "expired") else None,
            shares=filled_qty,
            fill_price=fill_price,
        )
        journal.log_trade(record)
        # Peak-price tracking is owned by the caller (cutloss_scan_model uses
        # the per-model state dict; _liquidate_all wipes the whole dict before
        # invoking us). We deliberately do not touch it here.

    except Exception as e:
        logger.error(f"[CUTLOSS] {mc.name}: order failed for {symbol}: {e}")
        raise


def _liquidate_all(mc: ModelConfig, positions: list, reason: str, logger):
    """Liquidate all positions for a model (portfolio-level stop).

    The caller is responsible for wiping `state["peak_prices"]` before we
    place sells — that way the trip flag and an empty peak dict are
    persisted atomically even if a sell fails midway.
    """
    logger.warning(f"[CUTLOSS] {mc.name}: LIQUIDATING ALL {len(positions)} positions ({reason})")

    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("qty", 0))
        if qty > 0:
            try:
                _execute_cutloss_sell(mc, sym, qty, reason, 0.0, logger)
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to liquidate {sym}: {e}")


def _soft_scale_portfolio(mc: ModelConfig, positions: list,
                          current_equity: float, target_exposure: float,
                          tier: str, dd_pct: float, logger) -> int:
    """Scale gross exposure down to `target_exposure` of equity by selling
    pro-rata across positions. Used by Tier 1 and Tier 2 of the soft
    portfolio stop.

    Each position is partial-sold using `notional` market orders so that
    weights stay roughly proportional to entry weights (no "let the winners
    run" bias against the model's intended portfolio shape).

    Returns the count of positions partially-or-fully sold.
    """
    if not positions:
        return 0
    total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
    if total_mv <= 0:
        return 0
    target_mv = current_equity * target_exposure
    excess_mv = total_mv - target_mv
    if excess_mv <= 0:
        return 0

    fraction_to_sell = min(max(excess_mv / total_mv, 0.0), 1.0)
    journal = TradeJournal(mc.name)
    n_sold = 0
    reason = f"soft_scale_{tier.lower()}"

    for p in positions:
        sym = p["symbol"]
        mv = float(p.get("market_value", 0) or 0)
        if mv <= 0:
            continue
        sell_notional = round(mv * fraction_to_sell, 2)
        if sell_notional < 5.0:
            continue  # skip dust slices
        alpaca_sym = _to_alpaca_symbol(sym)
        try:
            resp = alpaca_request(
                "POST", "v2/orders", mc,
                data={
                    "symbol": alpaca_sym,
                    "notional": sell_notional,
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                },
                logger=logger,
            ) or {}
            status = resp.get("status", "submitted")
            logger.info(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell {sym} "
                f"${sell_notional:.2f} ({fraction_to_sell*100:.1f}% of "
                f"position) reason={reason} status={status}"
            )
            try:
                ts = datetime.now(timezone.utc)
                journal.log_trade(TradeRecord(
                    trade_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                    run_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d')}",
                    model=mc.name,
                    timestamp=ts.isoformat(),
                    symbol=sym,
                    side="sell",
                    action=reason,
                    order_type="market",
                    time_in_force="day",
                    notional_usd=sell_notional,
                    order_id=resp.get("id"),
                    order_status=status,
                    current_price=float(p.get("current_price", 0) or 0),
                ))
            except Exception:
                pass  # journal failure shouldn't block the sell
            n_sold += 1
        except Exception as e:
            logger.error(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell failed for {sym}: {e}"
            )

    logger.info(
        f"[CUTLOSS] {mc.name}: {tier} scaled {n_sold} positions pro-rata "
        f"(sold ~{fraction_to_sell*100:.1f}% of each), DD was {dd_pct:+.2f}%, "
        f"target_exposure={target_exposure*100:.0f}%."
    )
    return n_sold


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
