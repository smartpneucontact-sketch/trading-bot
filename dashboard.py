#!/usr/bin/env python3
"""Live dashboard for the ML trading bot with multi-model support.

Runs 24/7 on Railway. Serves a web UI showing:
- Live portfolio value & account summary per model (from Alpaca)
- Live positions with unrealized P&L
- Equity curve (portfolio history)
- Recent trades from the trade journal
- Bot status, run history, logs
- Manual trigger buttons

Also runs the trading pipeline on schedule via APScheduler.

Usage:
    python dashboard.py              # Start dashboard + scheduler
    python dashboard.py --port 8080  # Custom port
"""

import functools
import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from flask import Flask, jsonify, Response, request as flask_request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import pipeline

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

PORT = int(os.environ.get("PORT", 8080))
TZ = os.environ.get("TZ", "America/New_York")
BASE_DIR = Path(__file__).parent

DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"
TRADE_DIR = DATA_DIR / "trades"

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_DIR.mkdir(parents=True, exist_ok=True)

# Auth token for state-changing endpoints (set DASHBOARD_AUTH_TOKEN env var)
import secrets as _secrets
DASHBOARD_AUTH_TOKEN = os.environ.get("DASHBOARD_AUTH_TOKEN", None)


def _check_auth() -> bool:
    """Verify Authorization header matches expected token.
    Returns True if no token is configured (local dev) or if token matches."""
    if not DASHBOARD_AUTH_TOKEN:
        return True  # No token set; skip auth for local dev
    token = flask_request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return _secrets.compare_digest(token, DASHBOARD_AUTH_TOKEN) if token else False


def _validate_model_name(model_name: str) -> bool:
    """Validate that model_name is a known registered model."""
    if not model_name or not isinstance(model_name, str):
        return False
    if not all(c.isalnum() for c in model_name):
        return False
    return model_name in pipeline.MODEL_REGISTRY

# In-memory store for live status
bot_status = {
    "state": "idle",
    "last_run_at": None,
    "last_run_status": None,
    "last_run_duration": None,
    "last_error": None,
    "next_run_at": None,
    "current_step": None,
    "started_at": datetime.now().isoformat(),
    "total_runs": 0,
    "models": {},
}
status_lock = threading.Lock()

# Simple cache for Alpaca data (avoid hammering API)
from collections import OrderedDict
_api_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
CACHE_TTL = 30  # seconds
MAX_CACHE_SIZE = 500  # Maximum entries before LRU eviction


def _cached_api_call(key, fn, ttl=CACHE_TTL):
    """Cache API responses for ttl seconds with LRU eviction."""
    with _cache_lock:
        if key in _api_cache:
            data, ts = _api_cache[key]
            if time.time() - ts < ttl:
                _api_cache.move_to_end(key)
                return data
    try:
        result = fn()
        with _cache_lock:
            _api_cache[key] = (result, time.time())
            _api_cache.move_to_end(key)
            while len(_api_cache) > MAX_CACHE_SIZE:
                _api_cache.popitem(last=False)
        return result
    except Exception as e:
        logging.getLogger("dashboard").warning(f"API call failed ({key}): {e}")
        # Return stale cache if available
        with _cache_lock:
            if key in _api_cache:
                return _api_cache[key][0]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_state_path(model_name: str) -> Path:
    return STATE_DIR / f"pipeline_state_{model_name}.json"


def _load_model_state(model_name: str) -> dict:
    state_path = _get_state_path(model_name)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception as e:
            logging.getLogger("dashboard").warning(f"Failed to load {model_name} state: {e}")
    return {"run_count": 0, "history": []}


def _initialize_model_status():
    global bot_status
    with status_lock:
        for mc in pipeline.get_active_models():
            if mc.name not in bot_status["models"]:
                bot_status["models"][mc.name] = {
                    "state": "idle",
                    "run_count": 0,
                    "last_run_at": None,
                    "last_run_status": None,
                    "last_run_duration": None,
                }


def _get_model_config(model_name: str):
    """Get ModelConfig by name."""
    for mc in pipeline.get_active_models():
        if mc.name == model_name:
            return mc
    return None


def _load_recent_trades(model_name: str, limit: int = 50) -> list:
    """Load recent trades from the JSONL journal."""
    journal_path = TRADE_DIR / f"trades_{model_name}.jsonl"
    if not journal_path.exists():
        return []
    try:
        lines = journal_path.read_text().strip().split("\n")
        trades = []
        for line in lines[-limit:]:
            if line.strip():
                trades.append(json.loads(line))
        return list(reversed(trades))  # newest first
    except Exception as e:
        logging.getLogger("dashboard").warning(f"Failed to load trades for {model_name}: {e}")
        return []


def _fetch_alpaca_orders(mc, limit: int = 200) -> list:
    """Fetch recent orders from Alpaca API, cached for 60s."""
    def _fetch():
        logger = logging.getLogger("dashboard")
        orders = pipeline.alpaca_request(
            "GET",
            f"v2/orders?status=all&limit={limit}&direction=desc",
            mc, logger=logger
        )
        return orders if isinstance(orders, list) else []
    return _cached_api_call(f"orders_all_{mc.name}", _fetch, ttl=60) or []


def _load_trades_merged(model_name: str, limit: int = 50) -> list:
    """Load trades merged from journal + Alpaca order history.

    - Journal entries get updated with Alpaca fill info (status, price, qty).
    - Alpaca orders not in journal (e.g. pre-fix cutloss sells) are backfilled.
    - Deduplicates by order_id AND by (symbol, side, shares) within same day.
    - Normalises field names for the dashboard UI (filled_price, notional).
    """
    journal_trades = _load_recent_trades(model_name, limit=500)  # load more for matching
    mc = _get_model_config(model_name)

    # Normalise field names on journal entries for the dashboard UI
    def _normalise(t):
        # filled_price: UI reads t.filled_price
        if "fill_price" in t and "filled_price" not in t:
            t["filled_price"] = t.pop("fill_price")
        # notional: UI reads t.notional
        if "notional_usd" in t and "notional" not in t:
            t["notional"] = t.pop("notional_usd")
        # side: some old entries use trade_action instead of side
        if "trade_action" in t and "side" not in t:
            t["side"] = t.pop("trade_action")
        return t

    def _trade_day(t):
        """Extract date portion from timestamp for same-day dedup."""
        ts = t.get("filled_at") or t.get("timestamp") or ""
        return ts[:10]  # "2026-04-15T13:38:..." -> "2026-04-15"

    def _dedup_key(t):
        """Key for deduplicating: same symbol + side + ~shares on same day."""
        shares = t.get("shares") or 0
        try:
            shares = round(float(shares), 1)
        except (ValueError, TypeError):
            shares = 0
        return (_trade_day(t), t.get("symbol", ""), t.get("side", ""), shares)

    if not mc:
        return [_normalise(t) for t in journal_trades[:limit]]

    # Fetch Alpaca order history
    alpaca_orders = _fetch_alpaca_orders(mc, limit=500)
    if not alpaca_orders:
        return [_normalise(t) for t in journal_trades[:limit]]

    # Index Alpaca orders by order_id for fast lookup
    alpaca_by_id = {}
    for o in alpaca_orders:
        oid = o.get("id")
        if oid:
            alpaca_by_id[oid] = o

    # Track which Alpaca order_ids are covered by journal entries
    journal_order_ids = set()

    # Merge: update journal entries with Alpaca fill data
    seen_order_ids = {}  # order_id -> index in merged list
    seen_dedup = {}      # dedup_key -> index in merged list (for cross-source dedup)
    merged = []
    for t in journal_trades:
        oid = t.get("order_id")
        if oid and oid in alpaca_by_id:
            journal_order_ids.add(oid)
            ao = alpaca_by_id[oid]
            # Update status, fill price, filled qty from Alpaca
            t["order_status"] = ao.get("status", t.get("order_status", ""))
            fq = ao.get("filled_qty")
            fp = ao.get("filled_avg_price")
            if fq and float(fq or 0) > 0:
                t["shares"] = float(fq)
            if fp and float(fp or 0) > 0:
                t["filled_price"] = float(fp)
            filled_at = ao.get("filled_at")
            if filled_at:
                t["filled_at"] = filled_at
            # Compute notional from fill data
            if t.get("filled_price") and t.get("shares"):
                t["notional"] = round(t["shares"] * t["filled_price"], 2)

        # Dedup by order_id: skip if we already have a better entry
        if oid and oid in seen_order_ids:
            existing_idx = seen_order_ids[oid]
            existing = merged[existing_idx]
            if t.get("order_status") == "filled" and existing.get("order_status") != "filled":
                merged[existing_idx] = _normalise(t)
            continue

        nt = _normalise(t)

        # Dedup by (symbol, side, shares, day): if a "filled" version from
        # Alpaca backfill or a matched journal entry already exists, skip
        # the unmatched "submitted" journal entry.
        dk = _dedup_key(nt)
        if dk in seen_dedup:
            existing_idx = seen_dedup[dk]
            existing = merged[existing_idx]
            # Keep whichever has "filled" status
            if nt.get("order_status") == "filled" and existing.get("order_status") != "filled":
                merged[existing_idx] = nt
                if oid:
                    seen_order_ids[oid] = existing_idx
            continue

        idx = len(merged)
        if oid:
            seen_order_ids[oid] = idx
        seen_dedup[dk] = idx
        merged.append(nt)

    # Backfill: add Alpaca orders not in journal (cutloss sells that were
    # never journaled due to the journal.log() bug, plus any other missing orders)
    for o in alpaca_orders:
        oid = o.get("id")
        if oid in journal_order_ids:
            continue
        filled_qty = float(o.get("filled_qty", 0) or 0)
        filled_price = float(o.get("filled_avg_price", 0) or 0)
        notional = round(filled_qty * filled_price, 2) if filled_price else 0
        ts = o.get("filled_at") or o.get("submitted_at") or o.get("created_at", "")
        side = o.get("side", "")
        entry = {
            "timestamp": ts,
            "symbol": o.get("symbol", ""),
            "side": side,
            "action": "cutloss_backfill" if side == "sell" else "backfill",
            "order_type": o.get("type", "market"),
            "time_in_force": o.get("time_in_force", ""),
            "notional": notional,
            "order_status": o.get("status", ""),
            "order_id": oid,
            "shares": filled_qty,
            "filled_price": filled_price,
            "source": "alpaca_backfill",
        }

        # Dedup against journal entries by (symbol, side, shares, day)
        dk = _dedup_key(entry)
        if dk in seen_dedup:
            existing_idx = seen_dedup[dk]
            existing = merged[existing_idx]
            # Alpaca "filled" always wins over journal "submitted"
            if entry.get("order_status") == "filled" and existing.get("order_status") != "filled":
                merged[existing_idx] = entry
            continue

        seen_dedup[dk] = len(merged)
        merged.append(entry)

    # Clean up stale entries: if a journal entry still shows "pending_new" or
    # "submitted" and it's older than 1 hour, the status is stale (orders
    # resolve in seconds). Mark them so the UI shows the right info.
    from datetime import datetime, timezone, timedelta
    stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    stale_statuses = {"pending_new", "submitted", "accepted", "new"}
    for t in merged:
        if t.get("order_status") in stale_statuses:
            ts_str = t.get("filled_at") or t.get("timestamp") or ""
            try:
                # Parse ISO timestamp (handles both Z and +00:00 suffix)
                ts_clean = ts_str.replace("Z", "+00:00")
                if "+" not in ts_clean and ts_clean:
                    ts_clean += "+00:00"
                entry_time = datetime.fromisoformat(ts_clean)
                if entry_time < stale_cutoff:
                    t["order_status"] = "filled*"
            except (ValueError, TypeError):
                pass

    # Sort by timestamp descending
    def _ts_key(t):
        return t.get("filled_at") or t.get("timestamp") or ""
    merged.sort(key=_ts_key, reverse=True)

    return merged[:limit]


CUTLOSS_ACTIONS = {
    "hard_stop", "trailing_stop", "portfolio_stop",
    # Phase 1 soft tiered portfolio stop (partial sells at Tier 1 / Tier 2)
    "soft_scale_tier1", "soft_scale_tier2",
}


def _load_cutloss_events(model_name: str, limit: int = 100) -> list:
    """Load cutloss events from the trade journal + cutloss log files.

    Returns a merged list of events from both sources, newest first.
    Journal entries have full trade details; log entries have the trigger info.
    """
    events = []

    # Source 1: Trade journal (has structured data for sells after the fix)
    journal_path = TRADE_DIR / f"trades_{model_name}.jsonl"
    if journal_path.exists():
        try:
            lines = journal_path.read_text().strip().split("\n")
            for line in lines:
                if line.strip():
                    trade = json.loads(line)
                    if trade.get("action") in CUTLOSS_ACTIONS:
                        events.append({
                            "timestamp": trade.get("timestamp", ""),
                            "symbol": trade.get("symbol", ""),
                            "trigger": trade.get("action", ""),
                            "side": trade.get("side", "sell"),
                            "shares": trade.get("shares"),
                            "order_status": trade.get("order_status", ""),
                            "order_id": trade.get("order_id"),
                            "error_message": trade.get("error_message"),
                            "source": "journal",
                        })
        except Exception:
            pass

    # Source 2: Cutloss log files (has trigger details, pct from peak/entry)
    try:
        log_files = sorted(LOG_DIR.glob("cutloss_*.log"), reverse=True)[:30]
        for log_file in log_files:
            for raw_line in log_file.read_text().strip().split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                # Match cutloss trigger lines like:
                #   [CUTLOSS] v7: HARD STOP on IPGP! -8.23% from entry ...
                #   [CUTLOSS] v7: TRAILING STOP on TEAM! -5.12% from peak ...
                #   [CUTLOSS] v7: SELLING IPGP qty=57.56 reason=hard_stop (-8.23%)
                #   [CUTLOSS] v7: PORTFOLIO STOP triggered! ...
                if f"[CUTLOSS] {model_name}:" not in line:
                    continue
                # Extract timestamp from log line prefix (YYYY-MM-DD HH:MM:SS)
                ts = line[:19] if len(line) >= 19 else ""
                if "HARD STOP on" in line or "TRAILING STOP on" in line:
                    # Parse: [CUTLOSS] v7: TRAILING STOP on TEAM! -5.12% from peak $72.50
                    parts = line.split("STOP on ")
                    if len(parts) >= 2:
                        rest = parts[1]  # "TEAM! -5.12% from peak $72.50 ..."
                        sym = rest.split("!")[0].strip()
                        trigger = "trailing_stop" if "TRAILING" in line else "hard_stop"
                        pct_str = rest.split("!")[1].strip() if "!" in rest else ""
                        # Deduplicate: skip if we already have a journal entry for same model+symbol+timestamp
                        already_in = any(
                            e["symbol"] == sym and e["source"] == "journal"
                            and e["timestamp"][:16] == ts[:16]
                            for e in events
                        )
                        if not already_in:
                            events.append({
                                "timestamp": ts,
                                "symbol": sym,
                                "trigger": trigger,
                                "detail": pct_str,
                                "side": "sell",
                                "source": "log",
                            })
                elif "PORTFOLIO STOP triggered" in line:
                    events.append({
                        "timestamp": ts,
                        "symbol": "*ALL*",
                        "trigger": "portfolio_stop",
                        "detail": line.split("triggered!")[1].strip() if "triggered!" in line else "",
                        "side": "sell",
                        "source": "log",
                    })
                elif "SELLING" in line and "reason=" in line:
                    # Parse: [CUTLOSS] v7: SELLING IPGP qty=57.56 reason=hard_stop (-8.23%)
                    try:
                        after_selling = line.split("SELLING ")[1]
                        sym = after_selling.split()[0]
                        reason = ""
                        if "reason=" in after_selling:
                            reason = after_selling.split("reason=")[1].split()[0]
                        already_in = any(
                            e["symbol"] == sym and e["timestamp"][:16] == ts[:16]
                            for e in events
                        )
                        if not already_in:
                            events.append({
                                "timestamp": ts,
                                "symbol": sym,
                                "trigger": reason,
                                "side": "sell",
                                "source": "log",
                            })
                    except (IndexError, ValueError):
                        pass
    except Exception:
        pass

    # Sort newest first, deduplicate, limit
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_trading_pipeline(force=False, model_filter=None):
    global bot_status

    with status_lock:
        if bot_status["state"] == "running":
            logging.getLogger("dashboard").warning("Pipeline already running, skipping")
            return
        bot_status["state"] = "running"
        bot_status["current_step"] = "starting"
        if model_filter:
            if model_filter in bot_status["models"]:
                bot_status["models"][model_filter]["state"] = "running"
        else:
            for model_name in bot_status["models"]:
                bot_status["models"][model_name]["state"] = "running"

    start_time = time.time()

    try:
        pipeline.run_pipeline(dry_run=False, force=force, model_filter=model_filter)
        elapsed = time.time() - start_time

        with status_lock:
            bot_status["state"] = "idle"
            bot_status["last_run_at"] = datetime.now().isoformat()
            bot_status["last_run_status"] = "success"
            bot_status["last_run_duration"] = round(elapsed, 1)
            bot_status["last_error"] = None
            bot_status["current_step"] = None
            bot_status["total_runs"] += 1

            targets = [model_filter] if model_filter else list(bot_status["models"].keys())
            for mn in targets:
                if mn in bot_status["models"]:
                    state = _load_model_state(mn)
                    bot_status["models"][mn].update({
                        "state": "idle",
                        "run_count": state.get("run_count", 0),
                        "last_run_status": "success",
                        "last_run_at": datetime.now().isoformat(),
                        "last_run_duration": round(elapsed, 1),
                    })

    except Exception as e:
        elapsed = time.time() - start_time
        full_error = f"{e}\n{traceback.format_exc()}"
        # Sanitize: only store first line (exception message) in status,
        # log full traceback to server logs only
        safe_error = str(e).split("\n")[0][:500]

        with status_lock:
            bot_status["state"] = "error"
            bot_status["last_run_at"] = datetime.now().isoformat()
            bot_status["last_run_status"] = "error"
            bot_status["last_run_duration"] = round(elapsed, 1)
            bot_status["last_error"] = safe_error
            bot_status["current_step"] = None
            bot_status["total_runs"] += 1

            targets = [model_filter] if model_filter else list(bot_status["models"].keys())
            for mn in targets:
                if mn in bot_status["models"]:
                    bot_status["models"][mn]["state"] = "error"
                    bot_status["models"][mn]["last_run_status"] = "error"

        logging.getLogger("dashboard").error(f"Pipeline failed: {full_error}")


# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.errorhandler(500)
def handle_500(e):
    import traceback
    tb = traceback.format_exc()
    logging.getLogger("dashboard").error(f"500 error: {e}\n{tb}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    logging.getLogger("dashboard").error(f"Unhandled exception: {e}\n{tb}")
    return jsonify({"error": "Internal server error"}), 500


def _get_log_files() -> list[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_logs = list(LOG_DIR.glob("pipeline_*.log"))
    cutloss_logs = list(LOG_DIR.glob("cutloss_*.log"))
    return sorted(pipeline_logs + cutloss_logs, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_log(path: Path, tail: int = 200, level: str = None) -> tuple[str, int, int]:
    """Read a log file, optionally tailed and level-filtered.

    Returns (content, total_line_count, returned_line_count).
    `tail`: 0 or negative means "all lines".
    `level`: if set (ERROR | WARNING), keep only lines containing that token.
    """
    try:
        lines = path.read_text().splitlines()
        total = len(lines)
        if level:
            tag = f"[{level.upper()}]"
            lines = [ln for ln in lines if tag in ln]
        if tail and tail > 0 and len(lines) > tail:
            lines = lines[-tail:]
        return "\n".join(lines), total, len(lines)
    except Exception as e:
        return f"Error reading log: {e}", 0, 0


# ── API ENDPOINTS ─────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    with status_lock:
        s = dict(bot_status)
    s["models"] = {}
    for mc in pipeline.get_active_models():
        state = _load_model_state(mc.name)
        s["models"][mc.name] = {
            "run_count": state.get("run_count", 0),
            "history": state.get("history", [])[-5:],
            **bot_status.get("models", {}).get(mc.name, {})
        }
    return jsonify(s)


@app.route("/api/account/<model_name>")
def api_account(model_name):
    """Live account data from Alpaca."""
    mc = _get_model_config(model_name)
    if not mc:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    logger = logging.getLogger("dashboard")

    def fetch():
        acct = pipeline.alpaca_request("GET", "v2/account", mc, logger=logger)
        return {
            "equity": float(acct.get("equity", 0)),
            "portfolio_value": float(acct.get("portfolio_value", 0)),
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "last_equity": float(acct.get("last_equity", 0)),
            "long_market_value": float(acct.get("long_market_value", 0)),
            "short_market_value": float(acct.get("short_market_value", 0)),
            "initial_margin": float(acct.get("initial_margin", 0)),
            "status": acct.get("status", "unknown"),
            "currency": acct.get("currency", "USD"),
            "pattern_day_trader": acct.get("pattern_day_trader", False),
        }

    data = _cached_api_call(f"account_{model_name}", fetch)
    if data is None:
        return jsonify({"error": "Failed to fetch account"}), 500
    return jsonify(data)


@app.route("/api/positions/<model_name>")
def api_positions(model_name):
    """Live positions from Alpaca."""
    mc = _get_model_config(model_name)
    if not mc:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    logger = logging.getLogger("dashboard")

    def fetch():
        positions = pipeline.alpaca_request("GET", "v2/positions", mc, logger=logger)
        result = []
        for p in positions:
            result.append({
                "symbol": p["symbol"],
                "qty": float(p["qty"]),
                "market_value": float(p["market_value"]),
                "cost_basis": float(p.get("cost_basis", 0)),
                "unrealized_pl": float(p["unrealized_pl"]),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)) * 100,
                "current_price": float(p.get("current_price", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "change_today": float(p.get("change_today", 0)) * 100,
                "side": p["side"],
            })
        result.sort(key=lambda x: x["market_value"], reverse=True)
        return result

    data = _cached_api_call(f"positions_{model_name}", fetch)
    if data is None:
        return jsonify({"error": "Failed to fetch positions"}), 500
    return jsonify(data)


@app.route("/api/history/<model_name>")
def api_history_model(model_name):
    """Portfolio history from Alpaca (equity curve)."""
    mc = _get_model_config(model_name)
    if not mc:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    period = flask_request.args.get("period", "1M")  # 1D, 1W, 1M, 3M, 1A, all
    logger = logging.getLogger("dashboard")

    def fetch():
        hist = pipeline.alpaca_request(
            "GET",
            f"v2/account/portfolio/history?period={period}&timeframe=1D&extended_hours=false",
            mc, logger=logger
        )
        timestamps = hist.get("timestamp", [])
        equity = hist.get("equity", [])
        profit_loss = hist.get("profit_loss", [])
        profit_loss_pct = hist.get("profit_loss_pct", [])
        return {
            "timestamps": timestamps,
            "equity": equity,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
            "base_value": hist.get("base_value", 0),
        }

    data = _cached_api_call(f"history_{model_name}_{period}", fetch, ttl=120)
    if data is None:
        return jsonify({"error": "Failed to fetch portfolio history"}), 500
    return jsonify(data)


@app.route("/api/trades/<model_name>")
def api_trades(model_name):
    """Recent trades from journal merged with Alpaca order history."""
    try:
        limit = max(1, min(int(flask_request.args.get("limit", 50)), 1000))
    except (ValueError, TypeError):
        limit = 50
    trades = _load_trades_merged(model_name, limit)
    return jsonify(trades)


@app.route("/api/cutloss/<model_name>")
def api_cutloss(model_name):
    """Cutloss events for a model from journal + log files."""
    try:
        limit = max(1, min(int(flask_request.args.get("limit", 100)), 500))
    except (ValueError, TypeError):
        limit = 100
    events = _load_cutloss_events(model_name, limit)
    return jsonify(events)


# Trigger labels grouped by mechanism — Phase 1 added the soft tiered stop,
# so the dashboard now distinguishes per-position stops, soft scaling,
# and full liquidations.
_PER_POSITION_TRIGGERS = {"hard_stop", "trailing_stop"}
_SOFT_SCALE_TRIGGERS = {"soft_scale_tier1", "soft_scale_tier2"}
_PORTFOLIO_STOP_TRIGGERS = {"portfolio_stop"}


@app.route("/api/cutloss-stats/<model_name>")
def api_cutloss_stats(model_name):
    """Aggregate cutloss activity stats — events today / 7d / 30d.

    Reads the trade journal directly (more reliable than parsing logs)
    and produces counts + notional per window, broken down by trigger
    type (per-position stops vs soft-scale partial sells vs full
    portfolio liquidation).
    """
    if not _validate_model_name(model_name):
        return jsonify({"error": "invalid model"}), 400

    journal_path = TRADE_DIR / f"trades_{model_name}.jsonl"
    now = datetime.now()
    cutoffs = {
        "today":  now.replace(hour=0, minute=0, second=0, microsecond=0),
        "last_7d":  now - timedelta(days=7),
        "last_30d": now - timedelta(days=30),
    }

    def _empty_window() -> dict:
        return {
            "events": 0,
            "notional": 0.0,
            "per_position_stops": 0,
            "soft_scale_partials": 0,
            "portfolio_stops": 0,
            "by_trigger": {},
            "by_symbol": {},
        }

    windows = {name: _empty_window() for name in cutoffs}
    last_event = None
    total_events = 0

    if journal_path.exists():
        try:
            for raw in journal_path.read_text().strip().split("\n"):
                line = raw.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                except json.JSONDecodeError:
                    continue
                trigger = trade.get("action") or ""
                if trigger not in CUTLOSS_ACTIONS:
                    continue

                ts_raw = trade.get("timestamp") or ""
                try:
                    # Timestamps are ISO-8601 UTC; strip tzinfo for naive compare.
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                except (TypeError, ValueError):
                    continue

                notional = float(trade.get("notional_usd") or 0.0)
                # Backfill: legacy cutloss trades recorded notional_usd=0
                # because Alpaca's DELETE-position endpoint doesn't return
                # a notional. Reconstruct from shares × fill_price when
                # both are available.
                if notional == 0.0:
                    shares = trade.get("shares")
                    fill_price = trade.get("fill_price")
                    if shares and fill_price:
                        try:
                            notional = round(float(shares) * float(fill_price), 2)
                        except (TypeError, ValueError):
                            pass
                symbol = trade.get("symbol") or "?"
                total_events += 1
                if last_event is None or ts_raw > last_event["timestamp"]:
                    last_event = {
                        "timestamp": ts_raw,
                        "symbol": symbol,
                        "trigger": trigger,
                        "notional": notional,
                    }

                for name, cutoff in cutoffs.items():
                    if ts < cutoff:
                        continue
                    w = windows[name]
                    w["events"] += 1
                    w["notional"] += notional
                    w["by_trigger"][trigger] = w["by_trigger"].get(trigger, 0) + 1
                    w["by_symbol"][symbol] = w["by_symbol"].get(symbol, 0) + 1
                    if trigger in _PER_POSITION_TRIGGERS:
                        w["per_position_stops"] += 1
                    elif trigger in _SOFT_SCALE_TRIGGERS:
                        w["soft_scale_partials"] += 1
                    elif trigger in _PORTFOLIO_STOP_TRIGGERS:
                        w["portfolio_stops"] += 1
        except Exception as e:
            return jsonify({"error": f"failed to read journal: {e}"}), 500

    # Round notional + sort top symbols
    for name in windows:
        w = windows[name]
        w["notional"] = round(w["notional"], 2)
        # Top 5 most-stopped symbols in this window
        w["top_symbols"] = sorted(
            w["by_symbol"].items(), key=lambda kv: -kv[1]
        )[:5]

    return jsonify({
        "model": model_name,
        "total_events_all_time": total_events,
        "last_event": last_event,
        "windows": windows,
        "journal_path": str(journal_path),
        "journal_exists": journal_path.exists(),
    })


@app.route("/api/portfolio/<model_name>")
def api_portfolio_model(model_name):
    state = _load_model_state(model_name)
    history = state.get("history", [])
    if not history:
        return jsonify({"portfolio": [], "date": None})
    latest = history[-1]
    return jsonify({
        "date": latest.get("date"),
        "portfolio": latest.get("target_symbols", []),
        "predictions": latest.get("predictions", {}),
        "conviction_weights": latest.get("conviction_weights", {}),
        "regime_exposure": latest.get("regime_exposure"),
    })


# ── PERFORMANCE BENCHMARK ─────────────────────────────────────────────────

# Cache for universe benchmark (expensive; recompute at most every 10 min)
_perf_cache = {}
_perf_cache_lock = threading.Lock()
PERF_CACHE_TTL = 600  # 10 minutes


def _get_universe_symbols() -> list[str]:
    """Load universe symbols from the pipeline's cached symbol list."""
    cache_path = DATA_DIR / "universe_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return data.get("symbols", [])
        except Exception:
            pass
    # Fallback: try to scrape S&P 500
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
            storage_options={"User-Agent": "Mozilla/5.0"}
        )
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []


"""Performance is the model's *equity-curve* return measured over four windows
(1d, 7d, 30d, inception) compared to the same-window return of every stock
in the model's tradeable universe.

The headline metric is the **percentile rank** of the model's return within
the distribution of universe returns — "V6 returned +X% in 30d, beating Y%
of the universe." This captures stock-picking skill, execution timing, and
risk management in a single number that is comparable across windows."""

PERF_WINDOWS = [("1d", 1), ("7d", 7), ("30d", 30), ("inception", None)]
INDICATOR_HELP = {
    "model_return": (
        "Total return on the model's *equity curve* over the window. "
        "Includes realized P&L from closed positions, unrealized P&L on "
        "open positions, and cash drag. This is the actual money the "
        "account made or lost."
    ),
    "univ_n": (
        "Number of stocks in the tradeable universe that had price data "
        "for the full window. Stocks that IPO'd or delisted mid-window "
        "are excluded so returns are comparable."
    ),
    "univ_mean": "Simple average of every universe stock's window return.",
    "univ_median": (
        "Middle value of universe returns. Less skewed by extreme outliers "
        "than the mean — usually the better single-number summary of 'a "
        "typical stock.'"
    ),
    "univ_std": "Standard deviation of universe returns — measures spread.",
    "univ_pct_positive": "Share of the universe with a positive return over the window.",
    "univ_pctile": (
        "Where the model ranks within the universe distribution. 50 = beat "
        "exactly half the stocks. 90 = top decile. 10 = bottom decile. "
        "This is the 'how good was the pick relative to the alternatives' "
        "number."
    ),
    "spy_return": "Buy-and-hold SPY return over the same window.",
    "alpha_vs_median":
        "Model return minus universe median — outperformance vs the typical stock.",
    "alpha_vs_spy":
        "Model return minus SPY — outperformance vs the broad market.",
}


# Universe-bars cache: one fetch covers all windows for the day.
_universe_bars_cache: dict = {"start": None, "bars": None, "fetched_at": 0.0}
_universe_bars_lock = threading.Lock()
UNIV_BARS_TTL = 3600  # 1 hour — bars don't change intraday for 1Day timeframe


def _alpaca_data_get(mc, path: str, params: dict, timeout: int = 30):
    """Wrap a requests.get to Alpaca's market-data host using mc's keys."""
    import requests as _requests
    headers = {
        "APCA-API-KEY-ID": mc.alpaca_key,
        "APCA-API-SECRET-KEY": mc.alpaca_secret,
    }
    return _requests.get(
        f"https://data.alpaca.markets/{path}",
        headers=headers, params=params, timeout=timeout,
    )


# Cache the working data feed so we don't keep paying the 403 round-trip.
# Cleared if a feed stops working (HTTP 4xx) so the next call re-discovers.
_perf_active_feed: Optional[str] = None
_perf_feed_lock = threading.Lock()


def _alpaca_bars_with_fallback(mc, path: str, base_params: dict, logger=None):
    """Hit Alpaca with feed=sip → fall back to feed=iex on 401/403/422.

    Caches the working feed in `_perf_active_feed` so subsequent calls go
    straight to the working one. On any non-auth error, surface as-is.
    Returns (response, feed_used).
    """
    global _perf_active_feed
    with _perf_feed_lock:
        cached = _perf_active_feed
    feeds = [cached] if cached else ["sip", "iex"]
    last_resp = None
    for feed in feeds:
        params = {**base_params, "feed": feed}
        try:
            resp = _alpaca_data_get(mc, path, params, timeout=30)
        except Exception as e:
            if logger:
                logger.warning(f"[PERF] feed={feed} request failed: {e}")
            continue
        last_resp = resp
        if resp.status_code == 200:
            with _perf_feed_lock:
                if _perf_active_feed != feed:
                    _perf_active_feed = feed
                    if logger:
                        logger.info(f"[PERF] data feed selected: {feed}")
            return resp, feed
        if resp.status_code in (401, 403, 422):
            if logger:
                logger.info(
                    f"[PERF] feed={feed} unavailable (HTTP {resp.status_code}); trying next"
                )
            # If the cached feed just failed, drop it so the next attempt
            # re-discovers from the full feed list.
            with _perf_feed_lock:
                if _perf_active_feed == feed:
                    _perf_active_feed = None
            continue
        # Non-auth HTTP error — return so caller can log + bail
        return resp, feed
    return last_resp, (feeds[-1] if feeds else "iex")


def _to_alpaca_symbol(s: str) -> str:
    """Convert yfinance-style class shares (BRK-B) to Alpaca-style (BRK.B).

    The pipeline stores the universe in yfinance format because that's what
    yfinance expects for class shares. Alpaca's data API uses a dot for the
    same names. Pattern: ends with `-<single uppercase ASCII letter>`.
    """
    if (
        len(s) >= 3
        and s[-2] == "-"
        and s[-1].isascii()
        and s[-1].isalpha()
        and s[-1].isupper()
    ):
        return s[:-2] + "." + s[-1]
    return s


def _fetch_bars_paged(mc, symbols: list, start_iso: str, logger=None) -> tuple:
    """Fetch daily bars for `symbols` since `start_iso`.

    - Translates yfinance-format class shares (BRK-B) to Alpaca format (BRK.B)
      before sending to the API; results are mapped back to the original keys.
    - Uses the SIP feed when available, falling back to IEX.
    - Batches to stay under Alpaca's per-request symbol cap and paginates on
      `next_page_token`.
    - Logs a coverage summary so silent drops are visible in the logs UI.

    Returns (bars_by_original_symbol, feed_used).
    """
    out: dict = {}
    BATCH = 180  # well under Alpaca's documented multi-symbol cap

    # original → alpaca and alpaca → original mappings
    alpaca_to_original = {_to_alpaca_symbol(s): s for s in symbols}
    alpaca_syms = list(alpaca_to_original.keys())

    feed_used = None
    for i in range(0, len(alpaca_syms), BATCH):
        batch = alpaca_syms[i:i + BATCH]
        page_token = None
        pages = 0
        while True:
            base_params = {
                "symbols": ",".join(batch),
                "start": start_iso,
                "timeframe": "1Day",
                "limit": 10000,
                "adjustment": "split",
            }
            if page_token:
                base_params["page_token"] = page_token

            resp, feed = _alpaca_bars_with_fallback(
                mc, "v2/stocks/bars", base_params, logger=logger,
            )
            feed_used = feed

            if resp is None or resp.status_code != 200:
                if logger:
                    code = resp.status_code if resp is not None else "no-response"
                    body = resp.text[:200] if resp is not None else ""
                    logger.warning(
                        f"[PERF] universe bars batch {i} feed={feed}: HTTP {code} {body}"
                    )
                break

            data = resp.json()
            for alpaca_sym, blist in (data.get("bars") or {}).items():
                if not blist:
                    continue
                original = alpaca_to_original.get(alpaca_sym, alpaca_sym)
                out.setdefault(original, []).extend(blist)
            page_token = data.get("next_page_token")
            pages += 1
            if not page_token or pages > 10:
                break

    if logger:
        got = set(out.keys())
        missing = [s for s in symbols if s not in got]
        sample = ", ".join(missing[:20]) if missing else ""
        logger.info(
            f"[PERF] Universe coverage on feed={feed_used}: "
            f"{len(out)}/{len(symbols)} symbols got bars"
            + (f" ({len(missing)} missing — first 20: {sample})" if missing else "")
        )
    return out, feed_used


def _get_universe_bars(mc, syms: list, lookback_days: int, logger) -> tuple:
    """Get universe bars covering at least `lookback_days`, with caching.

    Returns (bars_by_symbol, feed_used).
    """
    # Pad start a few days for weekends/holidays to ensure we have a bar
    # before each window's nominal start.
    start_dt = datetime.now() - timedelta(days=lookback_days + 5)
    start_iso = start_dt.strftime("%Y-%m-%d")

    with _universe_bars_lock:
        cache = _universe_bars_cache
        fresh = (
            cache["bars"] is not None
            and cache["start"] == start_iso
            and time.time() - cache["fetched_at"] < UNIV_BARS_TTL
        )
        if fresh:
            return cache["bars"], cache.get("feed", "iex")

    # Fetch outside the lock — Alpaca calls take seconds. We accept the risk
    # that two requests both fetch; the second will simply overwrite.
    logger.info(
        f"[PERF] Fetching universe bars: {len(syms)} symbols since {start_iso}"
    )
    t0 = time.time()
    bars, feed = _fetch_bars_paged(mc, syms, start_iso, logger=logger)
    logger.info(
        f"[PERF] Universe fetch finished in {time.time() - t0:.1f}s "
        f"(feed={feed})"
    )

    with _universe_bars_lock:
        _universe_bars_cache.update({
            "start": start_iso, "bars": bars, "fetched_at": time.time(),
            "feed": feed,
        })
    return bars, feed


def _bar_return_in_window(bars: list, window_start_ms: int) -> Optional[float]:
    """% return for a symbol from the first bar at-or-after window_start_ms
    to the most recent bar.  None if the symbol lacks data in the window."""
    if not bars:
        return None
    # Bars are sorted by `t` (ISO string). Find first bar at or after the
    # window start.
    first = None
    for b in bars:
        try:
            ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000
        except Exception:
            continue
        if ts >= window_start_ms:
            first = b
            break
    if first is None:
        return None
    last = bars[-1]
    try:
        entry = float(first["o"])
        exit_ = float(last["c"])
    except (KeyError, ValueError, TypeError):
        return None
    if entry <= 0:
        return None
    return (exit_ / entry - 1) * 100


def _percentile_rank(value: float, distribution: list) -> Optional[float]:
    """Return the percentile of `value` in `distribution` (0–100).
    50 = beats half the distribution; 100 = beats every value."""
    if not distribution:
        return None
    n_below = sum(1 for v in distribution if v < value)
    return round(n_below / len(distribution) * 100, 1)


def _model_return_for_window(equity: list, ts_ms: list, window_days: Optional[int]) -> Optional[float]:
    """Compute the model's % return over a window from the equity curve."""
    if not equity or len(equity) < 2 or len(ts_ms) != len(equity):
        return None
    if window_days is None:
        # inception
        start_eq = next((e for e in equity if e > 0), 0)
        if start_eq <= 0:
            return None
        return (equity[-1] / start_eq - 1) * 100
    cutoff_ms = (datetime.now() - timedelta(days=window_days)).timestamp() * 1000
    # Find the equity snapshot at or just before the cutoff.
    start_eq = None
    for t, e in zip(ts_ms, equity):
        if t <= cutoff_ms and e > 0:
            start_eq = e
        elif t > cutoff_ms:
            break
    # If the window predates the equity history, fall back to the first
    # snapshot (effectively the inception return for shorter histories).
    if start_eq is None:
        start_eq = next((e for e in equity if e > 0), 0)
    if start_eq is None or start_eq <= 0:
        return None
    return (equity[-1] / start_eq - 1) * 100


def _window_start_ms(window_days: Optional[int], inception_ts_ms: Optional[int]) -> Optional[int]:
    """Convert a window length to a start timestamp in milliseconds."""
    if window_days is None:
        return inception_ts_ms
    return int((datetime.now() - timedelta(days=window_days)).timestamp() * 1000)


def _compute_performance(model_name: str) -> dict:
    """Compute model performance vs full tradeable universe across windows.

    Headline metric: percentile rank of the model's equity-curve return in
    the distribution of buy-and-hold returns across the entire tradeable
    universe over the same window.
    """
    logger = logging.getLogger("dashboard")

    # Cache check
    with _perf_cache_lock:
        if model_name in _perf_cache:
            cached_data, cached_at = _perf_cache[model_name]
            if time.time() - cached_at < PERF_CACHE_TTL:
                return cached_data

    mc = _get_model_config(model_name)
    if not mc:
        return {"error": f"Model {model_name} not found"}

    # 1) Model equity history. We use period=all (not 1A) so Alpaca returns
    # only the days the account has actually existed. With period=1A Alpaca
    # pads pre-funding days with a synthetic $100k equity, which makes
    # `inception_ts_ms = ts_ms[0]` point to ~1 year ago instead of the
    # actual account-funded date — and then SPY / universe returns get
    # measured over a full year while the model's return reflects only the
    # real trading days, producing wildly inflated alphas at inception.
    try:
        hist = pipeline.alpaca_request(
            "GET",
            "v2/account/portfolio/history?period=all&timeframe=1D&extended_hours=false",
            mc, logger=logger,
        )
    except Exception as e:
        return {"error": f"Failed to fetch portfolio history: {e}"}

    timestamps = hist.get("timestamp") or []
    equity = [float(e) for e in (hist.get("equity") or [])]
    # Alpaca timestamps are seconds → milliseconds for consistency
    ts_ms = [int(t) * 1000 for t in timestamps]

    if len(equity) < 2:
        return {"error": "Insufficient equity history (need ≥ 2 daily snapshots)", "model": model_name}

    inception_ts_ms = ts_ms[0] if ts_ms else None

    # 2) Current positions (for the held-positions table at the bottom)
    try:
        positions = pipeline.alpaca_request("GET", "v2/positions", mc, logger=logger) or []
    except Exception:
        positions = []
    model_positions = [{
        "symbol": p["symbol"],
        "avg_entry_price": float(p.get("avg_entry_price", 0)),
        "current_price":  float(p.get("current_price", 0)),
        "market_value":   float(p.get("market_value", 0)),
        "cost_basis":     float(p.get("cost_basis", 0)),
        "unrealized_plpc": float(p.get("unrealized_plpc", 0)) * 100,
        "qty":            float(p.get("qty", 0)),
    } for p in positions]

    # 3) Universe bars — one fetch covers every window. We're permissive
    # with the symbol filter (just ASCII + reasonable length); Alpaca will
    # silently omit tickers it can't price for the IEX feed and we drop
    # those during stat aggregation.
    universe_syms = _get_universe_symbols()
    universe_syms_clean = [
        s for s in universe_syms
        if s and s.isascii() and len(s) <= 8 and "/" not in s
    ]

    # Determine the longest lookback we need (max of finite windows; inception
    # uses whatever the equity history covers).
    finite_lookbacks = [days for _, days in PERF_WINDOWS if days is not None]
    max_window = max(finite_lookbacks) if finite_lookbacks else 30
    if inception_ts_ms is not None:
        days_since_inception = max(
            1, int((datetime.now().timestamp() * 1000 - inception_ts_ms) / 86_400_000)
        )
        max_window = max(max_window, days_since_inception)

    universe_bars, data_feed = _get_universe_bars(
        mc, universe_syms_clean, max_window, logger,
    )

    # SPY (separate single-symbol fetch — also cached implicitly inside the
    # universe call when "SPY" happens to be in syms; usually it isn't).
    spy_bars: list = []
    start_dt = datetime.now() - timedelta(days=max_window + 5)
    spy_params = {
        "start": start_dt.strftime("%Y-%m-%d"),
        "timeframe": "1Day",
        "limit": 10000,
        "adjustment": "split",
    }
    try:
        resp, _ = _alpaca_bars_with_fallback(
            mc, "v2/stocks/SPY/bars", spy_params, logger=logger,
        )
        if resp is not None and resp.status_code == 200:
            spy_bars = resp.json().get("bars") or []
    except Exception as e:
        logger.warning(f"[PERF] SPY fetch: {e}")

    # 4) Per-window stats
    windows_out = {}
    for label, days in PERF_WINDOWS:
        win_start_ms = _window_start_ms(days, inception_ts_ms)
        model_ret = _model_return_for_window(equity, ts_ms, days)

        univ_returns = []
        for sym, blist in universe_bars.items():
            r = _bar_return_in_window(blist, win_start_ms)
            if r is not None:
                univ_returns.append(r)

        spy_ret = _bar_return_in_window(spy_bars, win_start_ms) if spy_bars else None

        if univ_returns:
            mean_ = float(np.mean(univ_returns))
            median_ = float(np.median(univ_returns))
            std_ = float(np.std(univ_returns))
            pct_pos = sum(1 for r in univ_returns if r > 0) / len(univ_returns) * 100
            pctile = _percentile_rank(model_ret, univ_returns) if model_ret is not None else None
        else:
            mean_ = median_ = std_ = pct_pos = 0.0
            pctile = None

        windows_out[label] = {
            "model_return":     None if model_ret is None else round(model_ret, 2),
            "univ_n":           len(univ_returns),
            "univ_mean":        round(mean_, 2),
            "univ_median":      round(median_, 2),
            "univ_std":         round(std_, 2),
            "univ_pct_positive": round(pct_pos, 1),
            "univ_pctile":      pctile,
            "spy_return":       None if spy_ret is None else round(spy_ret, 2),
            "alpha_vs_median":  None if model_ret is None else round(model_ret - median_, 2),
            "alpha_vs_spy":     None if (model_ret is None or spy_ret is None) else round(model_ret - spy_ret, 2),
            # A small bucketed histogram for the UI to draw without re-sending
            # every return. 21 buckets between -20% and +20% (clipped).
            "univ_histogram":   _bucket_returns(univ_returns) if univ_returns else None,
        }

    # 5) Pick the headline window — prefer "inception" if it has data.
    headline_window = "inception"
    headline = windows_out.get(headline_window) or {}
    if headline.get("model_return") is None:
        # Fall back to the longest window that does have data
        for label in ("30d", "7d", "1d"):
            if windows_out.get(label, {}).get("model_return") is not None:
                headline_window = label
                headline = windows_out[label]
                break

    result = {
        "model": model_name,
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "n_positions": len(model_positions),
        "universe_size_total": len(universe_syms_clean),
        "data_feed": data_feed,
        "windows": windows_out,
        "headline": {
            "window":       headline_window,
            "model_return": headline.get("model_return"),
            "univ_pctile":  headline.get("univ_pctile"),
            "univ_n":       headline.get("univ_n"),
        },
        "current_positions": sorted(model_positions, key=lambda x: x["unrealized_plpc"], reverse=True),
        "indicator_help": INDICATOR_HELP,
    }

    with _perf_cache_lock:
        _perf_cache[model_name] = (result, time.time())
        if len(_perf_cache) > 50:
            oldest = min(_perf_cache, key=lambda m: _perf_cache[m][1])
            del _perf_cache[oldest]

    return result


def _bucket_returns(returns: list, lo: float = -20.0, hi: float = 20.0,
                    n_buckets: int = 21) -> dict:
    """Build a small histogram for the UI: returns clipped to [lo, hi]."""
    width = (hi - lo) / n_buckets
    counts = [0] * n_buckets
    for r in returns:
        idx = int((max(lo, min(hi - 1e-9, r)) - lo) / width)
        idx = max(0, min(n_buckets - 1, idx))
        counts[idx] += 1
    return {
        "lo": lo,
        "hi": hi,
        "width": width,
        "counts": counts,
        "total": len(returns),
    }


@app.route("/api/performance/<model_name>")
def api_performance(model_name):
    """Performance comparison: model vs universe across windows."""
    try:
        result = _compute_performance(model_name)
        if "error" in result and "windows" not in result:
            return jsonify(result), 404 if "not found" in result.get("error", "") else 500
        return jsonify(result)
    except Exception as e:
        logging.getLogger("dashboard").error(f"Performance API error for {model_name}: {e}")
        return jsonify({"error": f"Internal error: {str(e)}", "model": model_name}), 500


@app.route("/api/description/<model_name>")
def api_description(model_name):
    """Model description and architecture details."""
    try:
        desc = pipeline.MODEL_DESCRIPTIONS.get(model_name)
        if not desc:
            return jsonify({"error": f"No description for model {model_name}"}), 404

        mc = _get_model_config(model_name)
        result = dict(desc)
        result["name"] = model_name

        # Add live config details
        if mc:
            result["enable_cutloss"] = mc.enable_cutloss
            if mc.enable_cutloss:
                result["cutloss_config"] = {
                    "hard_stop": f"{mc.cutloss_hard_stop}%",
                    "trailing_stop": f"{mc.cutloss_trailing_stop}%",
                    "portfolio_stop": f"{mc.cutloss_portfolio_stop}%",
                }
        else:
            result["enable_cutloss"] = False

        return jsonify(result)
    except Exception as e:
        logging.getLogger("dashboard").error(f"Description API error for {model_name}: {e}")
        return jsonify({"error": str(e)}), 500


# ── CONFIG ENDPOINTS (model slot management) ─────────────────────────────

@app.route("/api/config/models")
def api_config_models():
    """Get current model slot configuration."""
    config = pipeline.load_model_config()
    # Enrich with model info
    for slot in config.get("slots", []):
        model_name = slot.get("model", "")
        slot["model_file_exists"] = pipeline._resolve_model_path(model_name).exists()
        slot["description"] = pipeline.MODEL_DESCRIPTIONS.get(model_name, {}).get("title", model_name)
    config["available_models"] = list(pipeline.MODEL_REGISTRY.keys())
    config["active_models"] = [mc.name for mc in pipeline.get_active_models()]
    return jsonify(config)


@app.route("/api/config/models", methods=["POST"])
def api_config_update():
    """Update model slot configuration."""
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = flask_request.get_json()
        if not data or "slots" not in data:
            return jsonify({"error": "Missing 'slots' in request body"}), 400

        config = pipeline.load_model_config()
        new_slots = data["slots"]

        # Validate
        # Cap raised to 8 (was 3) so the dashboard accepts the v9 + combo_v1
        # slots added after the original 3-slot design. The JS side already
        # renders Math.max(slots.length, 4) so it scales automatically.
        if len(new_slots) > 8:
            return jsonify({"error": "Maximum 8 slots allowed"}), 400

        for slot in new_slots:
            model = slot.get("model", "")
            if model and model not in pipeline.MODEL_REGISTRY:
                return jsonify({"error": f"Unknown model: {model}"}), 400

        config["slots"] = new_slots
        pipeline.save_model_config(config)

        # Re-initialize model status in dashboard
        _initialize_model_status()

        # Clear API caches since models may have changed
        with _cache_lock:
            _api_cache.clear()

        return jsonify({"ok": True, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/test-key", methods=["POST"])
def api_config_test_key():
    """Test an Alpaca API key pair."""
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = flask_request.get_json()
        key = data.get("key", "")
        secret = data.get("secret", "")
        if not key or not secret:
            return jsonify({"ok": False, "error": "Both key and secret are required"}), 400

        result = pipeline.test_alpaca_key(key, secret)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── LOG ENDPOINTS ─────────────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs_list():
    files = _get_log_files()[:50]
    result = []
    for f in files:
        name = f.name
        if name.startswith("pipeline_"):
            cat = "pipeline"
            # Extract model from filename: pipeline_v7_20260422_133527.log
            parts = name.replace("pipeline_", "").split("_")
            model = parts[0].upper() if parts else "?"
            date_str = parts[1] if len(parts) > 1 else ""
        elif name.startswith("cutloss_"):
            cat = "cutloss"
            model = "ALL"
            date_str = name.replace("cutloss_", "").replace(".log", "")
        else:
            cat = "other"
            model = "?"
            date_str = ""
        # Format date nicely: 20260422 -> 2026-04-22
        if len(date_str) == 8:
            date_display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            date_display = date_str
        size_kb = round(f.stat().st_size / 1024, 1)
        result.append({
            "filename": name,
            "category": cat,
            "model": model,
            "date": date_display,
            "size_kb": size_kb,
        })
    return jsonify(result)


def _resolve_log_path(filename: str) -> Optional[Path]:
    """Validate `filename` and return its absolute Path, or None if invalid."""
    log_path = (LOG_DIR / filename).resolve()
    if not str(log_path).startswith(str(LOG_DIR.resolve())):
        return None
    allowed_prefixes = ("pipeline_", "cutloss_")
    if not log_path.exists() or not any(filename.startswith(p) for p in allowed_prefixes):
        return None
    return log_path


@app.route("/api/logs/<filename>")
def api_log_content(filename):
    log_path = _resolve_log_path(filename)
    if log_path is None:
        return jsonify({"error": "Not found"}), 404

    # ?tail=500 (default) | 100 | 2000 | all
    tail_arg = (flask_request.args.get("tail") or "500").lower()
    tail = 0 if tail_arg == "all" else max(0, int(tail_arg)) if tail_arg.isdigit() else 500
    level = flask_request.args.get("level")  # ERROR | WARNING | None

    content, total, returned = _read_log(log_path, tail=tail, level=level)
    return jsonify({
        "filename": filename,
        "content": content,
        "total_lines": total,
        "returned_lines": returned,
        "mtime": log_path.stat().st_mtime,
    })


@app.route("/api/logs/<filename>/download")
def api_log_download(filename):
    log_path = _resolve_log_path(filename)
    if log_path is None:
        return jsonify({"error": "Not found"}), 404
    return Response(
        log_path.read_bytes(),
        mimetype="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── ACTIONS ───────────────────────────────────────────────────────────────

@app.route("/run")
def trigger_run():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    model_filter = flask_request.args.get("model", None)
    if model_filter and not _validate_model_name(model_filter):
        return jsonify({"error": "Invalid model name"}), 400
    force = "force" in flask_request.args
    dry = "dry" in flask_request.args

    with status_lock:
        if bot_status["state"] == "running":
            return jsonify({"error": "Pipeline already running"}), 409

    if dry:
        def _dry():
            global bot_status
            with status_lock:
                bot_status["state"] = "running"
                bot_status["current_step"] = "dry run"
            try:
                pipeline.run_pipeline(dry_run=True, force=True, model_filter=model_filter)
                with status_lock:
                    bot_status["state"] = "idle"
                    bot_status["last_run_at"] = datetime.now().isoformat()
                    bot_status["last_run_status"] = "success (dry)"
                    bot_status["current_step"] = None
            except Exception as e:
                with status_lock:
                    bot_status["state"] = "error"
                    bot_status["last_error"] = str(e)
                    bot_status["current_step"] = None
        threading.Thread(target=_dry, daemon=True).start()
    else:
        threading.Thread(
            target=run_trading_pipeline,
            kwargs={"force": force, "model_filter": model_filter},
            daemon=True
        ).start()

    return jsonify({"status": "triggered", "model": model_filter or "all", "dry": dry})


# ═══════════════════════════════════════════════════════════════════════════
# EXTERNAL API — /api/v1/ (read-only, CORS-enabled, no auth)
# ═══════════════════════════════════════════════════════════════════════════

def _cors_response(data, status=200):
    """Wrap jsonify response with CORS headers for external access."""
    resp = jsonify(data)
    resp.status_code = status
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/v1/models", methods=["GET", "OPTIONS"])
def apiv1_models():
    """List ALL registered models (V4-V8+) with status info."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})

    active_names = {mc.name for mc in pipeline.get_active_models()}
    active_map = {mc.name: mc for mc in pipeline.get_active_models()}

    models = []
    for name, reg in pipeline.MODEL_REGISTRY.items():
        desc = pipeline.MODEL_DESCRIPTIONS.get(name, {})
        state = _load_model_state(name)
        mc = active_map.get(name)

        entry = {
            "name": name,
            "title": desc.get("title", name.upper()),
            "feature_version": reg.get("feature_version", name),
            "active": name in active_names,
            "run_count": state.get("run_count", 0),
            "last_rebalance": state.get("last_rebalance"),
        }
        if mc:
            entry["enable_cutloss"] = mc.enable_cutloss
            entry["has_alpaca_keys"] = bool(mc.alpaca_key)
        else:
            entry["enable_cutloss"] = False
            entry["has_alpaca_keys"] = False

        models.append(entry)

    return _cors_response({"models": models, "total": len(models), "active_count": len(active_names)})


@app.route("/api/v1/models/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_model_detail(model_name):
    """Full detail for a single model: description, state, account, positions.

    Works for ANY registered model (V4-V8+). Alpaca data only available
    for models that are active (assigned to a slot with API keys).
    """
    if flask_request.method == "OPTIONS":
        return _cors_response({})

    # Check if model exists in registry at all
    if model_name not in pipeline.MODEL_REGISTRY:
        return _cors_response({"error": f"Model {model_name} not found in registry"}, 404)

    mc = _get_model_config(model_name)  # None if not active
    logger = logging.getLogger("dashboard")

    result = {
        "name": model_name,
        "active": mc is not None,
        "registry": pipeline.MODEL_REGISTRY.get(model_name, {}),
    }

    # Description (always available)
    desc = pipeline.MODEL_DESCRIPTIONS.get(model_name, {})
    result["description"] = desc

    # Bundle metadata via LRU-cached loader (see _get_bundle_metadata).
    # combo_v1 is 1 KB so cost is trivial; v9 is 8.5 MB so caching saves
    # ~40 MB of pickle deserialization per minute at /v2's 30s refresh.
    for k, v in _get_bundle_metadata(model_name).items():
        result[k] = v

    # State (always available — from state file on disk)
    state = _load_model_state(model_name)
    result["run_count"] = state.get("run_count", 0)
    result["last_rebalance"] = state.get("last_rebalance")
    result["last_run"] = state.get("last_run")
    # Convenient nested form for the v2 dashboard
    result["state"] = {
        "run_count": state.get("run_count", 0),
        "last_rebalance": state.get("last_rebalance"),
        "last_run": state.get("last_run"),
        "portfolio_stop_tripped_date": state.get("portfolio_stop_tripped_date"),
    }
    # Cutloss state — match the tiers from core/risk.py exactly:
    #   Tier 3 (= "tripped" when persisted): daily DD ≤ pstop × 7/3
    #   Tier 2: daily DD ≤ pstop × 5/3 (no trip flag)
    #   Tier 1: daily DD ≤ pstop (no trip flag)
    #   else:  "normal"
    # Tier 2/3 only fire intraday (the 60s scanner) so without a live
    # equity number we fall back to "tripped" iff today's persisted flag
    # is set. With a live equity we can compute the current tier.
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("portfolio_stop_tripped_date") == today:
        result["cutloss_state"] = "tripped"
    else:
        result["cutloss_state"] = "normal"  # may be upgraded below
    # Upgrade with a live tier read if we have an active slot + start equity
    if mc and mc.enable_cutloss and state.get("daily_portfolio_start_date") == today:
        try:
            acct_live = pipeline.alpaca_request("GET", "v2/account", mc,
                                                logger=logger)
            cur_eq = float(acct_live.get("equity", 0))
            start_eq = float(state.get("daily_portfolio_start", 0) or 0)
            if start_eq > 0 and cur_eq > 0:
                dd_pct = (cur_eq / start_eq - 1.0) * 100.0
                pstop = float(mc.cutloss_portfolio_stop)  # negative
                if dd_pct <= pstop * (7.0 / 3.0):
                    result["cutloss_state"] = "tier3"
                elif dd_pct <= pstop * (5.0 / 3.0):
                    result["cutloss_state"] = "tier2"
                elif dd_pct <= pstop:
                    result["cutloss_state"] = "tier1"
                result["cutloss_daily_dd_pct"] = round(dd_pct, 4)
        except Exception:
            pass

    # Full run history
    history = state.get("history", [])
    result["history_count"] = len(history)
    if history:
        latest = history[-1]
        # direct_weights strategies (combo_v1) store under "weights";
        # ml_ranker strategies (v4/v5/v6/v8/v9) store under "predictions".
        # Surface whichever exists so the dashboard works for both.
        result["latest_picks"] = {
            "date": latest.get("date"),
            "symbols": latest.get("target_symbols", []),
            "predictions": latest.get("predictions", {}),
            "weights": latest.get("weights", {}),
            "strategy_type": latest.get("strategy_type", "ml_ranker"),
            "conviction_weights": latest.get("conviction_weights", {}),
            "regime_exposure": latest.get("regime_exposure"),
        }
        result["history"] = history[-20:]  # Last 20 runs

    # Recent trades (from journal — always available if file exists)
    trades = _load_recent_trades(model_name, 50)
    result["trade_count"] = len(trades)

    # Alpaca data (only if model is active with keys)
    if mc:
        result["enable_cutloss"] = mc.enable_cutloss
        if mc.enable_cutloss:
            result["cutloss_config"] = {
                "hard_stop": mc.cutloss_hard_stop,
                "trailing_stop": mc.cutloss_trailing_stop,
                "portfolio_stop": mc.cutloss_portfolio_stop,
            }

        # Account
        try:
            acct = pipeline.alpaca_request("GET", "v2/account", mc, logger=logger)
            result["account"] = {
                "equity": float(acct.get("equity", 0)),
                "cash": float(acct.get("cash", 0)),
                "portfolio_value": float(acct.get("portfolio_value", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
                "long_market_value": float(acct.get("long_market_value", 0)),
                "last_equity": float(acct.get("last_equity", 0)),
            }
        except Exception:
            result["account"] = None

        # Positions
        try:
            positions = pipeline.alpaca_request("GET", "v2/positions", mc, logger=logger)
            result["positions"] = [{
                "symbol": p["symbol"],
                "qty": float(p["qty"]),
                "market_value": float(p["market_value"]),
                "unrealized_pl": float(p["unrealized_pl"]),
                "unrealized_plpc": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "current_price": float(p.get("current_price", 0)),
            } for p in positions]
        except Exception:
            result["positions"] = None
    else:
        result["account"] = None
        result["positions"] = None
        result["note"] = "Model not active (not assigned to a trading slot). No live Alpaca data."

    return _cors_response(result)


@app.route("/api/v1/positions/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_positions(model_name):
    """Live positions for a model."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    mc = _get_model_config(model_name)
    if not mc:
        return _cors_response({"error": f"Model {model_name} not found"}, 404)

    logger = logging.getLogger("dashboard")
    try:
        positions = pipeline.alpaca_request("GET", "v2/positions", mc, logger=logger)
        data = [{
            "symbol": p["symbol"],
            "qty": float(p["qty"]),
            "market_value": float(p["market_value"]),
            "cost_basis": float(p.get("cost_basis", 0)),
            "unrealized_pl": float(p["unrealized_pl"]),
            "unrealized_plpc": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
            "avg_entry_price": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "change_today": round(float(p.get("change_today", 0)) * 100, 2),
        } for p in positions]
        data.sort(key=lambda x: x["market_value"], reverse=True)
        total_value = sum(p["market_value"] for p in data)
        total_pl = sum(p["unrealized_pl"] for p in data)
        return _cors_response({
            "model": model_name,
            "count": len(data),
            "total_market_value": round(total_value, 2),
            "total_unrealized_pl": round(total_pl, 2),
            "positions": data,
        })
    except Exception as e:
        return _cors_response({"error": str(e)}, 500)


@app.route("/api/v1/trades/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_trades(model_name):
    """Recent trades for a model."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    try:
        limit = max(1, min(int(flask_request.args.get("limit", 50)), 1000))
    except (ValueError, TypeError):
        limit = 50
    trades = _load_trades_merged(model_name, limit)

    return _cors_response({
        "model": model_name,
        "count": len(trades),
        "trades": trades,
    })


@app.route("/api/v1/cutloss/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_cutloss(model_name):
    """Cutloss events for a model."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    try:
        limit = max(1, min(int(flask_request.args.get("limit", 100)), 500))
    except (ValueError, TypeError):
        limit = 100
    events = _load_cutloss_events(model_name, limit)
    return _cors_response({
        "model": model_name,
        "count": len(events),
        "events": events,
    })


@app.route("/api/v1/account/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_account(model_name):
    """Live Alpaca account info for a model."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    mc = _get_model_config(model_name)
    if not mc:
        return _cors_response({"error": f"Model {model_name} not found"}, 404)

    logger = logging.getLogger("dashboard")
    try:
        acct = pipeline.alpaca_request("GET", "v2/account", mc, logger=logger)
        return _cors_response({
            "model": model_name,
            "equity": float(acct.get("equity", 0)),
            "cash": float(acct.get("cash", 0)),
            "portfolio_value": float(acct.get("portfolio_value", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "long_market_value": float(acct.get("long_market_value", 0)),
            "last_equity": float(acct.get("last_equity", 0)),
            "status": acct.get("status", "unknown"),
        })
    except Exception as e:
        return _cors_response({"error": str(e)}, 500)


@app.route("/api/v1/equity/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_equity(model_name):
    """Equity curve (portfolio history) for a model."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    mc = _get_model_config(model_name)
    if not mc:
        return _cors_response({"error": f"Model {model_name} not found"}, 404)

    period = flask_request.args.get("period", "1M")
    logger = logging.getLogger("dashboard")
    try:
        hist = pipeline.alpaca_request(
            "GET",
            f"v2/account/portfolio/history?period={period}&timeframe=1D&extended_hours=false",
            mc, logger=logger
        )
        return _cors_response({
            "model": model_name,
            "period": period,
            "timestamps": hist.get("timestamp", []),
            "equity": hist.get("equity", []),
            "profit_loss": hist.get("profit_loss", []),
            "profit_loss_pct": hist.get("profit_loss_pct", []),
            "base_value": hist.get("base_value", 0),
        })
    except Exception as e:
        return _cors_response({"error": str(e)}, 500)


@app.route("/api/v1/performance/<model_name>", methods=["GET", "OPTIONS"])
def apiv1_performance(model_name):
    """Performance benchmark: model vs universe."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    try:
        result = _compute_performance(model_name)
        return _cors_response(result)
    except Exception as e:
        return _cors_response({"error": str(e)}, 500)


@app.route("/api/v1/docs", methods=["GET", "OPTIONS"])
def apiv1_docs():
    """API documentation."""
    if flask_request.method == "OPTIONS":
        return _cors_response({})

    base_url = flask_request.url_root.rstrip("/")
    docs = {
        "name": "ML Trading Bot API",
        "version": "v1",
        "description": "Read-only API for accessing trading bot data. CORS enabled, no authentication required.",
        "base_url": f"{base_url}/api/v1",
        "endpoints": [
            {
                "path": "/api/v1/docs",
                "method": "GET",
                "description": "This documentation page",
            },
            {
                "path": "/api/v1/models",
                "method": "GET",
                "description": "List all active models with basic info (name, title, run count)",
            },
            {
                "path": "/api/v1/models/<model_name>",
                "method": "GET",
                "description": "Full detail for one model: description, account, positions, latest picks",
            },
            {
                "path": "/api/v1/account/<model_name>",
                "method": "GET",
                "description": "Live Alpaca account info (equity, cash, buying power)",
            },
            {
                "path": "/api/v1/positions/<model_name>",
                "method": "GET",
                "description": "Live positions with P&L. Returns count, total value, and per-position detail",
            },
            {
                "path": "/api/v1/trades/<model_name>",
                "method": "GET",
                "description": "Recent trades. Optional ?limit=N (default 50)",
            },
            {
                "path": "/api/v1/equity/<model_name>",
                "method": "GET",
                "description": "Equity curve over time. Optional ?period=1D|1W|1M|3M|1A|all (default 1M)",
            },
            {
                "path": "/api/v1/performance/<model_name>",
                "method": "GET",
                "description": "Performance benchmark: model returns vs universe average",
            },
        ],
        "all_models": list(pipeline.MODEL_REGISTRY.keys()),
        "active_models": [mc.name for mc in pipeline.get_active_models()],
        "notes": [
            "All endpoints under /api/v1/ are read-only with CORS enabled.",
            "/api/v1/models lists ALL registered models (V4-V8+), not just active ones.",
            "/api/v1/models/<name> returns description, run history, and predictions for ANY registered model.",
            "Alpaca live data (account, positions, trades, equity) only available for models assigned to a trading slot.",
            "New models added to MODEL_REGISTRY automatically appear in the API — no code changes needed.",
        ],
        "examples": [
            f"{base_url}/api/v1/models",
            f"{base_url}/api/v1/models/v6",
            f"{base_url}/api/v1/models/v8",
            f"{base_url}/api/v1/positions/v6",
            f"{base_url}/api/v1/trades/v6?limit=20",
            f"{base_url}/api/v1/equity/v6?period=3M",
        ],
    }
    return _cors_response(docs)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "uptime_since": bot_status["started_at"]})


@app.route("/api/v1/market-status", methods=["GET", "OPTIONS"])
def apiv1_market_status():
    """Whether the US equity market is currently open. Uses the same
    `core.market.is_market_open()` used by the pipeline so the dashboard
    indicator always matches reality (DST-aware, holiday-aware).
    """
    if flask_request.method == "OPTIONS":
        return _cors_response({})
    try:
        from core.market import is_market_open
        is_open = bool(is_market_open())
    except Exception as e:
        return _cors_response({"error": str(e)}, 500)
    return _cors_response({
        "is_open": is_open,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    })


# ── Bundle metadata cache ─────────────────────────────────────────────────
# /api/v1/models/<name> exposes a subset of the model bundle (tag, version,
# strategy_type, backtest_reference, ...). Without caching, each call
# deserializes the full pickle — that's 8.5 MB for v9 — every time. At
# /v2's 30s refresh × 5 slots that's >40 MB/min of pure waste. The cache
# is keyed by (path, mtime) so a fresh bundle (re-deploy or rebuild)
# invalidates automatically without a server restart.

_BUNDLE_META_KEYS = ("tag", "version", "strategy_type", "horizon",
                     "saved_at", "backtest_reference")


@functools.lru_cache(maxsize=32)
def _load_bundle_meta_cached(path_str: str, mtime: float) -> dict:
    from core.runner import load_model_bundle
    bundle = load_model_bundle(path_str)
    return {k: bundle[k] for k in _BUNDLE_META_KEYS if k in bundle}


def _get_bundle_metadata(model_name: str) -> dict:
    """Return the cached metadata subset of a model bundle, or {} on miss."""
    try:
        model_path = pipeline._resolve_model_path(model_name)
        if not model_path.exists():
            return {}
        return _load_bundle_meta_cached(str(model_path), model_path.stat().st_mtime)
    except Exception:
        return {}


# ── V2 DASHBOARD ──────────────────────────────────────────────────────────
# Clean, focused UI redesign. Lives at /v2 alongside the original /. Uses the
# existing /api/v1/* endpoints client-side. Single self-contained route — no
# external templates, no asset pipeline.
#
# Layout:
#   ▸ Hero: aggregate equity + today's P&L across all live slots
#   ▸ Slot grid: per-slot KPI cards (equity, day/week/month P&L, drawdown,
#                # positions, cutloss state, last rebal)
#   ▸ Backtest-vs-live comparison row for each slot that carries a
#     `backtest_reference` in its bundle (combo_v1, etc.)
#   ▸ Top winners + losers across the union of all slot positions
@app.route("/v2")
def index_v2():
    return V2_DASHBOARD_HTML


V2_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Trading Bot — V2</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a26;
    --panel-2: #1a2434;
    --line: #233146;
    --text: #e4e8ef;
    --muted: #8a94a4;
    --accent: #4dd0e1;
    --good: #4ade80;
    --bad: #f87171;
    --warn: #fbbf24;
    --neutral: #94a3b8;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    font-size: 14px; line-height: 1.5;
  }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
  header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
  header h1 { font-size: 22px; margin: 0; font-weight: 600; }
  header .tag { color: var(--muted); font-size: 12px; margin-left: 12px; padding: 2px 8px; background: var(--panel-2); border-radius: 4px; }
  header .links a { color: var(--accent); text-decoration: none; margin-left: 16px; font-size: 13px; }
  header .links a:hover { text-decoration: underline; }

  .hero { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
  .stat {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px 20px;
  }
  .stat .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 8px; }
  .stat .value { font-size: 28px; font-weight: 600; font-feature-settings: "tnum"; }
  .stat .sub { font-size: 13px; color: var(--muted); margin-top: 4px; font-feature-settings: "tnum"; }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  .neutral { color: var(--neutral); }
  .warn { color: var(--warn); }

  h2 { font-size: 16px; margin: 0 0 12px 0; font-weight: 600; color: var(--text); }
  h2 .count { color: var(--muted); font-weight: 400; font-size: 13px; margin-left: 8px; }

  .slot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; margin-bottom: 32px; }
  .slot {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 20px; position: relative;
  }
  .slot .head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .slot .name { font-size: 18px; font-weight: 600; }
  .slot .strategy { font-size: 11px; padding: 3px 8px; background: var(--panel-2); border-radius: 4px; color: var(--muted); margin-left: 8px; vertical-align: middle; }
  .slot .equity { font-size: 28px; font-weight: 600; font-feature-settings: "tnum"; margin-bottom: 6px; }
  .slot .day-pnl { font-size: 15px; font-weight: 500; font-feature-settings: "tnum"; }
  .slot .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line); }
  .slot .kpi .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 4px; }
  .slot .kpi .v { font-size: 15px; font-weight: 500; font-feature-settings: "tnum"; }
  .slot .footer { display: flex; align-items: center; justify-content: space-between; margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); }
  .slot .pill { padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; }
  .pill-good { background: rgba(74, 222, 128, 0.15); color: var(--good); }
  .pill-bad { background: rgba(248, 113, 113, 0.15); color: var(--bad); }
  .pill-warn { background: rgba(251, 191, 36, 0.15); color: var(--warn); }
  .slot .bt-ref { margin-top: 14px; padding: 10px 12px; background: var(--panel-2); border-radius: 8px; font-size: 12px; color: var(--muted); }
  .slot .bt-ref strong { color: var(--text); font-weight: 500; }

  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 20px;
  }
  table { width: 100%; border-collapse: collapse; }
  th { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 500; text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  td { padding: 8px 8px; border-bottom: 1px solid var(--line); font-size: 13px; font-feature-settings: "tnum"; }
  td:first-child { font-weight: 500; }
  tr:last-child td { border-bottom: 0; }

  .loading { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }
  .error { color: var(--bad); padding: 12px 16px; background: rgba(248, 113, 113, 0.08); border: 1px solid rgba(248, 113, 113, 0.2); border-radius: 8px; font-size: 13px; }

  @media (max-width: 800px) {
    .hero { grid-template-columns: repeat(2, 1fr); }
    .panels { grid-template-columns: 1fr; }
    .slot-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">

<header>
  <div>
    <h1>ML Trading Bot</h1>
    <span class="tag">V2 / clean UI</span>
  </div>
  <div class="links">
    <a href="/">Original dashboard</a>
    <a href="/api/v1/docs">API docs</a>
  </div>
</header>

<section id="hero" class="hero">
  <div class="stat"><div class="label">Total equity</div><div id="hero-equity" class="value">…</div><div id="hero-equity-sub" class="sub">across all slots</div></div>
  <div class="stat"><div class="label">Today's P&L</div><div id="hero-day-pnl" class="value">…</div><div id="hero-day-pnl-sub" class="sub">all slots combined</div></div>
  <div class="stat"><div class="label">Active slots</div><div id="hero-slots" class="value">…</div><div id="hero-slots-sub" class="sub">paper trading</div></div>
  <div class="stat"><div class="label">Market</div><div id="hero-market" class="value">…</div><div id="hero-market-sub" class="sub">…</div></div>
</section>

<h2>Slots <span class="count" id="slot-count"></span></h2>
<section id="slot-grid" class="slot-grid">
  <div class="loading">Loading slot data…</div>
</section>

<section class="panels">
  <div class="panel">
    <h2>Top winners <span class="count">across all positions</span></h2>
    <div id="winners"><div class="loading">…</div></div>
  </div>
  <div class="panel">
    <h2>Top losers <span class="count">across all positions</span></h2>
    <div id="losers"><div class="loading">…</div></div>
  </div>
</section>

</div>

<script>
const fmtUSD = (x) => x == null || isNaN(x) ? '—' : (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmtUSDc = (x) => x == null || isNaN(x) ? '—' : (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = (x) => x == null || isNaN(x) ? '—' : (x >= 0 ? '+' : '') + (x * 100).toFixed(2) + '%';
const fmtPctNoSign = (x) => x == null || isNaN(x) ? '—' : (x * 100).toFixed(1) + '%';
const cls = (x) => x == null || isNaN(x) || x === 0 ? 'neutral' : (x > 0 ? 'good' : 'bad');

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

function pctChange(series) {
  if (!series || series.length < 2) return null;
  const first = series[0]?.equity ?? series[0]?.value ?? series[0];
  const last = series[series.length - 1]?.equity ?? series[series.length - 1]?.value ?? series[series.length - 1];
  if (!first || !last) return null;
  return last / first - 1;
}

async function loadSlot(modelName) {
  const result = { name: modelName, error: null };
  try {
    const model = await jget(`/api/v1/models/${modelName}`);
    result.model = model;
    result.active = !!model.active;
    if (!result.active) return result;
  } catch (e) { result.error = e.message; return result; }
  try { result.account = await jget(`/api/v1/account/${modelName}`); } catch (e) {}
  try { result.positions = await jget(`/api/v1/positions/${modelName}`); } catch (e) {}
  try { result.equity1M = await jget(`/api/v1/equity/${modelName}?period=1M`); } catch (e) {}
  try { result.equity1W = await jget(`/api/v1/equity/${modelName}?period=1W`); } catch (e) {}
  try { result.equity1D = await jget(`/api/v1/equity/${modelName}?period=1D`); } catch (e) {}
  return result;
}

function pickEquitySeries(eq) {
  if (!eq) return null;
  if (Array.isArray(eq.equity)) return eq.equity;
  if (Array.isArray(eq.points)) return eq.points;
  if (Array.isArray(eq)) return eq;
  return null;
}

function maxDrawdown(series) {
  if (!series || series.length < 2) return null;
  const vals = series.map(p => p?.equity ?? p?.value ?? p).filter(Number.isFinite);
  if (vals.length < 2) return null;
  let peak = vals[0], maxDD = 0;
  for (const v of vals) {
    if (v > peak) peak = v;
    const dd = v / peak - 1;
    if (dd < maxDD) maxDD = dd;
  }
  return maxDD;
}

function renderSlot(r) {
  const el = document.createElement('div');
  el.className = 'slot';
  if (!r.active) {
    el.innerHTML = `<div class="head"><div><span class="name">${r.name}</span><span class="strategy">inactive</span></div></div><div class="footer">Slot not assigned an Alpaca key.</div>`;
    return el;
  }
  if (r.error) {
    el.innerHTML = `<div class="head"><span class="name">${r.name}</span></div><div class="error">${r.error}</div>`;
    return el;
  }
  const a = r.account || {};
  const positions = (r.positions && r.positions.positions) || [];
  const equity = +(a.equity ?? a.portfolio_value ?? 0);
  const lastEquity = +(a.last_equity ?? equity);
  const dayPnl = equity - lastEquity;
  const dayPnlPct = lastEquity ? dayPnl / lastEquity : 0;

  const s1M = pickEquitySeries(r.equity1M);
  const s1W = pickEquitySeries(r.equity1W);
  const ret1M = pctChange(s1M);
  const ret1W = pctChange(s1W);
  const dd = maxDrawdown(s1M);

  const cutlossState = (r.model && r.model.cutloss_state) || null;
  const lastRebalRaw = r.model && r.model.state && r.model.state.last_rebalance;
  let pill = `<span class="pill pill-good">healthy</span>`;
  if (cutlossState === "tier3" || cutlossState === "tripped") pill = `<span class="pill pill-bad">circuit-broken</span>`;
  else if (cutlossState === "tier1" || cutlossState === "tier2") pill = `<span class="pill pill-warn">${cutlossState}</span>`;
  else if (!lastRebalRaw && positions.length === 0) pill = `<span class="pill" style="background:rgba(148,163,184,0.15);color:var(--neutral)">pending first rebalance</span>`;

  const lastRebal = r.model && r.model.state && r.model.state.last_rebalance;
  const lastRebalStr = lastRebal ? new Date(lastRebal).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : 'never';

  let btRef = '';
  if (r.model && r.model.backtest_reference) {
    const bt = r.model.backtest_reference;
    btRef = `<div class="bt-ref" title="Backtest reference figures from the model bundle's backtest_reference dict — not live performance.">Backtest claim: <strong>${fmtPctNoSign(bt.mean_monthly_return)}</strong> mean/mo, Sharpe <strong>${bt.sharpe?.toFixed(2)}</strong>, max DD <strong>${fmtPctNoSign(bt.max_drawdown)}</strong> over ${bt.window || 'reference window'}.</div>`;
  }

  el.innerHTML = `
    <div class="head">
      <div>
        <span class="name">${r.name}</span>
        <span class="strategy">${(r.model && r.model.tag) || (r.model && r.model.version) || 'ml_ranker'}</span>
      </div>
      ${pill}
    </div>
    <div class="equity">${fmtUSDc(equity)}</div>
    <div class="day-pnl ${cls(dayPnl)}">${fmtUSD(dayPnl)} <span style="opacity: 0.7; font-size: 13px;">today (${fmtPct(dayPnlPct)})</span></div>
    <div class="kpis">
      <div class="kpi"><div class="k">7-day return</div><div class="v ${cls(ret1W)}">${fmtPct(ret1W)}</div></div>
      <div class="kpi"><div class="k">30-day return</div><div class="v ${cls(ret1M)}">${fmtPct(ret1M)}</div></div>
      <div class="kpi" title="Max drawdown computed over the last 30-day rolling window. Not comparable to backtest since-inception DD."><div class="k">Max DD (30d)</div><div class="v ${cls(dd)}">${fmtPct(dd)}</div></div>
      <div class="kpi"><div class="k">Positions</div><div class="v">${positions.length}</div></div>
      <div class="kpi"><div class="k">Cash</div><div class="v">${fmtUSD(+(a.cash || 0))}</div></div>
      <div class="kpi"><div class="k">Last rebalance</div><div class="v">${lastRebalStr}</div></div>
    </div>
    ${btRef}
  `;
  return el;
}

function renderTopMovers(allPositions) {
  const tbl = (rows) => {
    if (!rows.length) return '<div class="loading">No positions found.</div>';
    const html = `<table><thead><tr><th>Slot</th><th>Symbol</th><th style="text-align:right">P&L</th><th style="text-align:right">% Δ</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${r.slot}</td><td>${r.symbol}</td><td style="text-align:right" class="${cls(r.pnl)}">${fmtUSD(r.pnl)}</td><td style="text-align:right" class="${cls(r.pct)}">${fmtPct(r.pct)}</td></tr>`).join('')}
    </tbody></table>`;
    return html;
  };
  const sorted = allPositions.slice().filter(p => isFinite(p.pct)).sort((a,b) => b.pct - a.pct);
  document.getElementById('winners').innerHTML = tbl(sorted.slice(0, 10));
  document.getElementById('losers').innerHTML = tbl(sorted.slice(-10).reverse());
}

async function refresh() {
  try {
    const models = await jget('/api/v1/models');
    const all = (models.models || models).filter(m => m.active !== false);
    document.getElementById('hero-slots').textContent = all.length;
    document.getElementById('slot-count').textContent = `(${all.length} active)`;

    const grid = document.getElementById('slot-grid');
    grid.innerHTML = '';

    let totalEquity = 0, totalDayPnl = 0;
    const allPositions = [];
    const slotResults = await Promise.all(all.map(m => loadSlot(m.name)));
    for (const r of slotResults) {
      grid.appendChild(renderSlot(r));
      if (r.active && r.account) {
        const eq = +(r.account.equity || 0);
        const lastEq = +(r.account.last_equity || eq);
        totalEquity += eq;
        totalDayPnl += (eq - lastEq);
      }
      const positions = (r.positions && r.positions.positions) || [];
      for (const p of positions) {
        // /api/v1/positions/<name> returns unrealized_plpc as a percent
        // (e.g. 5.2 means 5.2%). The hero/KPI fmtPct expects DECIMAL, so
        // divide by 100 once at intake. Endpoints are not changing because
        // the original / dashboard depends on the percent encoding.
        const plpcPct = +((p.unrealized_plpc ?? p.unrealized_pl_pct) || 0);
        allPositions.push({
          slot: r.name,
          symbol: p.symbol,
          pnl: +(p.unrealized_pl || p.unrealized_pnl || 0),
          pct: plpcPct / 100,
        });
      }
    }

    document.getElementById('hero-equity').textContent = fmtUSDc(totalEquity);
    document.getElementById('hero-equity-sub').textContent = `across ${all.length} slot${all.length === 1 ? '' : 's'}`;
    document.getElementById('hero-day-pnl').textContent = fmtUSD(totalDayPnl);
    document.getElementById('hero-day-pnl').className = 'value ' + cls(totalDayPnl);
    const dpp = totalEquity ? (totalDayPnl / (totalEquity - totalDayPnl)) : 0;
    document.getElementById('hero-day-pnl-sub').textContent = fmtPct(dpp) + ' all slots combined';

    renderTopMovers(allPositions);

    // Market hours: source of truth is the server-side is_market_open()
    // via /api/v1/market-status (DST + holiday aware). JS time-zone math
    // can't replicate that reliably.
    let isOpen = null;
    try {
      const ms = await jget('/api/v1/market-status');
      isOpen = !!ms.is_open;
    } catch (e) {}
    const now = new Date();
    document.getElementById('hero-market').textContent =
      isOpen === null ? '—' : (isOpen ? 'OPEN' : 'CLOSED');
    document.getElementById('hero-market').className =
      'value ' + (isOpen === true ? 'good' : isOpen === false ? 'neutral' : 'neutral');
    document.getElementById('hero-market-sub').textContent =
      now.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  } catch (e) {
    document.getElementById('slot-grid').innerHTML = `<div class="error">Load failed: ${e.message}</div>`;
  }
}

refresh();
setInterval(refresh, 30000);  // refresh every 30s
</script>
</body>
</html>"""


# ── MAIN HTML PAGE ────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        active_models = pipeline.get_active_models()
    except Exception as e:
        logging.getLogger("dashboard").error(f"get_active_models failed: {e}")
        active_models = []
    model_names_json = json.dumps([mc.name for mc in active_models])

    with status_lock:
        s = dict(bot_status)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>ML Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        :root {{
            --bg: #0b0d11; --card: #13161d; --card-border: #1e222d;
            --text: #c9d1d9; --text-dim: #6b7280; --text-bright: #f0f6fc;
            --accent: #3b82f6; --green: #10b981; --red: #ef4444; --yellow: #f59e0b;
            --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        }}
        body {{ font-family: var(--font); background: var(--bg); color: var(--text);
               padding: 16px; max-width: 1400px; margin: 0 auto; }}

        /* Header */
        .header {{ display: flex; align-items: center; justify-content: space-between;
                  margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }}
        .header h1 {{ color: var(--text-bright); font-size: 22px; }}
        .header .meta {{ color: var(--text-dim); font-size: 13px; }}

        /* Cards */
        .card {{ background: var(--card); border: 1px solid var(--card-border);
                border-radius: 10px; padding: 18px; margin-bottom: 16px; }}
        .card h2 {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase;
                   letter-spacing: 1.2px; margin-bottom: 12px; font-weight: 600; }}

        /* Summary row */
        .summary-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                       gap: 12px; margin-bottom: 16px; }}
        .metric {{ background: var(--card); border: 1px solid var(--card-border);
                  border-radius: 10px; padding: 16px; }}
        .metric .label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase;
                         letter-spacing: 0.8px; margin-bottom: 6px; }}
        .metric .val {{ color: var(--text-bright); font-size: 22px; font-weight: 700; }}
        .metric .sub {{ color: var(--text-dim); font-size: 12px; margin-top: 4px; }}
        .metric .val.green {{ color: var(--green); }}
        .metric .val.red {{ color: var(--red); }}

        /* Tabs */
        .tabs {{ display: flex; gap: 0; border-bottom: 2px solid var(--card-border);
                margin-bottom: 16px; overflow-x: auto; }}
        .tab-btn {{ background: none; color: var(--text-dim); border: none;
                   padding: 10px 20px; cursor: pointer; font-size: 13px;
                   font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
                   border-bottom: 2px solid transparent; transition: all 0.15s;
                   white-space: nowrap; }}
        .tab-btn:hover {{ color: var(--text-bright); }}
        .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Sub-tabs within model */
        .sub-tabs {{ display: flex; gap: 4px; margin-bottom: 14px; }}
        .sub-tab {{ background: transparent; color: var(--text-dim); border: 1px solid var(--card-border);
                   padding: 6px 14px; cursor: pointer; font-size: 12px; border-radius: 6px;
                   font-weight: 500; transition: all 0.15s; }}
        .sub-tab:hover {{ color: var(--text); border-color: var(--text-dim); }}
        .sub-tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
        .sub-content {{ display: none; }}
        .sub-content.active {{ display: block; }}

        /* Table */
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 8px 10px; color: var(--text-dim);
             border-bottom: 2px solid var(--card-border); font-size: 11px;
             text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
        td {{ padding: 7px 10px; border-bottom: 1px solid #161a24; }}
        tr:hover td {{ background: #161a24; }}
        .text-right {{ text-align: right; }}
        .green {{ color: var(--green); }}
        .red {{ color: var(--red); }}
        .mono {{ font-family: var(--mono); font-size: 12px; }}

        /* Chart container */
        .chart-container {{ height: 260px; position: relative; margin: 10px 0; }}
        .chart-container canvas {{ width: 100% !important; height: 100% !important; }}

        /* Log */
        .log-box {{ background: #0a0c10; border: 1px solid var(--card-border); border-radius: 8px;
                   padding: 14px; font-family: var(--mono); font-size: 11px; line-height: 1.55;
                   max-height: 520px; overflow-y: auto; white-space: pre-wrap;
                   word-break: break-all; color: #8b95a8; }}
        .log-line {{ display: block; padding: 0 4px; border-radius: 2px; }}
        .log-line.info {{ color: #8b95a8; }}
        .log-line.warn {{ color: #f59e0b; }}
        .log-line.err  {{ color: #f87171; background: rgba(239, 68, 68, 0.07); }}
        .log-line.crit {{ color: #fff;     background: rgba(239, 68, 68, 0.4);
                          font-weight: 600; padding: 2px 6px; }}
        .log-line.match {{ background: rgba(245, 158, 11, 0.18);
                           outline: 1px solid rgba(245, 158, 11, 0.4); }}
        .log-line.hidden {{ display: none; }}
        .log-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
                         margin: 8px 0; font-size: 12px; }}
        .log-controls input[type=search] {{
            background: var(--card); color: var(--text); border: 1px solid var(--card-border);
            padding: 6px 10px; border-radius: 6px; font-size: 12px; min-width: 180px;
            font-family: var(--mono);
        }}
        .log-controls input[type=search]:focus {{ outline: none; border-color: var(--accent); }}
        .log-controls label {{ color: var(--text-dim); font-size: 11px; display: flex;
                               gap: 5px; align-items: center; cursor: pointer; }}
        .log-controls .live-on {{ color: var(--green); font-weight: 600; }}
        .log-meta {{ color: var(--text-dim); font-size: 11px; margin-left: auto;
                     font-family: var(--mono); }}

        /* Buttons */
        .btn {{ display: inline-block; padding: 7px 18px; border-radius: 6px; font-size: 12px;
               font-weight: 600; text-decoration: none; cursor: pointer; border: none;
               transition: all 0.15s; margin-right: 6px; color: #fff; }}
        .btn-primary {{ background: var(--accent); }}
        .btn-primary:hover {{ background: #2563eb; }}
        .btn-secondary {{ background: #374151; }}
        .btn-secondary:hover {{ background: #4b5563; }}
        .btn-danger {{ background: var(--red); }}
        .actions {{ margin: 12px 0; display: flex; gap: 8px; flex-wrap: wrap; }}

        /* Status badge */
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
                 font-size: 11px; font-weight: 600; color: #fff; }}
        .badge-green {{ background: var(--green); }}
        .badge-red {{ background: var(--red); }}
        .badge-yellow {{ background: var(--yellow); }}

        /* Period selector */
        .period-sel {{ display: flex; gap: 4px; margin-bottom: 10px; }}
        .period-btn {{ background: transparent; color: var(--text-dim); border: 1px solid var(--card-border);
                      padding: 4px 10px; cursor: pointer; font-size: 11px; border-radius: 4px;
                      font-weight: 500; }}
        .period-btn:hover {{ color: var(--text); }}
        .period-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

        /* Grid layouts */
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        /* Spinner */
        .spinner {{ display: inline-block; width: 14px; height: 14px;
                   border: 2px solid var(--card-border); border-top-color: var(--accent);
                   border-radius: 50%; animation: spin 0.8s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        /* Refresh indicator */
        .refresh-bar {{ position: fixed; top: 0; left: 0; height: 2px; background: var(--accent);
                       transition: width 0.3s; z-index: 999; }}

        select {{ background: var(--card); color: var(--text); border: 1px solid var(--card-border);
                 padding: 6px 10px; border-radius: 6px; font-size: 12px; }}
    </style>
</head>
<body>
    <div id="refresh-bar" class="refresh-bar" style="width:0%"></div>

    <div class="header">
        <div>
            <h1>ML Trading Bot</h1>
            <p style="margin:2px 0 8px 0;color:var(--text-dim);font-size:13px;">by Arsen Khanguieldyan</p>
            <div class="meta">H{pipeline.HORIZON}_LongOnly{pipeline.TOP_N} &middot; Paper Trading &middot;
                <span id="status-badge"></span> &middot; Auto-refresh 15s &middot;
                <span id="header-clock" style="font-family:var(--mono);"></span></div>
        </div>
        <div class="actions">
            <button class="btn btn-primary" onclick="triggerRun(false)">Run All</button>
            <button class="btn btn-secondary" onclick="triggerRun(true)">Dry Run</button>
        </div>
    </div>

    <!-- Global summary metrics (filled by JS) -->
    <div id="global-summary" class="summary-row"></div>

    <!-- Model tabs -->
    <div class="tabs" id="model-tabs"></div>

    <!-- Model content (filled by JS) -->
    <div id="model-content"></div>

    <!-- Logs section -->
    <div class="card">
        <h2>Logs</h2>
        <p style="color:var(--text-dim);font-size:12px;margin:-4px 0 12px 0;">
            <strong>Pipeline</strong> — model runs: data download, predictions, rebalancing.
            <strong>Cut-Loss</strong> — intraday scanner: stop triggers, sells, redistributions.
        </p>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <div class="sub-tabs" style="margin:0;">
                <button class="sub-tab active" id="log-filter-all" onclick="filterLogs('all')">All</button>
                <button class="sub-tab" id="log-filter-pipeline" onclick="filterLogs('pipeline')">Pipeline</button>
                <button class="sub-tab" id="log-filter-cutloss" onclick="filterLogs('cutloss')">Cut-Loss</button>
            </div>
            <select id="log-model-filter" onchange="filterLogs(currentLogFilter)" style="
                background:var(--card-bg);color:var(--text);border:1px solid var(--card-border);
                border-radius:6px;padding:4px 8px;font-size:12px;">
                <option value="">All models</option>
            </select>
        </div>
        <div id="log-list" style="margin-top:10px;max-height:220px;overflow-y:auto;"></div>

        <div class="log-controls">
            <input type="search" id="log-search" placeholder="Search lines (e.g. portfolio_stop, AAPL)…"
                   oninput="applyLogSearch()" />
            <label>
                Tail
                <select id="log-tail" onchange="reloadCurrentLog()">
                    <option value="100">100</option>
                    <option value="500" selected>500</option>
                    <option value="2000">2000</option>
                    <option value="all">All</option>
                </select>
            </label>
            <label>
                Level
                <select id="log-level" onchange="reloadCurrentLog()">
                    <option value="">All</option>
                    <option value="WARNING">Warn+</option>
                    <option value="ERROR">Errors</option>
                </select>
            </label>
            <label onclick="toggleLiveTail()">
                <input type="checkbox" id="log-live" /> <span id="log-live-label">Live tail</span>
            </label>
            <button class="btn btn-secondary" id="log-download-btn" style="padding:5px 12px;font-size:11px"
                    onclick="downloadCurrentLog()" disabled>Download raw</button>
            <span class="log-meta" id="log-meta"></span>
        </div>

        <div id="log-box" class="log-box">Select a log file above to view its contents.</div>
    </div>

    <!-- Error display -->
    <div id="error-card" class="card" style="display:none;border-color:var(--red);margin-top:16px">
        <h2 style="color:var(--red)">Last Error</h2>
        <div id="error-content" class="log-box" style="color:var(--red)"></div>
    </div>

    <!-- Settings section (collapsible) -->
    <div class="card" id="settings-section">
        <h2 style="cursor:pointer" onclick="toggleSettings()">
            Settings <span id="settings-toggle" style="font-size:10px">&#9660;</span>
        </h2>
        <div id="settings-content" style="display:none">
            <p style="color:var(--text-dim);font-size:12px;margin-bottom:14px">
                3 trading slots available. Assign a model and Alpaca API keys to each slot.
                Changes take effect on the next pipeline run.
            </p>
            <div id="slots-container"></div>
            <div style="margin-top:14px;display:flex;gap:8px">
                <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
                <span id="save-status" style="font-size:12px;color:var(--text-dim);align-self:center"></span>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <script>
    const MODELS = {model_names_json};
    let activeModel = MODELS[0] || null;
    let activeSubTab = {{}};
    let charts = {{}};
    let refreshTimer = null;

    // ── Helpers ──────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const fmt$ = v => v == null ? '-' : '$' + Number(v).toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}});
    const fmtPct = v => v == null ? '-' : (v >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
    const fmtN = (v, d=2) => v == null ? '-' : Number(v).toLocaleString('en-US', {{minimumFractionDigits:d, maximumFractionDigits:d}});
    const cls = v => v > 0 ? 'green' : v < 0 ? 'red' : '';

    // ── Time formatting (always in ET) ────────────────────
    // Accepts: ISO strings, Unix seconds, Unix milliseconds, JS Date.
    function _toDate(input) {{
        if (input == null || input === '') return null;
        if (input instanceof Date) return isNaN(input) ? null : input;
        if (typeof input === 'number') {{
            // Heuristic: <1e12 means seconds, otherwise milliseconds
            const ms = input < 1e12 ? input * 1000 : input;
            const d = new Date(ms);
            return isNaN(d) ? null : d;
        }}
        if (typeof input === 'string') {{
            // Naive ISO without timezone? Treat as UTC so the conversion to
            // ET is deterministic. Pipeline emits local ISO (no Z) for many
            // fields — these are saved on Railway whose TZ is UTC.
            const s = /[Z+]|[+-]\\d\\d:?\\d\\d$/.test(input) ? input : input + 'Z';
            const d = new Date(s);
            return isNaN(d) ? null : d;
        }}
        return null;
    }}
    const _ET_DATETIME = new Intl.DateTimeFormat('en-US', {{
        timeZone: 'America/New_York', year: 'numeric', month: 'short', day: '2-digit',
        hour: '2-digit', minute: '2-digit', hour12: false,
    }});
    const _ET_TIME = new Intl.DateTimeFormat('en-US', {{
        timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit',
        second: '2-digit', hour12: false,
    }});
    const _ET_DATE = new Intl.DateTimeFormat('en-US', {{
        timeZone: 'America/New_York', month: 'short', day: 'numeric',
    }});
    const _ET_DATE_FULL = new Intl.DateTimeFormat('en-US', {{
        timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
    }});
    function fmtET(input, opts) {{
        const d = _toDate(input);
        if (d == null) return '-';
        const mode = (opts && opts.mode) || 'datetime';
        if (mode === 'time') return _ET_TIME.format(d) + ' ET';
        if (mode === 'date') return _ET_DATE.format(d);
        if (mode === 'date-full') return _ET_DATE_FULL.format(d);
        return _ET_DATETIME.format(d) + ' ET';
    }}

    async function api(url) {{
        const r = await fetch(url);
        if (!r.ok) return null;
        return r.json();
    }}

    // ── Build tabs ──────────────────────────────────────
    function buildTabs() {{
        let html = '';
        MODELS.forEach((m, i) => {{
            html += `<button class="tab-btn ${{i===0?'active':''}}" onclick="switchModel('${{m}}', this)">${{m.toUpperCase()}}</button>`;
        }});
        $('model-tabs').innerHTML = html;

        // Build content containers
        let content = '';
        MODELS.forEach(m => {{
            activeSubTab[m] = 'portfolio';
            content += `
            <div id="model-${{m}}" class="tab-content ${{m===activeModel?'active':''}}">
                <div class="sub-tabs">
                    <button class="sub-tab active" onclick="switchSub('${{m}}','portfolio',this)">Portfolio</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','positions',this)">Positions</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','chart',this)">Equity Curve</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','performance',this)">Performance</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','trades',this)">Trades</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','cutloss',this)">Cut-Loss</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','history',this)">Run History</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','about',this)">About</button>
                </div>
                <div id="sub-portfolio-${{m}}" class="sub-content active"></div>
                <div id="sub-positions-${{m}}" class="sub-content"></div>
                <div id="sub-chart-${{m}}" class="sub-content"></div>
                <div id="sub-performance-${{m}}" class="sub-content"></div>
                <div id="sub-trades-${{m}}" class="sub-content"></div>
                <div id="sub-cutloss-${{m}}" class="sub-content"></div>
                <div id="sub-history-${{m}}" class="sub-content"></div>
                <div id="sub-about-${{m}}" class="sub-content"></div>
                <div class="actions" style="margin-top:14px">
                    <button class="btn btn-primary" onclick="triggerModelRun('${{m}}',false)">Run ${{m.toUpperCase()}}</button>
                    <button class="btn btn-secondary" onclick="triggerModelRun('${{m}}',true)">Dry Run</button>
                </div>
            </div>`;
        }});
        $('model-content').innerHTML = content;
    }}

    function switchModel(name, btn) {{
        activeModel = name;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        $('model-'+name).classList.add('active');
        refreshAll();
    }}

    function switchSub(model, sub, btn) {{
        activeSubTab[model] = sub;
        btn.parentElement.querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        btn.parentElement.parentElement.querySelectorAll('.sub-content').forEach(el => el.classList.remove('active'));
        $(`sub-${{sub}}-${{model}}`).classList.add('active');
        refreshAll();
    }}

    // ── Data loaders ────────────────────────────────────
    async function loadGlobalSummary() {{
        // Aggregate across all models
        let totalEquity = 0, totalPL = 0, totalPositions = 0;
        const modelSummaries = [];

        for (const m of MODELS) {{
            const acct = await api(`/api/account/${{m}}`);
            const pos = await api(`/api/positions/${{m}}`);
            if (acct) {{
                const dayPL = acct.equity - acct.last_equity;
                totalEquity += acct.equity;
                totalPL += dayPL;
                modelSummaries.push({{ name: m, equity: acct.equity, dayPL, positions: pos ? pos.length : 0 }});
                totalPositions += pos ? pos.length : 0;
            }}
        }}

        const status = await api('/api/status');
        const botState = status ? status.state : 'unknown';
        const stateColor = botState === 'idle' ? 'badge-green' : botState === 'running' ? 'badge-yellow' : 'badge-red';

        $('status-badge').innerHTML = `<span class="badge ${{stateColor}}">${{botState.toUpperCase()}}</span>`;

        let html = `
            <div class="metric">
                <div class="label">Total Portfolio</div>
                <div class="val">${{fmt$(totalEquity)}}</div>
                <div class="sub">${{MODELS.length}} model(s) active</div>
            </div>
            <div class="metric">
                <div class="label">Day P&L</div>
                <div class="val ${{cls(totalPL)}}">${{fmt$(totalPL)}}</div>
                <div class="sub">${{totalEquity > 0 ? fmtPct(totalPL / (totalEquity - totalPL) * 100) : '-'}}</div>
            </div>
            <div class="metric">
                <div class="label">Total Positions</div>
                <div class="val">${{totalPositions}}</div>
                <div class="sub">across all models</div>
            </div>`;

        modelSummaries.forEach(ms => {{
            html += `
            <div class="metric">
                <div class="label">${{ms.name.toUpperCase()}} Equity</div>
                <div class="val">${{fmt$(ms.equity)}}</div>
                <div class="sub"><span class="${{cls(ms.dayPL)}}">${{fmt$(ms.dayPL)}} today</span> &middot; ${{ms.positions}} pos</div>
            </div>`;
        }});

        if (status) {{
            html += `
            <div class="metric">
                <div class="label">Last Run</div>
                <div class="val" style="font-size:16px">${{status.last_run_at ? fmtET(status.last_run_at) : 'Never'}}</div>
                <div class="sub">${{status.last_run_duration ? status.last_run_duration + 's' : ''}}</div>
            </div>
            <div class="metric">
                <div class="label">Next Run</div>
                <div class="val" style="font-size:16px">${{status.next_run_at ? fmtET(status.next_run_at) : '-'}}</div>
                <div class="sub">Total: ${{status.total_runs}} runs</div>
            </div>`;

            if (status.last_error) {{
                $('error-card').style.display = 'block';
                $('error-content').textContent = status.last_error;
            }} else {{
                $('error-card').style.display = 'none';
            }}
        }}

        $('global-summary').innerHTML = html;
    }}

    async function loadPortfolio(model) {{
        const [acct, portfolio] = await Promise.all([
            api(`/api/account/${{model}}`),
            api(`/api/portfolio/${{model}}`)
        ]);

        let html = '<div class="grid-2">';

        // Account card
        html += '<div class="card"><h2>Account</h2>';
        if (acct) {{
            const dayPL = acct.equity - acct.last_equity;
            html += `
                <table>
                    <tr><td>Equity</td><td class="text-right mono"><strong>${{fmt$(acct.equity)}}</strong></td></tr>
                    <tr><td>Cash</td><td class="text-right mono">${{fmt$(acct.cash)}}</td></tr>
                    <tr><td>Long Market Value</td><td class="text-right mono">${{fmt$(acct.long_market_value)}}</td></tr>
                    <tr><td>Buying Power</td><td class="text-right mono">${{fmt$(acct.buying_power)}}</td></tr>
                    <tr><td>Day P&L</td><td class="text-right mono ${{cls(dayPL)}}">${{fmt$(dayPL)}} (${{fmtPct(acct.last_equity>0 ? dayPL/acct.last_equity*100 : 0)}})</td></tr>
                    <tr><td>Status</td><td class="text-right"><span class="badge badge-green">${{acct.status}}</span></td></tr>
                </table>`;
        }} else {{
            html += '<p style="color:var(--text-dim)">Unable to fetch account data</p>';
        }}
        html += '</div>';

        // Target portfolio card
        html += '<div class="card"><h2>Target Portfolio (Last Rebalance)</h2>';
        if (portfolio && portfolio.portfolio && portfolio.portfolio.length > 0) {{
            html += `<p style="font-size:12px;color:var(--text-dim);margin-bottom:8px">${{portfolio.date ? fmtET(portfolio.date) : ''}}</p>`;
            if (portfolio.regime_exposure != null) {{
                html += `<p style="font-size:12px;margin-bottom:8px">Regime exposure: <strong>${{(portfolio.regime_exposure*100).toFixed(0)}}%</strong></p>`;
            }}
            html += '<table><tr><th>#</th><th>Symbol</th><th class="text-right">Pred</th>';
            if (portfolio.conviction_weights && Object.keys(portfolio.conviction_weights).length > 0) {{
                html += '<th class="text-right">Weight</th>';
            }}
            html += '</tr>';
            portfolio.portfolio.forEach((sym, i) => {{
                const pred = portfolio.predictions[sym] || 0;
                const w = portfolio.conviction_weights ? portfolio.conviction_weights[sym] : null;
                html += `<tr>
                    <td>${{i+1}}</td>
                    <td><strong>${{sym}}</strong></td>
                    <td class="text-right mono ${{cls(pred)}}">${{pred > 0 ? '+' : ''}}${{pred.toFixed(2)}}%</td>`;
                if (w != null) html += `<td class="text-right mono">${{(w*100).toFixed(1)}}%</td>`;
                html += '</tr>';
            }});
            html += '</table>';
        }} else {{
            html += '<p style="color:var(--text-dim)">No portfolio yet</p>';
        }}
        html += '</div></div>';

        $(`sub-portfolio-${{model}}`).innerHTML = html;
    }}

    async function loadPositions(model) {{
        const positions = await api(`/api/positions/${{model}}`);
        let html = '<div class="card"><h2>Live Positions</h2>';

        if (positions && positions.length > 0) {{
            const totalMV = positions.reduce((s, p) => s + p.market_value, 0);
            const totalPL = positions.reduce((s, p) => s + p.unrealized_pl, 0);

            html += `<p style="font-size:13px;margin-bottom:10px">
                ${{positions.length}} positions &middot;
                Market value: <strong>${{fmt$(totalMV)}}</strong> &middot;
                Unrealized P&L: <strong class="${{cls(totalPL)}}">${{fmt$(totalPL)}}</strong>
            </p>`;

            html += `<table>
                <tr>
                    <th>Symbol</th><th class="text-right">Shares</th>
                    <th class="text-right">Avg Entry</th><th class="text-right">Price</th>
                    <th class="text-right">Mkt Value</th><th class="text-right">P&L</th>
                    <th class="text-right">P&L %</th><th class="text-right">Today</th>
                </tr>`;
            positions.forEach(p => {{
                html += `<tr>
                    <td><strong>${{p.symbol}}</strong></td>
                    <td class="text-right mono">${{fmtN(p.qty, p.qty % 1 === 0 ? 0 : 2)}}</td>
                    <td class="text-right mono">${{fmt$(p.avg_entry_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.current_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.market_value)}}</td>
                    <td class="text-right mono ${{cls(p.unrealized_pl)}}">${{fmt$(p.unrealized_pl)}}</td>
                    <td class="text-right mono ${{cls(p.unrealized_plpc)}}">${{fmtPct(p.unrealized_plpc)}}</td>
                    <td class="text-right mono ${{cls(p.change_today)}}">${{fmtPct(p.change_today)}}</td>
                </tr>`;
            }});
            html += '</table>';
        }} else {{
            html += '<p style="color:var(--text-dim)">No open positions</p>';
        }}
        html += '</div>';
        $(`sub-positions-${{model}}`).innerHTML = html;
    }}

    async function loadEquityCurve(model, period='1M') {{
        const container = $(`sub-chart-${{model}}`);

        // Period buttons
        let periodHTML = '<div class="period-sel">';
        ['1W','1M','3M','6M','1A','all'].forEach(p => {{
            periodHTML += `<button class="period-btn ${{p===period?'active':''}}" onclick="loadEquityCurve('${{model}}','${{p}}')">${{p}}</button>`;
        }});
        periodHTML += `</div><div class="card"><h2>Equity Curve</h2><div class="chart-container"><canvas id="chart-${{model}}"></canvas></div></div>`;
        container.innerHTML = periodHTML;

        const data = await api(`/api/history/${{model}}?period=${{period}}`);
        if (!data || !data.timestamps || data.timestamps.length === 0) {{
            container.querySelector('.card').innerHTML += '<p style="color:var(--text-dim)">No history data</p>';
            return;
        }}

        const labels = data.timestamps.map(ts => fmtET(ts, {{ mode: 'date' }}));

        const chartKey = `chart-${{model}}`;
        if (charts[chartKey]) charts[chartKey].destroy();

        const ctx = $(`chart-${{model}}`).getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 260);
        const lastPL = data.profit_loss_pct[data.profit_loss_pct.length - 1] || 0;
        const lineColor = lastPL >= 0 ? '#10b981' : '#ef4444';
        gradient.addColorStop(0, lineColor + '30');
        gradient.addColorStop(1, lineColor + '00');

        charts[chartKey] = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Equity',
                    data: data.equity,
                    borderColor: lineColor,
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: '#1a1d27',
                        borderColor: '#2a2d37',
                        borderWidth: 1,
                        titleColor: '#c9d1d9',
                        bodyColor: '#f0f6fc',
                        callbacks: {{
                            label: ctx => fmt$(ctx.parsed.y)
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#6b7280', maxTicksLimit: 12, font: {{ size: 10 }} }},
                        grid: {{ color: '#1e222d' }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#6b7280',
                            font: {{ size: 10 }},
                            callback: v => '$' + (v/1000).toFixed(v >= 10000 ? 0 : 1) + 'k'
                        }},
                        grid: {{ color: '#1e222d' }}
                    }}
                }}
            }}
        }});
    }}

    async function loadTrades(model) {{
        const trades = await api(`/api/trades/${{model}}?limit=100`);
        let html = '<div class="card"><h2>Trade History</h2>';

        if (trades && trades.length > 0) {{
            const isAlpaca = trades[0].source === 'alpaca';
            if (isAlpaca) html += '<p style="font-size:11px;color:var(--text-dim);margin-bottom:8px">Source: Alpaca order history</p>';

            html += `<table>
                <tr><th>Time</th><th>Symbol</th><th>Side</th><th class="text-right">Shares</th>
                    <th class="text-right">Price</th><th class="text-right">Notional</th>
                    <th>Status</th></tr>`;
            trades.forEach(t => {{
                const action = t.trade_action || t.side || t.action || '-';
                const actionCls = action.includes('buy') ? 'green' :
                                  action.includes('sell') ? 'red' : '';
                const displayTs = fmtET(t.timestamp || '');
                html += `<tr>
                    <td class="mono" style="font-size:11px">${{displayTs}}</td>
                    <td><strong>${{t.symbol || '-'}}</strong></td>
                    <td class="${{actionCls}}">${{action.toUpperCase()}}</td>
                    <td class="text-right mono">${{t.shares ? fmtN(t.shares, t.shares % 1 === 0 ? 0 : 2) : '-'}}</td>
                    <td class="text-right mono">${{t.filled_price ? fmt$(t.filled_price) : '-'}}</td>
                    <td class="text-right mono">${{t.notional ? fmt$(t.notional) : '-'}}</td>
                    <td>${{t.order_status || '-'}}</td>
                </tr>`;
            }});
            html += '</table>';
        }} else {{
            html += '<p style="color:var(--text-dim)">No trades recorded yet</p>';
        }}
        html += '</div>';
        $(`sub-trades-${{model}}`).innerHTML = html;
    }}

    async function loadCutloss(model) {{
        // Fetch summary stats + event list in parallel so the panel paints once.
        const [stats, events] = await Promise.all([
            api(`/api/cutloss-stats/${{model}}`),
            api(`/api/cutloss/${{model}}`),
        ]);

        let html = '';

        // -- Summary stats card (today / 7d / 30d) -------------------------
        if (stats && stats.windows) {{
            const w = stats.windows;
            const fmtNotional = (n) => '$' + (n || 0).toLocaleString('en-US', {{maximumFractionDigits: 0}});
            const winColumn = (label, key) => {{
                const x = w[key] || {{}};
                const triggers = x.by_trigger || {{}};
                const totalEvents = x.events || 0;
                const sub = totalEvents === 0
                    ? '<div style="color:var(--text-dim);font-size:11px;margin-top:6px">no events</div>'
                    : `<div style="font-size:11px;color:var(--text-dim);margin-top:6px;line-height:1.5">
                        <div>${{x.per_position_stops || 0}} per-position stop${{x.per_position_stops === 1 ? '' : 's'}}</div>
                        <div>${{x.soft_scale_partials || 0}} soft-scale partial${{x.soft_scale_partials === 1 ? '' : 's'}}</div>
                        <div>${{x.portfolio_stops || 0}} portfolio liquidation${{x.portfolio_stops === 1 ? '' : 's'}}</div>
                       </div>`;
                const notionalLine = totalEvents > 0
                    ? `<div style="font-size:11px;color:var(--text-dim);margin-top:4px">${{fmtNotional(x.notional)}} notional</div>`
                    : '';
                return `<div style="background:#0a0c10;border:1px solid var(--card-border);border-radius:6px;padding:12px;flex:1;min-width:160px">
                    <div style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">${{label}}</div>
                    <div style="font-size:24px;font-weight:700;color:var(--text-bright)">${{totalEvents}}</div>
                    <div style="font-size:11px;color:var(--text-dim)">total event${{totalEvents === 1 ? '' : 's'}}</div>
                    ${{sub}}
                    ${{notionalLine}}
                </div>`;
            }};
            const last = stats.last_event;
            const lastLine = last
                ? `<div style="font-size:11px;color:var(--text-dim);margin-top:14px">
                    Latest: ${{fmtET(last.timestamp)}} &middot; <strong>${{last.symbol}}</strong> &middot;
                    ${{(last.trigger || '').replace('_', ' ')}}
                    ${{last.notional ? ' &middot; ' + fmtNotional(last.notional) : ''}}
                   </div>`
                : '';
            html += `<div class="card">
                <h2>&#9888; Cut-Loss Activity</h2>
                <p style="font-size:12px;color:var(--text-dim);margin-bottom:14px">
                    Per-position stops: hard (-8% from entry), trailing (-5% from peak).
                    Soft scale: Tier 1 (-3% daily DD) sells to 60% exposure; Tier 2 (-5%) to 30%.
                    Portfolio liquidation: Tier 3 (-7%) closes everything + trips today's flag.
                </p>
                <div style="display:flex;gap:10px;flex-wrap:wrap">
                    ${{winColumn('Today', 'today')}}
                    ${{winColumn('Last 7 days', 'last_7d')}}
                    ${{winColumn('Last 30 days', 'last_30d')}}
                </div>
                ${{lastLine}}
            </div>`;
        }}

        // -- Event list ----------------------------------------------------
        html += '<div class="card"><h2>Recent events</h2>';
        if (events && events.length > 0) {{
            html += `<p style="font-size:12px;color:var(--text-dim);margin-bottom:10px">
                Showing ${{events.length}} cutloss event(s) — newest first.
            </p>`;
            html += `<table>
                <tr><th>Time</th><th>Symbol</th><th>Trigger</th><th>Detail</th>
                    <th>Shares</th><th>Status</th><th>Source</th></tr>`;
            events.forEach(e => {{
                const displayTs = fmtET(e.timestamp || '');
                const triggerLabel = (e.trigger || '').replace(/_/g, ' ').toUpperCase();
                const isLiquidation = e.trigger === 'portfolio_stop';
                const isPerPositionHard = e.trigger === 'hard_stop';
                const isSoftScale = e.trigger === 'soft_scale_tier1' || e.trigger === 'soft_scale_tier2';
                const triggerCls = isLiquidation ? 'red'
                                 : isPerPositionHard ? 'red'
                                 : isSoftScale ? 'green'
                                 : 'yellow';
                const statusCls = (e.order_status === 'rejected' || e.order_status === 'canceled') ? 'red' : '';
                html += `<tr>
                    <td class="mono" style="font-size:11px">${{displayTs}}</td>
                    <td><strong>${{e.symbol || '-'}}</strong></td>
                    <td><span class="badge badge-${{triggerCls}}" style="font-size:10px">${{triggerLabel}}</span></td>
                    <td class="mono" style="font-size:11px">${{e.detail || '-'}}</td>
                    <td class="text-right mono">${{e.shares ? fmtN(e.shares, 2) : '-'}}</td>
                    <td class="${{statusCls}}">${{e.order_status || '-'}}</td>
                    <td style="font-size:10px;color:var(--text-dim)">${{e.source || '-'}}</td>
                </tr>`;
            }});
            html += '</table>';
        }} else {{
            html += '<p style="color:var(--text-dim)">No cutloss events recorded yet.</p>';
            html += '<p style="font-size:12px;color:var(--text-dim)">Events will appear here when the cut-loss scanner triggers a sell during market hours.</p>';
        }}
        html += '</div>';
        $(`sub-cutloss-${{model}}`).innerHTML = html;
    }}

    async function loadRunHistory(model) {{
        const state = await api(`/api/status`);
        const modelData = state && state.models ? state.models[model] : null;
        let html = '<div class="card"><h2>Run History</h2>';

        if (modelData && modelData.history && modelData.history.length > 0) {{
            html += `<table>
                <tr><th>Date</th><th>Stocks</th><th>Sells</th><th>Buys</th></tr>`;
            modelData.history.slice().reverse().forEach(entry => {{
                const date = fmtET(entry.date || '');
                const nStocks = (entry.target_symbols || []).length;
                const result = entry.result || {{}};
                html += `<tr>
                    <td class="mono">${{date}}</td>
                    <td>${{nStocks}}</td>
                    <td>${{result.n_sells != null ? result.n_sells : '?'}}</td>
                    <td>${{result.n_buys != null ? result.n_buys : '?'}}</td>
                </tr>`;
            }});
            html += '</table>';
        }} else {{
            html += '<p style="color:var(--text-dim)">No run history yet</p>';
        }}
        html += '</div>';
        $(`sub-history-${{model}}`).innerHTML = html;
    }}

    async function loadAbout(model) {{
        const data = await api(`/api/description/${{model}}`);
        let html = '<div class="card">';

        if (!data || data.error) {{
            html += `<h2>About ${{model.toUpperCase()}}</h2>`;
            html += `<p style="color:var(--text-dim)">No description available</p>`;
            html += '</div>';
            $(`sub-about-${{model}}`).innerHTML = html;
            return;
        }}

        html += `<h2 style="color:var(--text-bright);font-size:16px;margin-bottom:4px;text-transform:none;letter-spacing:0">${{data.title || model.toUpperCase()}}</h2>`;
        html += `<p style="color:var(--text);margin-bottom:18px;font-size:13px;line-height:1.6">${{data.summary}}</p>`;

        const sections = [
            {{ icon: '&#9881;', label: 'Architecture', key: 'architecture' }},
            {{ icon: '&#128202;', label: 'Features', key: 'features' }},
            {{ icon: '&#128188;', label: 'Portfolio Strategy', key: 'portfolio' }},
            {{ icon: '&#128737;', label: 'Risk Management', key: 'risk' }},
            {{ icon: '&#127919;', label: 'Training', key: 'training' }},
        ];

        sections.forEach(sec => {{
            if (data[sec.key]) {{
                html += `<div style="margin-bottom:14px">
                    <div style="color:var(--text-dim);font-size:11px;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;font-weight:600">
                        ${{sec.icon}} ${{sec.label}}
                    </div>
                    <p style="color:var(--text);font-size:13px;line-height:1.6;
                              white-space:pre-line">${{data[sec.key]}}</p>
                </div>`;
            }}
        }});

        if (data.enable_cutloss && data.cutloss_config) {{
            html += `<div style="margin-top:14px;padding:14px;background:#1a1c24;border-radius:8px;
                                border:1px solid var(--yellow)30">
                <div style="color:var(--yellow);font-size:12px;font-weight:600;margin-bottom:8px">
                    &#9888; ACTIVE CUT-LOSS PROTECTION
                </div>
                <table style="font-size:12px">
                    <tr><td style="padding:3px 10px 3px 0">Hard stop (from entry)</td>
                        <td class="mono red">${{data.cutloss_config.hard_stop}}</td></tr>
                    <tr><td style="padding:3px 10px 3px 0">Trailing stop (from peak)</td>
                        <td class="mono red">${{data.cutloss_config.trailing_stop}}</td></tr>
                    <tr><td style="padding:3px 10px 3px 0">Portfolio drawdown stop</td>
                        <td class="mono red">${{data.cutloss_config.portfolio_stop}}</td></tr>
                </table>
                <p style="color:var(--text-dim);font-size:11px;margin-top:8px">
                    Scanner runs every 60 seconds during market hours (9:30 AM - 4:00 PM ET)
                </p>
            </div>`;
        }}

        html += '</div>';
        $(`sub-about-${{model}}`).innerHTML = html;
    }}

    // Track the active window per model so switching tabs doesn't reset.
    const perfActiveWindow = {{}};
    const PERF_WINDOW_LABELS = {{
        '1d': 'Last day', '7d': 'Last 7 days', '30d': 'Last 30 days', 'inception': 'Since inception'
    }};
    const PERF_WINDOW_ORDER = ['1d', '7d', '30d', 'inception'];

    function perfPctileNarrative(p, model, windowLabel) {{
        if (p == null) return '';
        const beat = p.toFixed(0);
        const beneath = (100 - p).toFixed(0);
        let tier = 'in the middle of';
        if (p >= 90) tier = 'in the top 10% of';
        else if (p >= 75) tier = 'in the top quartile of';
        else if (p >= 50) tier = 'better than half of';
        else if (p >= 25) tier = 'in the bottom half of';
        else if (p >= 10) tier = 'in the bottom quartile of';
        else tier = 'in the bottom 10% of';
        return `${{model.toUpperCase()}} is ${{tier}} the universe over the ${{windowLabel.toLowerCase()}} — beating ${{beat}}% of stocks, behind ${{beneath}}%.`;
    }}

    function perfPctileColor(p) {{
        if (p == null) return 'var(--text-dim)';
        if (p >= 75) return 'var(--green)';
        if (p >= 50) return '#84cc16';        // lime
        if (p >= 25) return 'var(--yellow)';
        return 'var(--red)';
    }}

    function renderHistogram(hist, modelReturn) {{
        if (!hist || !hist.counts) return '';
        const max = Math.max(...hist.counts, 1);
        const bars = hist.counts.map((c, i) => {{
            const h = c / max * 70;  // px
            const x = i / hist.counts.length * 100;
            const w = 100 / hist.counts.length;
            return `<div style="position:absolute;left:${{x}}%;width:calc(${{w}}% - 1px);
                                bottom:0;height:${{h}}px;background:var(--card-border);
                                border-radius:1px;"
                         title="${{c}} stocks"></div>`;
        }}).join('');
        // Marker for the model's return
        let marker = '';
        if (modelReturn != null) {{
            const clipped = Math.max(hist.lo, Math.min(hist.hi - 0.001, modelReturn));
            const x = (clipped - hist.lo) / (hist.hi - hist.lo) * 100;
            marker = `
                <div style="position:absolute;left:${{x}}%;top:-4px;bottom:-4px;width:2px;
                            background:var(--accent);transform:translateX(-1px);"></div>
                <div style="position:absolute;left:${{x}}%;top:-22px;font-size:10px;
                            color:var(--accent);font-weight:600;font-family:var(--mono);
                            transform:translateX(-50%);white-space:nowrap;">
                    ${{fmtPct(modelReturn)}}
                </div>`;
        }}
        return `
            <div style="position:relative;height:90px;margin:24px 0 6px 0;border-bottom:1px solid var(--card-border);">
                ${{bars}}${{marker}}
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-dim);font-family:var(--mono);">
                <span>${{fmtPct(hist.lo)}}</span>
                <span>0%</span>
                <span>${{fmtPct(hist.hi)}}</span>
            </div>
            <p style="font-size:11px;color:var(--text-dim);margin:6px 0 0 0;text-align:center;">
                Distribution of ${{hist.total}} universe-stock returns. Blue line = ${{modelReturn != null ? fmtPct(modelReturn) : '?'}} (model).
            </p>`;
    }}

    function renderPerfWindow(model, data, label) {{
        const w = data.windows[label];
        if (!w) return '<p style="color:var(--text-dim);">No data for this window.</p>';

        const help = data.indicator_help || {{}};
        const windowLabel = PERF_WINDOW_LABELS[label] || label;
        const modelRet = w.model_return;
        const pctile = w.univ_pctile;
        const pctileColor = perfPctileColor(pctile);

        const helpRow = (k, value, valueClass) => `
            <tr>
                <td style="color:var(--text);"><strong>${{k.label}}</strong>
                    <div style="font-size:11px;color:var(--text-dim);font-weight:400;line-height:1.4;margin-top:2px;">
                        ${{k.help || ''}}
                    </div>
                </td>
                <td class="text-right mono ${{valueClass || ''}}" style="vertical-align:top;font-size:14px;font-weight:600;white-space:nowrap;">
                    ${{value}}
                </td>
            </tr>`;

        let html = '';

        // Hero — percentile rank (the headline number)
        html += `<div style="background:linear-gradient(135deg, rgba(59,130,246,0.08), transparent);
                              border:1px solid var(--card-border);border-radius:10px;
                              padding:18px 22px;margin-bottom:18px;">
            <div style="font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;">
                ${{windowLabel}} — Percentile Rank
            </div>
            <div style="font-size:42px;font-weight:700;color:${{pctileColor}};font-family:var(--mono);
                        line-height:1.1;margin:6px 0 4px 0;">
                ${{pctile != null ? pctile.toFixed(0) + '<span style="font-size:22px;color:var(--text-dim);"> / 100</span>' : '—'}}
            </div>
            <div style="font-size:13px;color:var(--text);">
                ${{perfPctileNarrative(pctile, model, windowLabel)}}
            </div>
            <div style="font-size:11px;color:var(--text-dim);margin-top:10px;line-height:1.5;">
                Comparison universe: <strong>${{w.univ_n}}</strong> of ${{data.universe_size_total}} tradeable stocks
                ${{data.data_feed ? `· data feed: <code style="color:var(--text);background:var(--card-border);padding:1px 5px;border-radius:3px;font-size:10px;">${{data.data_feed.toUpperCase()}}</code>` : ''}}
                ${{(data.universe_size_total - w.univ_n) > 5
                    ? `<br><span style="color:var(--text-dim);">${{data.universe_size_total - w.univ_n}} stocks lacked price data for this window — likely names with no trades on the ${{(data.data_feed || 'iex').toUpperCase()}} feed during the period.</span>`
                    : ''}}
            </div>
        </div>`;

        // Comparison table with explanations
        html += '<table style="width:100%;font-size:13px;">';
        html += helpRow(
            {{ label: `${{model.toUpperCase()}} return`, help: help.model_return }},
            fmtPct(modelRet),
            cls(modelRet)
        );
        html += helpRow(
            {{ label: 'Universe median', help: help.univ_median }},
            fmtPct(w.univ_median),
            cls(w.univ_median)
        );
        html += helpRow(
            {{ label: 'Universe mean', help: help.univ_mean }},
            fmtPct(w.univ_mean),
            cls(w.univ_mean)
        );
        if (w.spy_return != null) {{
            html += helpRow(
                {{ label: 'SPY return', help: help.spy_return }},
                fmtPct(w.spy_return),
                cls(w.spy_return)
            );
        }}
        html += helpRow(
            {{ label: 'Alpha vs median', help: help.alpha_vs_median }},
            fmtPct(w.alpha_vs_median),
            cls(w.alpha_vs_median)
        );
        if (w.alpha_vs_spy != null) {{
            html += helpRow(
                {{ label: 'Alpha vs SPY', help: help.alpha_vs_spy }},
                fmtPct(w.alpha_vs_spy),
                cls(w.alpha_vs_spy)
            );
        }}
        html += helpRow(
            {{ label: 'Universe % positive', help: help.univ_pct_positive }},
            (w.univ_pct_positive != null ? w.univ_pct_positive.toFixed(0) + '%' : '—'),
            ''
        );
        html += helpRow(
            {{ label: 'Universe std dev', help: help.univ_std }},
            (w.univ_std != null ? w.univ_std.toFixed(2) + '%' : '—'),
            ''
        );
        html += '</table>';

        // Distribution histogram
        html += `<h3 style="color:var(--text-dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:18px 0 0 0;">
            Where ${{model.toUpperCase()}} sits in the universe distribution
        </h3>`;
        html += renderHistogram(w.univ_histogram, modelRet);

        return html;
    }}

    function renderPerfWindowTabs(model, data, activeLabel) {{
        let html = '<div class="sub-tabs" style="margin:8px 0 16px 0;">';
        PERF_WINDOW_ORDER.forEach(label => {{
            const w = data.windows[label] || {{}};
            const isActive = label === activeLabel;
            const ret = w.model_return;
            const retStr = ret != null ? fmtPct(ret) : '—';
            const retCls = ret != null ? cls(ret) : '';
            html += `<button class="sub-tab ${{isActive ? 'active' : ''}}"
                             onclick="setPerfWindow('${{model}}','${{label}}')">
                ${{PERF_WINDOW_LABELS[label]}}
                <span style="font-size:10px;font-family:var(--mono);margin-left:4px;
                             color:${{isActive ? '#fff' : 'var(--text-dim)'}};">
                    ${{retStr}}
                </span>
            </button>`;
        }});
        html += '</div>';
        return html;
    }}

    let _perfDataCache = {{}};

    function setPerfWindow(model, label) {{
        perfActiveWindow[model] = label;
        const data = _perfDataCache[model];
        if (data) renderPerformanceCard(model, data);
    }}

    function renderPerformanceCard(model, data) {{
        const container = $(`sub-performance-${{model}}`);
        const activeLabel = perfActiveWindow[model] || data.headline?.window || 'inception';
        perfActiveWindow[model] = activeLabel;

        let html = '<div class="card">';
        html += `<h2>Performance vs Universe</h2>`;
        html += `<p style="font-size:12px;color:var(--text-dim);margin:-4px 0 4px 0;">
            How the ${{model.toUpperCase()}} model's actual equity return compares to a buy-and-hold of every other stock in its tradeable universe.
            <span style="display:block;margin-top:4px;color:var(--text-dim);font-size:11px;">
                As of ${{data.as_of ? fmtET(data.as_of) : '?'}}.
                Universe bars cached for 1 hour.
            </span>
        </p>`;

        html += renderPerfWindowTabs(model, data, activeLabel);
        html += renderPerfWindow(model, data, activeLabel);

        // Held positions (de-emphasized footer)
        const positions = data.current_positions || [];
        if (positions.length > 0) {{
            html += `<details style="margin-top:24px;">
                <summary style="cursor:pointer;color:var(--text-dim);font-size:11px;
                                text-transform:uppercase;letter-spacing:1px;
                                padding:6px 0;">
                    Currently held positions (${{positions.length}})
                    <span style="text-transform:none;letter-spacing:0;font-size:11px;margin-left:6px;">
                        — unrealized P&L on still-open trades; sub-window of total return
                    </span>
                </summary>
                <table style="margin-top:8px;">
                    <tr><th>#</th><th>Symbol</th><th class="text-right">Entry</th>
                        <th class="text-right">Current</th><th class="text-right">Value</th>
                        <th class="text-right">P&L</th></tr>`;
            positions.forEach((p, i) => {{
                html += `<tr>
                    <td>${{i + 1}}</td>
                    <td><strong>${{p.symbol}}</strong></td>
                    <td class="text-right mono">${{fmt$(p.avg_entry_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.current_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.market_value)}}</td>
                    <td class="text-right mono ${{cls(p.unrealized_plpc)}}">${{fmtPct(p.unrealized_plpc)}}</td>
                </tr>`;
            }});
            html += '</table></details>';
        }}

        html += '</div>';
        container.innerHTML = html;
    }}

    async function loadPerformance(model) {{
        const container = $(`sub-performance-${{model}}`);
        container.innerHTML = '<div class="card"><h2>Performance vs Universe</h2><p style="color:var(--text-dim)"><span class="spinner"></span> Computing benchmark — fetching universe bars (10–60s on first load, cached 1h after).</p></div>';

        const data = await api(`/api/performance/${{model}}`);
        if (!data || (data.error && !data.windows)) {{
            container.innerHTML = `<div class="card"><h2>Performance vs Universe</h2>
                <p style="color:var(--red)">${{data ? data.error : 'Failed to fetch performance data'}}</p>
                </div>`;
            return;
        }}
        _perfDataCache[model] = data;
        renderPerformanceCard(model, data);
    }}

    let allLogs = [];
    let currentLogFilter = 'all';
    let currentLogFile = null;
    let liveTailHandle = null;
    const CRITICAL_TOKENS = [
        'PORTFOLIO STOP', 'HARD STOP', 'TRAILING STOP',
        'LIQUIDATING', 'FATAL', 'FAILED', 'circuit breaker'
    ];

    async function loadLogsList() {{
        allLogs = await api('/api/logs') || [];
        const models = [...new Set(allLogs.map(l => l.model).filter(m => m && m !== '?' && m !== 'ALL'))];
        const modelSel = $('log-model-filter');
        const prev = modelSel.value;
        modelSel.innerHTML = '<option value="">All models</option>';
        models.sort().forEach(m => {{
            modelSel.innerHTML += `<option value="${{m}}">${{m}}</option>`;
        }});
        if (prev) modelSel.value = prev;
        filterLogs(currentLogFilter);

        // First-paint convenience: auto-load the most recent log if nothing
        // is currently being viewed.
        if (!currentLogFile && allLogs.length > 0) {{
            loadLog(allLogs[0].filename);
        }}
    }}

    function filterLogs(category) {{
        currentLogFilter = category;
        ['all','pipeline','cutloss'].forEach(c => {{
            const btn = $(`log-filter-${{c}}`);
            if (btn) btn.className = 'sub-tab' + (c === category ? ' active' : '');
        }});

        const modelFilter = $('log-model-filter').value;
        const filtered = allLogs.filter(l => {{
            if (category !== 'all' && l.category !== category) return false;
            if (modelFilter && l.model !== modelFilter && l.model !== 'ALL') return false;
            return true;
        }});

        const list = $('log-list');
        if (filtered.length === 0) {{
            list.innerHTML = '<p style="color:var(--text-dim);font-size:12px;padding:8px;">No log files found.</p>';
            return;
        }}

        // Group by date — most recent first
        const byDate = new Map();
        filtered.forEach(l => {{
            const d = l.date || '?';
            if (!byDate.has(d)) byDate.set(d, []);
            byDate.get(d).push(l);
        }});

        let html = '';
        for (const [date, files] of byDate) {{
            const npipe = files.filter(f => f.category === 'pipeline').length;
            const ncut = files.filter(f => f.category === 'cutloss').length;
            html += `<div style="margin:6px 0 2px 0;color:var(--text-dim);font-size:11px;
                                 text-transform:uppercase;letter-spacing:1px;">
                ${{date}} <span style="color:var(--text-dim);text-transform:none;letter-spacing:0;
                                       font-size:10px;margin-left:6px;">
                    ${{npipe}} pipeline · ${{ncut}} cut-loss
                </span>
            </div>`;
            html += '<table style="width:100%;font-size:12px;border-collapse:collapse;">';
            files.forEach(l => {{
                const typeBadge = l.category === 'cutloss'
                    ? '<span style="background:#ef444433;color:#ef4444;padding:1px 6px;border-radius:4px;font-size:10px;">CUT-LOSS</span>'
                    : '<span style="background:#3b82f633;color:#3b82f6;padding:1px 6px;border-radius:4px;font-size:10px;">PIPELINE</span>';
                const isActive = (l.filename === currentLogFile);
                html += `<tr style="cursor:pointer;border-bottom:1px solid var(--card-border);
                                    background:${{isActive ? 'rgba(59,130,246,0.12)' : 'transparent'}};"
                             onclick="loadLog('${{l.filename}}')"
                             onmouseover="this.style.background='var(--card-border)'"
                             onmouseout="this.style.background='${{isActive ? 'rgba(59,130,246,0.12)' : 'transparent'}}'">
                    <td style="padding:5px 8px;font-family:var(--mono);color:var(--accent);">${{l.filename}}</td>
                    <td style="padding:5px 8px;text-align:center;">${{typeBadge}}</td>
                    <td style="padding:5px 8px;text-align:center;font-weight:600;">${{l.model}}</td>
                    <td style="padding:5px 8px;text-align:right;font-family:var(--mono);">${{l.size_kb}}kb</td>
                </tr>`;
            }});
            html += '</table>';
        }}
        list.innerHTML = html;
    }}

    function classifyLine(line) {{
        // Critical events take priority over level so PORTFOLIO STOP at INFO
        // still gets the red banner.
        for (const token of CRITICAL_TOKENS) {{
            if (line.indexOf(token) !== -1) return 'crit';
        }}
        if (line.indexOf('[ERROR]') !== -1) return 'err';
        if (line.indexOf('[WARNING]') !== -1) return 'warn';
        return 'info';
    }}

    function escapeHtml(s) {{
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}

    function renderLogContent(content) {{
        const lines = content.split('\\n');
        const out = [];
        for (const ln of lines) {{
            const cls = classifyLine(ln);
            out.push('<span class="log-line ' + cls + '">' + escapeHtml(ln || '​') + '</span>');
        }}
        return out.join('');
    }}

    function buildLogQuery() {{
        const tail = $('log-tail').value;
        const level = $('log-level').value;
        const params = new URLSearchParams();
        if (tail) params.set('tail', tail);
        if (level) params.set('level', level);
        const qs = params.toString();
        return qs ? '?' + qs : '';
    }}

    async function loadLog(filename, opts) {{
        opts = opts || {{}};
        if (!filename) return;
        currentLogFile = filename;
        if (!opts.silent) $('log-box').innerHTML = '<span style="color:var(--text-dim);">Loading…</span>';
        $('log-download-btn').disabled = false;

        const data = await api(`/api/logs/${{filename}}` + buildLogQuery());
        if (!data) return;

        $('log-box').innerHTML = renderLogContent(data.content);
        const meta = $('log-meta');
        if (data.total_lines != null) {{
            meta.textContent = `${{data.returned_lines}} / ${{data.total_lines}} lines`;
        }}

        applyLogSearch();

        // Refresh the file list so the active row gets highlighted.
        if (!opts.silent) filterLogs(currentLogFilter);

        // Stick to bottom unless the user has scrolled up.
        const box = $('log-box');
        if (opts.silent) {{
            // live tail: only auto-scroll if user was already at bottom
            const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
            if (wasAtBottom) box.scrollTop = box.scrollHeight;
        }} else {{
            box.scrollTop = box.scrollHeight;
        }}
    }}

    function reloadCurrentLog() {{
        if (currentLogFile) loadLog(currentLogFile);
    }}

    function applyLogSearch() {{
        const q = ($('log-search').value || '').trim().toLowerCase();
        const lines = $('log-box').querySelectorAll('.log-line');
        let matches = 0;
        lines.forEach(el => {{
            el.classList.remove('match', 'hidden');
            if (!q) return;
            const text = el.textContent.toLowerCase();
            if (text.indexOf(q) === -1) {{
                el.classList.add('hidden');
            }} else {{
                el.classList.add('match');
                matches += 1;
            }}
        }});
        const meta = $('log-meta');
        if (q && meta && currentLogFile) {{
            meta.textContent = `${{matches}} match${{matches === 1 ? '' : 'es'}} for "${{q}}"`;
        }} else if (currentLogFile) {{
            // Restore the line counter from the box's data
            const total = $('log-box').querySelectorAll('.log-line').length;
            meta.textContent = `${{total}} lines`;
        }}
    }}

    function toggleLiveTail() {{
        const cb = $('log-live');
        // The label fires onclick; checkbox state has already toggled by then
        const labelSpan = $('log-live-label');
        if (cb.checked) {{
            labelSpan.classList.add('live-on');
            labelSpan.textContent = 'Live tail (5s)';
            if (liveTailHandle) clearInterval(liveTailHandle);
            liveTailHandle = setInterval(() => {{
                if (currentLogFile) loadLog(currentLogFile, {{ silent: true }});
            }}, 5000);
        }} else {{
            labelSpan.classList.remove('live-on');
            labelSpan.textContent = 'Live tail';
            if (liveTailHandle) {{ clearInterval(liveTailHandle); liveTailHandle = null; }}
        }}
    }}

    function downloadCurrentLog() {{
        if (!currentLogFile) return;
        window.location.href = `/api/logs/${{currentLogFile}}/download`;
    }}

    // ── Triggers ────────────────────────────────────────
    async function triggerRun(dry) {{
        const url = dry ? '/run?dry=1&force=1' : '/run?force=1';
        const r = await api(url);
        if (r) alert(dry ? 'Dry run triggered for all models' : 'Pipeline triggered for all models');
        setTimeout(refreshAll, 2000);
    }}

    async function triggerModelRun(model, dry) {{
        const url = dry ? `/run?model=${{model}}&dry=1&force=1` : `/run?model=${{model}}&force=1`;
        const r = await api(url);
        if (r) alert(dry ? `Dry run triggered for ${{model.toUpperCase()}}` : `Pipeline triggered for ${{model.toUpperCase()}}`);
        setTimeout(refreshAll, 2000);
    }}

    // ── Master refresh ──────────────────────────────────
    async function refreshAll() {{
        const bar = $('refresh-bar');
        bar.style.width = '30%';

        try {{
            await loadGlobalSummary();
            bar.style.width = '50%';

            // Only refresh active model's active sub-tab
            if (activeModel) {{
                const sub = activeSubTab[activeModel] || 'portfolio';
                if (sub === 'portfolio') await loadPortfolio(activeModel);
                else if (sub === 'positions') await loadPositions(activeModel);
                else if (sub === 'chart') await loadEquityCurve(activeModel);
                else if (sub === 'performance') await loadPerformance(activeModel);
                else if (sub === 'trades') await loadTrades(activeModel);
                else if (sub === 'cutloss') await loadCutloss(activeModel);
                else if (sub === 'history') await loadRunHistory(activeModel);
                else if (sub === 'about') await loadAbout(activeModel);
            }}
        }} catch(e) {{
            console.error('Refresh error:', e);
        }}

        bar.style.width = '100%';
        setTimeout(() => bar.style.width = '0%', 300);
    }}

    // ── Settings ────────────────────────────────────────
    let settingsOpen = false;
    let currentConfig = null;

    function toggleSettings() {{
        settingsOpen = !settingsOpen;
        $('settings-content').style.display = settingsOpen ? 'block' : 'none';
        $('settings-toggle').innerHTML = settingsOpen ? '&#9650;' : '&#9660;';
        if (settingsOpen && !currentConfig) loadSettings();
    }}

    async function loadSettings() {{
        const data = await api('/api/config/models');
        if (!data) return;
        currentConfig = data;
        renderSlots(data);
    }}

    function renderSlots(config) {{
        const available = config.available_models || ['v4','v5','v6','v7','v8','v9'];
        const slots = config.slots || [];
        // Show as many slots as exist in the config, with a minimum of 4
        // so v9 can be added even on a config file that pre-dates the v9 slot.
        const nSlots = Math.max(slots.length, 4);
        let html = '';

        for (let i = 0; i < nSlots; i++) {{
            const slot = slots[i] || {{ slot_id: i+1, model: '', enabled: true, alpaca_key: '', alpaca_secret: '' }};
            const enabled = slot.enabled !== false;
            const hasKey = slot.alpaca_key && slot.alpaca_key.length > 0;
            const modelExists = slot.model_file_exists !== false;

            html += `<div class="card" style="margin-bottom:12px;border-color:${{enabled && hasKey ? 'var(--green)' : 'var(--card-border)'}}">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
                    <div style="display:flex;align-items:center;gap:10px">
                        <strong style="color:var(--text-bright);font-size:14px">Slot ${{i+1}}</strong>
                        ${{enabled && hasKey ? '<span class="badge badge-green">ACTIVE</span>' :
                          !enabled ? '<span class="badge" style="background:#4b5563">DISABLED</span>' :
                          '<span class="badge badge-yellow">NO KEYS</span>'}}
                    </div>
                    <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-dim);cursor:pointer">
                        <input type="checkbox" id="slot-enabled-${{i}}" ${{enabled ? 'checked' : ''}}
                               onchange="updateSlotStatus(${{i}})"
                               style="accent-color:var(--green)"> Enabled
                    </label>
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
                    <div>
                        <label style="font-size:11px;color:var(--text-dim);display:block;margin-bottom:4px">MODEL</label>
                        <select id="slot-model-${{i}}" style="width:100%"
                                onchange="updateSlotStatus(${{i}})">
                            ${{available.map(m => `<option value="${{m}}" ${{slot.model === m ? 'selected' : ''}}>${{m.toUpperCase()}}</option>`).join('')}}
                        </select>
                    </div>
                    <div style="display:flex;align-items:flex-end;gap:6px">
                        <div style="flex:1">
                            <label style="font-size:11px;color:var(--text-dim);display:block;margin-bottom:4px">STATUS</label>
                            <div id="slot-status-${{i}}" style="font-size:12px;padding:6px 0">
                                ${{modelExists ? '<span class="green">Model file found</span>' : '<span class="red">Model file missing</span>'}}
                            </div>
                        </div>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
                    <div>
                        <label style="font-size:11px;color:var(--text-dim);display:block;margin-bottom:4px">ALPACA API KEY</label>
                        <input type="text" id="slot-key-${{i}}" value="${{slot.alpaca_key || ''}}"
                               placeholder="PK..."
                               style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                      padding:6px 10px;border-radius:6px;font-size:12px;font-family:var(--mono)">
                    </div>
                    <div>
                        <label style="font-size:11px;color:var(--text-dim);display:block;margin-bottom:4px">ALPACA SECRET KEY</label>
                        <input type="password" id="slot-secret-${{i}}" value="${{slot.alpaca_secret || ''}}"
                               placeholder="Secret..."
                               style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                      padding:6px 10px;border-radius:6px;font-size:12px;font-family:var(--mono)">
                    </div>
                </div>

                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                    <button class="btn btn-secondary" onclick="testKey(${{i}})" style="font-size:11px;padding:5px 12px">
                        Test Key
                    </button>
                    <span id="slot-test-${{i}}" style="font-size:11px"></span>
                </div>

                <div style="border-top:1px solid var(--card-border);padding-top:10px;margin-bottom:10px">
                    <label style="font-size:11px;color:var(--text-dim);display:block;margin-bottom:4px">
                        TARGET LEVERAGE
                        <span style="font-size:10px;color:var(--text-dim);font-weight:normal">
                          (1.0 = cash account, 2.0 = Reg-T margin; Alpaca paper supports ~2.37x)
                        </span>
                    </label>
                    <input type="number" id="slot-leverage-${{i}}" value="${{slot.target_leverage || 1.0}}"
                           min="1.0" max="4.0" step="0.1"
                           style="width:120px;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                  padding:6px 10px;border-radius:6px;font-size:12px;font-family:var(--mono)">
                </div>
                <div style="border-top:1px solid var(--card-border);padding-top:10px">
                    <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-dim);cursor:pointer">
                        <input type="checkbox" id="slot-cutloss-${{i}}" ${{slot.enable_cutloss ? 'checked' : ''}}
                               style="accent-color:var(--yellow)"
                               onchange="toggleCutloss(${{i}})"> Enable Cut-Loss Protection
                    </label>
                    <div id="slot-cutloss-config-${{i}}" style="display:${{slot.enable_cutloss ? 'grid' : 'none'}};
                                grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px">
                        <div>
                            <label style="font-size:10px;color:var(--text-dim);display:block;margin-bottom:3px">Hard Stop %</label>
                            <input type="number" id="slot-hard-${{i}}" value="${{slot.cutloss_hard_stop || -8}}" step="0.5"
                                   style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                          padding:4px 8px;border-radius:4px;font-size:12px;font-family:var(--mono)">
                        </div>
                        <div>
                            <label style="font-size:10px;color:var(--text-dim);display:block;margin-bottom:3px">Trailing Stop %</label>
                            <input type="number" id="slot-trail-${{i}}" value="${{slot.cutloss_trailing_stop || -5}}" step="0.5"
                                   style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                          padding:4px 8px;border-radius:4px;font-size:12px;font-family:var(--mono)">
                        </div>
                        <div>
                            <label style="font-size:10px;color:var(--text-dim);display:block;margin-bottom:3px">Portfolio Stop %</label>
                            <input type="number" id="slot-portfolio-${{i}}" value="${{slot.cutloss_portfolio_stop || -3}}" step="0.5"
                                   style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--card-border);
                                          padding:4px 8px;border-radius:4px;font-size:12px;font-family:var(--mono)">
                        </div>
                    </div>
                </div>
            </div>`;
        }}

        $('slots-container').innerHTML = html;
    }}

    function toggleCutloss(idx) {{
        const checked = $(`slot-cutloss-${{idx}}`).checked;
        $(`slot-cutloss-config-${{idx}}`).style.display = checked ? 'grid' : 'none';
    }}

    function updateSlotStatus(idx) {{
        // Visual feedback only — actual save happens on button click
    }}

    async function testKey(idx) {{
        const key = $(`slot-key-${{idx}}`).value.trim();
        const secret = $(`slot-secret-${{idx}}`).value.trim();
        const statusEl = $(`slot-test-${{idx}}`);

        if (!key || !secret) {{
            statusEl.innerHTML = '<span class="red">Enter both key and secret</span>';
            return;
        }}

        statusEl.innerHTML = '<span class="spinner"></span> Testing...';

        const resp = await fetch('/api/config/test-key', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ key, secret }})
        }});
        const result = await resp.json();

        if (result.ok) {{
            statusEl.innerHTML = `<span class="green">Connected!</span>
                <span style="color:var(--text-dim);margin-left:6px">
                    Equity: ${{fmt$(result.equity)}} &middot; Cash: ${{fmt$(result.cash)}} &middot;
                    Account: ${{result.account_number}}
                </span>`;
        }} else {{
            statusEl.innerHTML = `<span class="red">Failed: ${{result.error}}</span>`;
        }}
    }}

    async function saveConfig() {{
        const slots = [];
        // Iterate however many slot cards the renderSlots loop produced.
        // The cards expose ids slot-model-0, slot-model-1, ... slot-model-N.
        // Stop when the next slot's model dropdown is absent from the DOM.
        for (let i = 0; ; i++) {{
            const modelEl = $(`slot-model-${{i}}`);
            if (!modelEl) break;
            slots.push({{
                slot_id: i + 1,
                model: modelEl.value,
                enabled: $(`slot-enabled-${{i}}`).checked,
                alpaca_key: $(`slot-key-${{i}}`).value.trim(),
                alpaca_secret: $(`slot-secret-${{i}}`).value.trim(),
                enable_cutloss: $(`slot-cutloss-${{i}}`).checked,
                cutloss_hard_stop: parseFloat($(`slot-hard-${{i}}`).value) || -8.0,
                cutloss_trailing_stop: parseFloat($(`slot-trail-${{i}}`).value) || -5.0,
                cutloss_portfolio_stop: parseFloat($(`slot-portfolio-${{i}}`).value) || -3.0,
                target_leverage: parseFloat($(`slot-leverage-${{i}}`)?.value) || 1.0,
            }});
        }}

        $('save-status').innerHTML = '<span class="spinner"></span> Saving...';

        const resp = await fetch('/api/config/models', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ slots }})
        }});
        const result = await resp.json();

        if (result.ok) {{
            $('save-status').innerHTML = '<span class="green">Saved! Reload page to see changes.</span>';
            setTimeout(() => window.location.reload(), 1500);
        }} else {{
            $('save-status').innerHTML = `<span class="red">Error: ${{result.error}}</span>`;
        }}
    }}

    // ── Init ────────────────────────────────────────────
    function tickClock() {{
        const el = $('header-clock');
        if (!el) return;
        el.textContent = fmtET(new Date(), {{ mode: 'time' }});
    }}
    tickClock();
    setInterval(tickClock, 1000);

    buildTabs();
    refreshAll();
    loadLogsList();

    // Auto-refresh every 15 seconds
    setInterval(refreshAll, 15000);
    // Refresh logs list every 2 minutes
    setInterval(loadLogsList, 120000);
    </script>
</body>
</html>"""
    return Response(html, content_type="text/html")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dashboard")

    _initialize_model_status()

    scheduler = BackgroundScheduler(timezone=TZ)
    scheduler.add_job(
        run_trading_pipeline,
        CronTrigger(hour=9, minute=35, day_of_week="mon-fri", timezone=TZ),
        id="trading_pipeline",
        name="Daily ML Trading Pipeline",
    )
    # Cut-loss scanner: runs every 60 seconds during market hours for V7+
    cutloss_models = [mc for mc in pipeline.get_active_models() if mc.enable_cutloss]
    if cutloss_models:
        scheduler.add_job(
            pipeline.cutloss_scan,
            'interval',
            seconds=60,
            id="cutloss_scanner",
            name="Cut-Loss Scanner (V7+)",
        )
        logger.info(f"Cut-loss scanner enabled for: {[mc.name for mc in cutloss_models]}")

    scheduler.start()

    next_run = scheduler.get_job("trading_pipeline").next_run_time
    bot_status["next_run_at"] = str(next_run)
    logger.info(f"Scheduler started. Next run: {next_run}")
    logger.info(f"Dashboard: http://0.0.0.0:{PORT}")
    logger.info(f"Active models: {[m.name for m in pipeline.get_active_models()]}")

    app.run(host="0.0.0.0", port=PORT, debug=False)
