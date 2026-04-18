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

import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

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


CUTLOSS_ACTIONS = {"hard_stop", "trailing_stop", "portfolio_stop"}


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
    return sorted(LOG_DIR.glob("pipeline_*.log"), reverse=True)


def _read_log(path: Path, tail: int = 200) -> str:
    try:
        lines = path.read_text().splitlines()
        if len(lines) > tail:
            return f"... ({len(lines) - tail} lines omitted)\n" + "\n".join(lines[-tail:])
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading log: {e}"


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


def _compute_performance(model_name: str) -> dict:
    """Compute model performance vs universe benchmark."""
    logger = logging.getLogger("dashboard")

    # Check cache
    with _perf_cache_lock:
        if model_name in _perf_cache:
            cached_data, cached_at = _perf_cache[model_name]
            if time.time() - cached_at < PERF_CACHE_TTL:
                return cached_data

    mc = _get_model_config(model_name)
    if not mc:
        return {"error": f"Model {model_name} not found"}

    # 1) Get model positions from Alpaca
    try:
        positions = pipeline.alpaca_request("GET", "v2/positions", mc, logger=logger)
    except Exception as e:
        return {"error": f"Failed to fetch positions: {e}"}

    if not positions:
        return {"error": "No positions", "model": model_name}

    # Parse positions
    model_positions = []
    for p in positions:
        model_positions.append({
            "symbol": p["symbol"],
            "avg_entry_price": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "market_value": float(p.get("market_value", 0)),
            "cost_basis": float(p.get("cost_basis", 0)),
            "unrealized_plpc": float(p.get("unrealized_plpc", 0)) * 100,
            "qty": float(p.get("qty", 0)),
        })

    # Calculate model return (weighted by position size)
    total_value = sum(pos["market_value"] for pos in model_positions)
    total_cost = sum(pos["cost_basis"] for pos in model_positions)
    if total_cost > 0:
        model_return_pct = (total_value / total_cost - 1) * 100
    else:
        model_return_pct = 0.0

    model_avg_return = np.mean([pos["unrealized_plpc"] for pos in model_positions])

    # 2) Determine entry date from positions
    # Use Alpaca orders to find the most recent batch entry date
    try:
        orders = pipeline.alpaca_request(
            "GET",
            "v2/orders?status=filled&limit=200&direction=desc",
            mc, logger=logger
        )
        # Find the earliest fill date among recent buy orders
        fill_dates = []
        for o in orders:
            if o.get("side") == "buy" and o.get("filled_at"):
                fill_dates.append(o["filled_at"][:10])  # YYYY-MM-DD
        if fill_dates:
            # Most orders fill on same day (rebalance); use the most common date
            from collections import Counter
            date_counts = Counter(fill_dates)
            entry_date = date_counts.most_common(1)[0][0]
        else:
            entry_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    except Exception:
        entry_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    # 3) Get universe symbols
    universe_syms = _get_universe_symbols()
    if not universe_syms:
        return {
            "model": model_name,
            "model_return_pct": round(model_return_pct, 2),
            "model_avg_stock_return": round(model_avg_return, 2),
            "positions": model_positions,
            "entry_date": entry_date,
            "error": "Could not load universe symbols",
        }

    # 4) Download SPY benchmark via Alpaca (no yfinance rate limits)
    # Paper trading keys require feed=iex for market data access
    logger.info(f"[PERF] Computing benchmark for {model_name} from {entry_date}")

    import requests as _requests
    _alpaca_data_headers = {
        "APCA-API-KEY-ID": mc.alpaca_key,
        "APCA-API-SECRET-KEY": mc.alpaca_secret,
    }

    spy_return = None
    try:
        resp = _requests.get(
            "https://data.alpaca.markets/v2/stocks/SPY/bars",
            headers=_alpaca_data_headers,
            params={
                "start": entry_date,
                "timeframe": "1Day",
                "limit": 30,
                "adjustment": "split",
                "feed": "iex",
            },
            timeout=10,
        )
        logger.info(f"[PERF] SPY API status={resp.status_code}")
        if resp.status_code == 200:
            bars = resp.json().get("bars", [])
            if bars and len(bars) >= 1:
                spy_entry = float(bars[0]["o"])  # open on entry date
                spy_current = float(bars[-1]["c"])  # latest close
                if spy_entry > 0:
                    spy_return = (spy_current / spy_entry - 1) * 100
                    logger.info(f"[PERF] SPY: entry={spy_entry}, current={spy_current}, return={spy_return:.2f}%")
        else:
            logger.warning(f"[PERF] SPY API error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[PERF] SPY benchmark failed: {e}")

    # 5) Quick universe sample via Alpaca (50 random stocks)
    universe_returns = []
    if universe_syms:
        import random
        sample_size = min(50, len(universe_syms))
        sample_syms = random.sample(universe_syms, sample_size)
        # Filter out symbols with special chars (Alpaca won't recognize them)
        sample_syms = [s for s in sample_syms if s.isalpha() and s.isascii()]
        try:
            resp = _requests.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                headers=_alpaca_data_headers,
                params={
                    "symbols": ",".join(sample_syms),
                    "start": entry_date,
                    "timeframe": "1Day",
                    "limit": 30,
                    "adjustment": "split",
                    "feed": "iex",
                },
                timeout=15,
            )
            logger.info(f"[PERF] Universe API status={resp.status_code}")
            if resp.status_code == 200:
                all_bars = resp.json().get("bars", {})
                for sym, bars in all_bars.items():
                    if len(bars) >= 1:
                        entry_price = float(bars[0]["o"])
                        current_price = float(bars[-1]["c"])
                        if entry_price > 0:
                            ret = (current_price / entry_price - 1) * 100
                            universe_returns.append(ret)
                logger.info(f"[PERF] Universe: {len(universe_returns)}/{len(sample_syms)} stocks returned data")
            else:
                logger.warning(f"[PERF] Universe API error: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[PERF] Universe sample failed: {e}")

    # 6) Build result
    if universe_returns:
        universe_avg = float(np.mean(universe_returns))
        universe_median = float(np.median(universe_returns))
        universe_std = float(np.std(universe_returns))
        pct_positive = sum(1 for r in universe_returns if r > 0) / len(universe_returns) * 100
    else:
        universe_avg = universe_median = universe_std = pct_positive = 0.0

    result = {
        "model": model_name,
        "entry_date": entry_date,
        "n_positions": len(model_positions),
        "model_return_pct": round(model_return_pct, 2),
        "model_avg_stock_return": round(model_avg_return, 2),
        "universe_n": len(universe_returns),
        "universe_avg_return": round(universe_avg, 2),
        "universe_median_return": round(universe_median, 2),
        "universe_std": round(universe_std, 2),
        "universe_pct_positive": round(pct_positive, 1),
        "spy_return": round(spy_return, 2) if spy_return is not None else None,
        "alpha_vs_universe": round(model_return_pct - universe_avg, 2) if universe_returns else None,
        "alpha_vs_spy": round(model_return_pct - spy_return, 2) if spy_return is not None else None,
        "positions": sorted(model_positions, key=lambda x: x["unrealized_plpc"], reverse=True),
    }

    # Cache (cap at 50 entries to prevent unbounded growth)
    with _perf_cache_lock:
        _perf_cache[model_name] = (result, time.time())
        if len(_perf_cache) > 50:
            oldest = min(_perf_cache, key=lambda m: _perf_cache[m][1])
            del _perf_cache[oldest]

    return result


@app.route("/api/performance/<model_name>")
def api_performance(model_name):
    """Performance comparison: model vs universe benchmark."""
    try:
        result = _compute_performance(model_name)
        if "error" in result and "model_return_pct" not in result:
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
        if len(new_slots) > 3:
            return jsonify({"error": "Maximum 3 slots allowed"}), 400

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
    return jsonify([f.name for f in files])


@app.route("/api/logs/<filename>")
def api_log_content(filename):
    log_path = (LOG_DIR / filename).resolve()
    if not str(log_path).startswith(str(LOG_DIR.resolve())):
        return jsonify({"error": "Not found"}), 404
    if not log_path.exists() or not filename.startswith("pipeline_"):
        return jsonify({"error": "Not found"}), 404
    content = _read_log(log_path, tail=300)
    return jsonify({"filename": filename, "content": content})


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

    # State (always available — from state file on disk)
    state = _load_model_state(model_name)
    result["run_count"] = state.get("run_count", 0)
    result["last_rebalance"] = state.get("last_rebalance")
    result["last_run"] = state.get("last_run")

    # Full run history
    history = state.get("history", [])
    result["history_count"] = len(history)
    if history:
        latest = history[-1]
        result["latest_picks"] = {
            "date": latest.get("date"),
            "symbols": latest.get("target_symbols", []),
            "predictions": latest.get("predictions", {}),
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
                   padding: 14px; font-family: var(--mono); font-size: 11px; line-height: 1.6;
                   max-height: 400px; overflow-y: auto; white-space: pre-wrap;
                   word-break: break-all; color: #8b95a8; }}

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
            <div class="meta">H{pipeline.HORIZON}_LongOnly{pipeline.TOP_N} &middot; Paper Trading &middot;
                <span id="status-badge"></span> &middot; Auto-refresh 15s</div>
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
        <select id="log-select" onchange="loadLog(this.value)">
            <option value="">Select a log file...</option>
        </select>
        <div id="log-box" class="log-box" style="margin-top:10px">No log selected</div>
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
                <div class="val" style="font-size:16px">${{status.last_run_at ? status.last_run_at.slice(0,16) : 'Never'}}</div>
                <div class="sub">${{status.last_run_duration ? status.last_run_duration + 's' : ''}}</div>
            </div>
            <div class="metric">
                <div class="label">Next Run</div>
                <div class="val" style="font-size:16px">${{status.next_run_at ? status.next_run_at.slice(0,16) : '-'}}</div>
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
            html += `<p style="font-size:12px;color:var(--text-dim);margin-bottom:8px">${{portfolio.date ? portfolio.date.slice(0,16) : ''}}</p>`;
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

        const labels = data.timestamps.map(ts => {{
            const d = new Date(ts * 1000);
            return d.toLocaleDateString('en-US', {{month:'short', day:'numeric'}});
        }});

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
                const ts = t.timestamp || '';
                const displayTs = ts.length > 16 ? ts.slice(0,10) + ' ' + ts.slice(11,16) : ts.slice(0,16);
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
        const events = await api(`/api/cutloss/${{model}}`);
        let html = '<div class="card"><h2>&#9888; Cut-Loss Events</h2>';

        if (events && events.length > 0) {{
            html += `<p style="font-size:12px;color:var(--text-dim);margin-bottom:10px">
                Showing ${{events.length}} cutloss event(s). Triggers: hard stop (-8% from entry),
                trailing stop (-5% from peak), portfolio stop (-3% daily drawdown).
            </p>`;
            html += `<table>
                <tr><th>Time</th><th>Symbol</th><th>Trigger</th><th>Detail</th>
                    <th>Shares</th><th>Status</th><th>Source</th></tr>`;
            events.forEach(e => {{
                const ts = e.timestamp || '';
                const displayTs = ts.length > 16 ? ts.slice(0,10) + ' ' + ts.slice(11,16) : ts.slice(0,16);
                const triggerLabel = (e.trigger || '').replace('_', ' ').toUpperCase();
                const triggerCls = e.trigger === 'portfolio_stop' ? 'red' :
                                   e.trigger === 'hard_stop' ? 'red' : 'yellow';
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
                const date = (entry.date || '?').slice(0, 16);
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

    async function loadPerformance(model) {{
        const container = $(`sub-performance-${{model}}`);
        container.innerHTML = '<div class="card"><h2>Performance vs Universe</h2><p style="color:var(--text-dim)"><span class="spinner"></span> Computing benchmark (may take 30-60s on first load)...</p></div>';

        const data = await api(`/api/performance/${{model}}`);
        let html = '<div class="card"><h2>Performance vs Universe Benchmark</h2>';

        if (!data || data.error) {{
            html += `<p style="color:var(--red)">${{data ? data.error : 'Failed to fetch performance data'}}</p>`;
            html += '</div>';
            container.innerHTML = html;
            return;
        }}

        // Summary metrics
        const alphaColor = data.alpha_vs_universe >= 0 ? 'green' : 'red';
        const modelColor = data.model_return_pct >= 0 ? 'green' : 'red';
        const univColor = data.universe_avg_return >= 0 ? 'green' : 'red';

        html += `<p style="font-size:12px;color:var(--text-dim);margin-bottom:14px">
            Entry date: <strong>${{data.entry_date}}</strong> &middot;
            ${{data.n_positions}} positions &middot;
            Universe: ${{data.universe_n}} stocks
        </p>`;

        html += '<div class="summary-row" style="margin-bottom:18px">';
        html += `<div class="metric">
            <div class="label">Model Return</div>
            <div class="val ${{modelColor}}">${{fmtPct(data.model_return_pct)}}</div>
            <div class="sub">Weighted portfolio return</div>
        </div>`;
        html += `<div class="metric">
            <div class="label">Universe Avg</div>
            <div class="val ${{univColor}}">${{fmtPct(data.universe_avg_return)}}</div>
            <div class="sub">${{data.universe_n}} stocks, ${{data.universe_pct_positive?.toFixed(0) || '?'}}% positive</div>
        </div>`;
        html += `<div class="metric">
            <div class="label">Alpha vs Universe</div>
            <div class="val ${{alphaColor}}">${{fmtPct(data.alpha_vs_universe)}}</div>
            <div class="sub">Model picks vs random stock</div>
        </div>`;
        if (data.spy_return != null) {{
            const spyColor = data.spy_return >= 0 ? 'green' : 'red';
            const alphaSpyColor = data.alpha_vs_spy >= 0 ? 'green' : 'red';
            html += `<div class="metric">
                <div class="label">SPY Return</div>
                <div class="val ${{spyColor}}">${{fmtPct(data.spy_return)}}</div>
                <div class="sub">Alpha vs SPY: <span class="${{alphaSpyColor}}">${{fmtPct(data.alpha_vs_spy)}}</span></div>
            </div>`;
        }}
        html += '</div>';

        // Comparison bar chart
        html += '<div style="margin-bottom:18px">';
        html += '<h3 style="color:var(--text-dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Return Comparison</h3>';

        const bars = [
            {{ label: model.toUpperCase() + ' Portfolio', value: data.model_return_pct, color: '#3b82f6' }},
            {{ label: 'Universe Average', value: data.universe_avg_return, color: '#6b7280' }},
            {{ label: 'Universe Median', value: data.universe_median_return, color: '#4b5563' }},
        ];
        if (data.spy_return != null) {{
            bars.push({{ label: 'SPY', value: data.spy_return, color: '#f59e0b' }});
        }}

        const maxAbs = Math.max(...bars.map(b => Math.abs(b.value)), 0.01);

        bars.forEach(bar => {{
            const pct = Math.abs(bar.value) / maxAbs * 100;
            const isPos = bar.value >= 0;
            html += `<div style="display:flex;align-items:center;margin-bottom:6px">
                <div style="width:120px;font-size:12px;color:var(--text-dim);flex-shrink:0">${{bar.label}}</div>
                <div style="flex:1;height:22px;position:relative;display:flex;align-items:center">
                    <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--card-border)"></div>
                    <div style="position:absolute;${{isPos ? 'left:50%' : 'right:50%'}};height:18px;width:${{pct/2}}%;
                               background:${{bar.color}};border-radius:${{isPos ? '0 4px 4px 0' : '4px 0 0 4px'}};
                               min-width:2px"></div>
                </div>
                <div style="width:70px;text-align:right;font-family:var(--mono);font-size:12px;flex-shrink:0"
                     class="${{bar.value >= 0 ? 'green' : 'red'}}">${{fmtPct(bar.value)}}</div>
            </div>`;
        }});
        html += '</div>';

        // Universe stats
        html += `<div style="margin-bottom:18px">
            <h3 style="color:var(--text-dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Universe Statistics</h3>
            <table>
                <tr><td>Average return</td><td class="text-right mono ${{cls(data.universe_avg_return)}}">${{fmtPct(data.universe_avg_return)}}</td></tr>
                <tr><td>Median return</td><td class="text-right mono ${{cls(data.universe_median_return)}}">${{fmtPct(data.universe_median_return)}}</td></tr>
                <tr><td>Std deviation</td><td class="text-right mono">${{data.universe_std?.toFixed(2) || '-'}}%</td></tr>
                <tr><td>% positive</td><td class="text-right mono">${{data.universe_pct_positive?.toFixed(0) || '-'}}%</td></tr>
                <tr><td>Stocks analyzed</td><td class="text-right mono">${{data.universe_n}}</td></tr>
            </table>
        </div>`;

        // Position returns table
        if (data.positions && data.positions.length > 0) {{
            html += '<h3 style="color:var(--text-dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Position Returns (sorted by P&L %)</h3>';
            html += `<table>
                <tr><th>#</th><th>Symbol</th><th class="text-right">Entry</th>
                    <th class="text-right">Current</th><th class="text-right">Value</th>
                    <th class="text-right">Return</th></tr>`;
            data.positions.forEach((p, i) => {{
                html += `<tr>
                    <td>${{i+1}}</td>
                    <td><strong>${{p.symbol}}</strong></td>
                    <td class="text-right mono">${{fmt$(p.avg_entry_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.current_price)}}</td>
                    <td class="text-right mono">${{fmt$(p.market_value)}}</td>
                    <td class="text-right mono ${{cls(p.unrealized_plpc)}}">${{fmtPct(p.unrealized_plpc)}}</td>
                </tr>`;
            }});
            html += '</table>';
        }}

        html += '</div>';
        container.innerHTML = html;
    }}

    async function loadLogsList() {{
        const logs = await api('/api/logs');
        if (logs && logs.length > 0) {{
            const sel = $('log-select');
            sel.innerHTML = '<option value="">Select a log file...</option>';
            logs.forEach(name => {{
                sel.innerHTML += `<option value="${{name}}">${{name}}</option>`;
            }});
        }}
    }}

    async function loadLog(filename) {{
        if (!filename) return;
        const data = await api(`/api/logs/${{filename}}`);
        if (data) $('log-box').textContent = data.content;
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
        const available = config.available_models || ['v4','v5','v6','v7','v8'];
        const slots = config.slots || [];
        let html = '';

        for (let i = 0; i < 3; i++) {{
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
        for (let i = 0; i < 3; i++) {{
            slots.push({{
                slot_id: i + 1,
                model: $(`slot-model-${{i}}`).value,
                enabled: $(`slot-enabled-${{i}}`).checked,
                alpaca_key: $(`slot-key-${{i}}`).value.trim(),
                alpaca_secret: $(`slot-secret-${{i}}`).value.trim(),
                enable_cutloss: $(`slot-cutloss-${{i}}`).checked,
                cutloss_hard_stop: parseFloat($(`slot-hard-${{i}}`).value) || -8.0,
                cutloss_trailing_stop: parseFloat($(`slot-trail-${{i}}`).value) || -5.0,
                cutloss_portfolio_stop: parseFloat($(`slot-portfolio-${{i}}`).value) || -3.0,
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
