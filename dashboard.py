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
_api_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 30  # seconds


def _cached_api_call(key, fn, ttl=CACHE_TTL):
    """Cache API responses for ttl seconds."""
    with _cache_lock:
        if key in _api_cache:
            data, ts = _api_cache[key]
            if time.time() - ts < ttl:
                return data
    try:
        result = fn()
        with _cache_lock:
            _api_cache[key] = (result, time.time())
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
        error_msg = f"{e}\n{traceback.format_exc()}"

        with status_lock:
            bot_status["state"] = "error"
            bot_status["last_run_at"] = datetime.now().isoformat()
            bot_status["last_run_status"] = "error"
            bot_status["last_run_duration"] = round(elapsed, 1)
            bot_status["last_error"] = error_msg
            bot_status["current_step"] = None
            bot_status["total_runs"] += 1

            targets = [model_filter] if model_filter else list(bot_status["models"].keys())
            for mn in targets:
                if mn in bot_status["models"]:
                    bot_status["models"][mn]["state"] = "error"
                    bot_status["models"][mn]["last_run_status"] = "error"

        logging.getLogger("dashboard").error(f"Pipeline failed: {error_msg}")


# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


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
    """Recent trades from journal, falling back to Alpaca order history."""
    limit = int(flask_request.args.get("limit", 50))
    trades = _load_recent_trades(model_name, limit)

    # If journal is empty, fetch from Alpaca order history
    if not trades:
        mc = _get_model_config(model_name)
        if mc:
            def fetch_orders():
                logger = logging.getLogger("dashboard")
                orders = pipeline.alpaca_request(
                    "GET",
                    "v2/orders?status=all&limit=100&direction=desc",
                    mc, logger=logger
                )
                result = []
                for o in orders:
                    filled_qty = float(o.get("filled_qty", 0) or 0)
                    filled_price = float(o.get("filled_avg_price", 0) or 0)
                    notional = filled_qty * filled_price if filled_price else float(o.get("notional", 0) or 0)
                    result.append({
                        "timestamp": o.get("filled_at") or o.get("submitted_at") or o.get("created_at", ""),
                        "symbol": o.get("symbol", ""),
                        "trade_action": o.get("side", ""),
                        "notional": round(notional, 2),
                        "order_status": o.get("status", ""),
                        "shares": filled_qty,
                        "filled_price": filled_price,
                        "order_type": o.get("type", ""),
                        "error_message": "",
                        "source": "alpaca",
                    })
                return result
            alpaca_trades = _cached_api_call(f"orders_{model_name}", fetch_orders, ttl=60)
            if alpaca_trades:
                trades = alpaca_trades[:limit]

    return jsonify(trades)


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

    # 4) Download universe prices from entry_date to now
    logger.info(f"[PERF] Computing benchmark for {model_name}: "
                f"{len(universe_syms)} universe symbols from {entry_date}")

    start_dt = pd.Timestamp(entry_date) - timedelta(days=3)
    end_dt = pd.Timestamp.now() + timedelta(days=1)

    universe_returns = []
    batch_size = 100
    for i in range(0, len(universe_syms), batch_size):
        batch = universe_syms[i:i + batch_size]
        try:
            data = yf.download(
                batch, start=start_dt, end=end_dt,
                progress=False, threads=True
            )
            if data.empty:
                continue

            close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
            if isinstance(close, pd.Series):
                close = close.to_frame()

            for sym in batch:
                try:
                    if sym in close.columns:
                        series = close[sym].dropna()
                    elif len(batch) == 1:
                        series = close.iloc[:, 0].dropna()
                    else:
                        continue

                    after_entry = series[series.index >= pd.Timestamp(entry_date)]
                    if len(after_entry) < 2:
                        continue

                    entry_price = float(after_entry.iloc[0])
                    current_price = float(after_entry.iloc[-1])
                    if entry_price > 0:
                        ret = (current_price / entry_price - 1) * 100
                        universe_returns.append(ret)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[PERF] Batch download error: {e}")

    # 5) Compute SPY benchmark
    spy_return = None
    try:
        spy_data = yf.download("SPY", start=start_dt, end=end_dt, progress=False)
        if not spy_data.empty:
            spy_close = spy_data["Close"]
            if hasattr(spy_close, 'columns'):
                spy_close = spy_close.iloc[:, 0]
            after_entry = spy_close[spy_close.index >= pd.Timestamp(entry_date)]
            if len(after_entry) >= 2:
                spy_return = float((after_entry.iloc[-1] / after_entry.iloc[0] - 1) * 100)
    except Exception:
        pass

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
        "alpha_vs_universe": round(model_return_pct - universe_avg, 2),
        "alpha_vs_spy": round(model_return_pct - spy_return, 2) if spy_return is not None else None,
        "positions": sorted(model_positions, key=lambda x: x["unrealized_plpc"], reverse=True),
    }

    # Cache
    with _perf_cache_lock:
        _perf_cache[model_name] = (result, time.time())

    return result


@app.route("/api/performance/<model_name>")
def api_performance(model_name):
    """Performance comparison: model vs universe benchmark."""
    result = _compute_performance(model_name)
    if "error" in result and "model_return_pct" not in result:
        return jsonify(result), 404 if "not found" in result.get("error", "") else 500
    return jsonify(result)


@app.route("/api/description/<model_name>")
def api_description(model_name):
    """Model description and architecture details."""
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

    return jsonify(result)


# ── LOG ENDPOINTS ─────────────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs_list():
    files = _get_log_files()[:20]
    return jsonify([f.name for f in files])


@app.route("/api/logs/<filename>")
def api_log_content(filename):
    log_path = LOG_DIR / filename
    if not log_path.exists() or not filename.startswith("pipeline_"):
        return jsonify({"error": "Not found"}), 404
    content = _read_log(log_path, tail=300)
    return jsonify({"filename": filename, "content": content})


# ── ACTIONS ───────────────────────────────────────────────────────────────

@app.route("/run")
def trigger_run():
    model_filter = flask_request.args.get("model", None)
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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "uptime_since": bot_status["started_at"]})


# ── MAIN HTML PAGE ────────────────────────────────────────────────────────

@app.route("/")
def index():
    active_models = pipeline.get_active_models()
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
                    <button class="sub-tab" onclick="switchSub('${{m}}','history',this)">Run History</button>
                    <button class="sub-tab" onclick="switchSub('${{m}}','about',this)">About</button>
                </div>
                <div id="sub-portfolio-${{m}}" class="sub-content active"></div>
                <div id="sub-positions-${{m}}" class="sub-content"></div>
                <div id="sub-chart-${{m}}" class="sub-content"></div>
                <div id="sub-performance-${{m}}" class="sub-content"></div>
                <div id="sub-trades-${{m}}" class="sub-content"></div>
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
        periodHTML += '</div><div class="card"><h2>Equity Curve</h2><div class="chart-container"><canvas id="chart-${{model}}"></canvas></div></div>';
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
            {{ icon: '\u2699\ufe0f', label: 'Architecture', key: 'architecture' }},
            {{ icon: '\ud83d\udcca', label: 'Features', key: 'features' }},
            {{ icon: '\ud83d\udcbc', label: 'Portfolio Strategy', key: 'portfolio' }},
            {{ icon: '\ud83d\udee1\ufe0f', label: 'Risk Management', key: 'risk' }},
            {{ icon: '\ud83c\udfaf', label: 'Training', key: 'training' }},
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
                    \u26a0\ufe0f ACTIVE CUT-LOSS PROTECTION
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
                else if (sub === 'history') await loadRunHistory(activeModel);
                else if (sub === 'about') await loadAbout(activeModel);
            }}
        }} catch(e) {{
            console.error('Refresh error:', e);
        }}

        bar.style.width = '100%';
        setTimeout(() => bar.style.width = '0%', 300);
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
