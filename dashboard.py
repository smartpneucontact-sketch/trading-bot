#!/usr/bin/env python3
"""Live dashboard for the ML trading bot with multi-model support.

Runs 24/7 on Railway. Serves a web UI showing:
- Bot status (last run, next run) — shared across all models
- Per-model status, portfolio, and run history
- Full run logs
- Manual trigger buttons (all models or per-model)

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

from flask import Flask, jsonify, Response
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import pipeline

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

PORT = int(os.environ.get("PORT", 8080))
TZ = os.environ.get("TZ", "America/New_York")
BASE_DIR = Path(__file__).parent

# Persistent data directory — mount a Railway volume at /app/data to survive redeploys
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
STATE_DIR = DATA_DIR / "state"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for live status
bot_status = {
    "state": "idle",           # idle | running | error
    "last_run_at": None,
    "last_run_status": None,   # success | error
    "last_run_duration": None,
    "last_error": None,
    "next_run_at": None,
    "current_step": None,
    "started_at": datetime.now().isoformat(),
    "total_runs": 0,
    "models": {},              # Per-model status: {model_name: {state, run_count, ...}}
}
status_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_state_path(model_name: str) -> Path:
    """Get state file path for a model."""
    return STATE_DIR / f"pipeline_state_{model_name}.json"


def _load_model_state(model_name: str) -> dict:
    """Load state for a specific model."""
    state_path = _get_state_path(model_name)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception as e:
            logging.getLogger("dashboard").warning(f"Failed to load {model_name} state: {e}")
    return {"run_count": 0, "history": []}


def _initialize_model_status():
    """Initialize per-model status entries."""
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


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER (called by scheduler)
# ═══════════════════════════════════════════════════════════════════════════

def run_trading_pipeline(force=False, model_filter=None):
    """Execute the pipeline and update status.

    Args:
        force: Force rebalance even if not due
        model_filter: Run specific model only (e.g., "v4" or "v5")
    """
    global bot_status

    with status_lock:
        if bot_status["state"] == "running":
            logging.getLogger("dashboard").warning("Pipeline already running, skipping")
            return
        bot_status["state"] = "running"
        bot_status["current_step"] = "starting"

        # Mark per-model states as running
        if model_filter:
            if model_filter in bot_status["models"]:
                bot_status["models"][model_filter]["state"] = "running"
        else:
            for model_name in bot_status["models"]:
                bot_status["models"][model_name]["state"] = "running"

    start_time = time.time()

    try:
        # Call pipeline with model_filter if specified
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

            # Update per-model status
            if model_filter:
                if model_filter in bot_status["models"]:
                    state = _load_model_state(model_filter)
                    bot_status["models"][model_filter]["state"] = "idle"
                    bot_status["models"][model_filter]["run_count"] = state.get("run_count", 0)
                    bot_status["models"][model_filter]["last_run_status"] = "success"
                    bot_status["models"][model_filter]["last_run_at"] = datetime.now().isoformat()
                    bot_status["models"][model_filter]["last_run_duration"] = round(elapsed, 1)
            else:
                for model_name in bot_status["models"]:
                    state = _load_model_state(model_name)
                    bot_status["models"][model_name]["state"] = "idle"
                    bot_status["models"][model_name]["run_count"] = state.get("run_count", 0)
                    bot_status["models"][model_name]["last_run_status"] = "success"
                    bot_status["models"][model_name]["last_run_at"] = datetime.now().isoformat()
                    bot_status["models"][model_name]["last_run_duration"] = round(elapsed, 1)

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

            # Mark per-model as errored
            if model_filter:
                if model_filter in bot_status["models"]:
                    bot_status["models"][model_filter]["state"] = "error"
                    bot_status["models"][model_filter]["last_run_status"] = "error"
            else:
                for model_name in bot_status["models"]:
                    bot_status["models"][model_name]["state"] = "error"
                    bot_status["models"][model_name]["last_run_status"] = "error"

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


def _build_model_tab(model_name: str, active: bool = False) -> str:
    """Build HTML for a model tab and content."""
    state = _load_model_state(model_name)
    history = state.get("history", [])
    latest = history[-1] if history else None

    with status_lock:
        model_status = bot_status.get("models", {}).get(model_name, {})

    # Portfolio table
    portfolio_rows = ""
    if latest:
        preds = latest.get("predictions", {})
        for i, sym in enumerate(latest.get("target_symbols", [])):
            pred = preds.get(sym, 0)
            portfolio_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td><strong>{sym}</strong></td>
                <td class="{'green' if pred > 0 else 'red'}">{pred:+.2f}%</td>
            </tr>"""

    # History table
    history_rows = ""
    for entry in reversed(history[-20:]):
        date = entry.get("date", "?")[:16]
        n_stocks = len(entry.get("target_symbols", []))
        result = entry.get("result", {})
        sells = result.get("n_sells", "?")
        buys = result.get("n_buys", "?")
        history_rows += f"""
        <tr>
            <td>{date}</td>
            <td>{n_stocks}</td>
            <td>{sells}</td>
            <td>{buys}</td>
        </tr>"""

    last_run = model_status.get("last_run_at", "never")
    if last_run and last_run != "never":
        last_run = last_run[:16]

    model_color = {
        "idle": "#2ecc71", "running": "#f39c12", "error": "#e74c3c"
    }.get(model_status.get("state", "idle"), "#95a5a6")

    display = "block" if active else "none"

    html = f"""
    <div id="tab-{model_name}" class="tab-content" style="display:{display}">
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="kv">
                    <span class="label">Model</span>
                    <span class="value">{model_name.upper()}</span>
                </div>
                <div class="kv">
                    <span class="label">State</span>
                    <span class="status-badge" style="background:{model_color}">{model_status.get("state", "unknown").upper()}</span>
                </div>
                <div class="kv">
                    <span class="label">Last run</span>
                    <span class="value">{last_run}</span>
                </div>
                <div class="kv">
                    <span class="label">Last status</span>
                    <span class="value {'green' if model_status.get("last_run_status")=='success' else 'red' if model_status.get("last_run_status")=='error' else ''}">{model_status.get('last_run_status') or '-'}</span>
                </div>
                <div class="kv">
                    <span class="label">Duration</span>
                    <span class="value">{f"{model_status.get('last_run_duration')}s" if model_status.get('last_run_duration') else '-'}</span>
                </div>
                <div class="kv">
                    <span class="label">Total runs</span>
                    <span class="value">{state.get('run_count', 0)}</span>
                </div>
                <div class="actions">
                    <a class="btn" href="/run?model={model_name}&force=1">Run Now (Force)</a>
                    <a class="btn secondary" href="/run?model={model_name}&dry=1">Dry Run</a>
                </div>
            </div>

            <div class="card">
                <h2>Current Portfolio</h2>
                <table>
                    <tr><th>#</th><th>Symbol</th><th>Pred. Return</th></tr>
                    {portfolio_rows if portfolio_rows else '<tr><td colspan="3" style="color:#666">No portfolio yet</td></tr>'}
                </table>
            </div>
        </div>

        <div class="card" style="margin-bottom:20px">
            <h2>Run History</h2>
            <table>
                <tr><th>Date</th><th>Stocks</th><th>Sells</th><th>Buys</th></tr>
                {history_rows if history_rows else '<tr><td colspan="4" style="color:#666">No runs yet</td></tr>'}
            </table>
        </div>
    </div>
    """
    return html


@app.route("/")
def index():
    """Main dashboard page."""
    log_files = _get_log_files()

    # Get active models
    active_models = pipeline.get_active_models()
    if not active_models:
        return Response(
            """<!DOCTYPE html>
            <html>
            <head>
                <title>ML Trading Bot</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                           background: #0f1117; color: #e0e0e0; padding: 20px; }
                    .card { background: #1a1d27; border-radius: 10px; padding: 20px; border: 1px solid #2a2d37; }
                    h1 { color: #fff; margin-bottom: 10px; }
                </style>
            </head>
            <body>
                <h1>ML Trading Bot</h1>
                <div class="card">
                    <p style="color:#e74c3c">No models configured. Set environment variables:</p>
                    <ul style="margin-left:20px;margin-top:10px;">
                        <li>MODEL_V4_ALPACA_KEY / MODEL_V4_ALPACA_SECRET</li>
                        <li>MODEL_V5_ALPACA_KEY / MODEL_V5_ALPACA_SECRET</li>
                    </ul>
                </div>
            </body>
            </html>""",
            content_type="text/html"
        )

    # Build log selector
    log_options = ""
    for lf in log_files[:20]:
        log_options += f'<option value="{lf.name}">{lf.name}</option>'

    # Build model tabs and content
    tabs_html = ""
    content_html = ""
    for i, mc in enumerate(active_models):
        active = (i == 0)
        tabs_html += f'<button class="tab-btn {"active" if active else ""}" onclick="openTab(event, \'tab-{mc.name}\')">{mc.name.upper()}</button>'
        content_html += _build_model_tab(mc.name, active=active)

    # Bot status (shared across all models)
    with status_lock:
        s = dict(bot_status)

    status_color = {
        "idle": "#2ecc71", "running": "#f39c12", "error": "#e74c3c"
    }.get(s["state"], "#95a5a6")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ML Trading Bot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0f1117; color: #e0e0e0; padding: 20px; }}
        h1 {{ color: #fff; margin-bottom: 5px; }}
        h2 {{ color: #8b9dc3; margin: 25px 0 10px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; }}
        .subtitle {{ color: #666; margin-bottom: 20px; font-size: 14px; }}

        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}

        .card {{ background: #1a1d27; border-radius: 10px; padding: 20px; border: 1px solid #2a2d37; }}

        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
                        font-size: 13px; font-weight: 600; color: #fff; }}

        .kv {{ display: flex; justify-content: space-between; padding: 8px 0;
              border-bottom: 1px solid #2a2d37; font-size: 14px; }}
        .kv:last-child {{ border: none; }}
        .kv .label {{ color: #8b9dc3; }}
        .kv .value {{ color: #fff; font-weight: 500; }}

        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 8px; color: #8b9dc3; border-bottom: 2px solid #2a2d37;
             font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #1f2230; }}
        tr:hover {{ background: #1f2230; }}

        .green {{ color: #2ecc71; }}
        .red {{ color: #e74c3c; }}

        .log-box {{ background: #0a0c10; border: 1px solid #2a2d37; border-radius: 8px;
                   padding: 15px; font-family: 'SF Mono', 'Fira Code', monospace;
                   font-size: 12px; line-height: 1.5; max-height: 500px; overflow-y: auto;
                   white-space: pre-wrap; word-break: break-all; color: #a0a8c0; }}

        select {{ background: #1a1d27; color: #e0e0e0; border: 1px solid #2a2d37;
                 padding: 6px 10px; border-radius: 6px; font-size: 13px; margin-bottom: 10px; }}

        .btn {{ background: #3b82f6; color: #fff; border: none; padding: 8px 20px;
               border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600;
               text-decoration: none; display: inline-block; margin-right: 8px; }}
        .btn:hover {{ background: #2563eb; }}
        .btn.danger {{ background: #e74c3c; }}
        .btn.danger:hover {{ background: #c0392b; }}
        .btn.secondary {{ background: #374151; }}
        .btn.secondary:hover {{ background: #4b5563; }}

        .actions {{ margin: 15px 0; }}

        .tabs {{ display: flex; border-bottom: 2px solid #2a2d37; margin-bottom: 20px;
                gap: 0; }}
        .tab-btn {{ background: transparent; color: #8b9dc3; border: none; padding: 10px 20px;
                   cursor: pointer; font-size: 14px; font-weight: 600; text-transform: uppercase;
                   border-bottom: 3px solid transparent; transition: all 0.2s; }}
        .tab-btn:hover {{ color: #fff; }}
        .tab-btn.active {{ color: #3b82f6; border-bottom-color: #3b82f6; }}

        .tab-content {{ display: none; }}

        .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                         gap: 15px; margin-bottom: 20px; }}
        .overview-card {{ background: #1a1d27; border: 1px solid #2a2d37; border-radius: 8px;
                         padding: 15px; }}
        .overview-card h3 {{ color: #8b9dc3; font-size: 12px; text-transform: uppercase;
                            margin-bottom: 8px; letter-spacing: 0.5px; }}
        .overview-card .value {{ color: #fff; font-size: 18px; font-weight: 600; }}
        .overview-card .subtext {{ color: #666; font-size: 12px; margin-top: 4px; }}
    </style>
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tabbtns;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tabbtns = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tabbtns.length; i++) {{
                tabbtns[i].classList.remove("active");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.classList.add("active");
        }}
    </script>
</head>
<body>
    <h1>ML Trading Bot</h1>
    <p class="subtitle">H{pipeline.HORIZON}_LongOnly{pipeline.TOP_N} &middot; Paper Trading &middot; Auto-refreshes every 30s</p>

    <div class="card" style="margin-bottom:20px">
        <h2>Overview</h2>
        <div class="overview-grid">
            <div class="overview-card">
                <h3>Bot State</h3>
                <div class="value" style="color:{status_color}">{s['state'].upper()}</div>
            </div>
            <div class="overview-card">
                <h3>Last Run</h3>
                <div class="value">{s['last_run_at'][:16] if s['last_run_at'] else 'Never'}</div>
            </div>
            <div class="overview-card">
                <h3>Duration</h3>
                <div class="value">{f"{s['last_run_duration']}s" if s['last_run_duration'] else '-'}</div>
            </div>
            <div class="overview-card">
                <h3>Total Runs</h3>
                <div class="value">{s['total_runs']}</div>
            </div>
            <div class="overview-card">
                <h3>Active Models</h3>
                <div class="value">{len(active_models)}</div>
                <div class="subtext">{', '.join([m.name.upper() for m in active_models])}</div>
            </div>
            <div class="overview-card">
                <h3>Uptime Since</h3>
                <div class="value">{s['started_at'][:10]}</div>
            </div>
        </div>
        <div class="actions">
            <a class="btn" href="/run?force=1">Run All (Force)</a>
            <a class="btn secondary" href="/run?dry=1">Dry Run All</a>
        </div>
    </div>

    <div class="card">
        <div class="tabs">
            {tabs_html}
        </div>
        {content_html}
    </div>

    <div class="card">
        <h2>Logs</h2>
        <select onchange="window.location='/logs/'+this.value">
            <option value="">Select a log file...</option>
            {log_options}
        </select>
        {'<div class="log-box">' + _read_log(log_files[0]) + '</div>' if log_files else '<p style="color:#666">No logs yet</p>'}
    </div>

    {f'<div class="card" style="margin-top:20px;border-color:#e74c3c"><h2 style="color:#e74c3c">Last Error</h2><div class="log-box" style="color:#e74c3c">{s["last_error"]}</div></div>' if s.get("last_error") else ''}
</body>
</html>"""
    return Response(html, content_type="text/html")


@app.route("/api/status")
def api_status():
    """JSON API for bot status."""
    with status_lock:
        s = dict(bot_status)

    # Include per-model states
    s["models"] = {}
    for mc in pipeline.get_active_models():
        state = _load_model_state(mc.name)
        s["models"][mc.name] = {
            "run_count": state.get("run_count", 0),
            "history": state.get("history", []),
            **bot_status.get("models", {}).get(mc.name, {})
        }

    return jsonify(s)


@app.route("/api/portfolio")
def api_portfolio():
    """JSON API for current portfolio (all models)."""
    portfolios = {}
    for mc in pipeline.get_active_models():
        state = _load_model_state(mc.name)
        history = state.get("history", [])
        if history:
            latest = history[-1]
            portfolios[mc.name] = {
                "date": latest.get("date"),
                "portfolio": latest.get("target_symbols", []),
                "predictions": latest.get("predictions", {}),
            }
        else:
            portfolios[mc.name] = {
                "date": None,
                "portfolio": [],
                "predictions": {},
            }
    return jsonify(portfolios)


@app.route("/api/portfolio/<model_name>")
def api_portfolio_model(model_name):
    """JSON API for current portfolio of a specific model."""
    state = _load_model_state(model_name)
    history = state.get("history", [])
    if not history:
        return jsonify({"portfolio": [], "date": None})
    latest = history[-1]
    return jsonify({
        "date": latest.get("date"),
        "portfolio": latest.get("target_symbols", []),
        "predictions": latest.get("predictions", {}),
    })


@app.route("/api/history")
def api_history():
    """JSON API for run history (all models)."""
    histories = {}
    for mc in pipeline.get_active_models():
        state = _load_model_state(mc.name)
        histories[mc.name] = state.get("history", [])
    return jsonify(histories)


@app.route("/api/history/<model_name>")
def api_history_model(model_name):
    """JSON API for run history of a specific model."""
    state = _load_model_state(model_name)
    return jsonify(state.get("history", []))


@app.route("/logs/<filename>")
def view_log(filename):
    """View a specific log file."""
    log_path = LOG_DIR / filename
    if not log_path.exists() or not filename.startswith("pipeline_"):
        return "Log not found", 404
    content = _read_log(log_path, tail=500)
    return Response(
        f"""<!DOCTYPE html>
<html><head><title>{filename}</title>
<style>
body {{ background: #0a0c10; color: #a0a8c0; font-family: 'SF Mono', monospace;
       font-size: 12px; padding: 20px; line-height: 1.5; }}
a {{ color: #3b82f6; }}
pre {{ white-space: pre-wrap; word-break: break-all; }}
</style></head><body>
<a href="/">&larr; Back to dashboard</a>
<h2 style="color:#fff;margin:10px 0">{filename}</h2>
<pre>{content}</pre>
</body></html>""",
        content_type="text/html",
    )


@app.route("/run")
def trigger_run():
    """Manually trigger a pipeline run.

    Query parameters:
        model=<v4|v5>  - Run specific model only (default: all)
        force=1         - Force rebalance
        dry=1           - Dry run mode
    """
    from flask import request

    model_filter = request.args.get("model", None)
    force = "force" in request.args
    dry = "dry" in request.args

    with status_lock:
        if bot_status["state"] == "running":
            return Response(
                "<html><body style='background:#0f1117;color:#fff;padding:40px;font-family:sans-serif'>"
                "<h2>Pipeline is already running</h2>"
                "<p>Wait for it to finish. <a href='/' style='color:#3b82f6'>Back</a></p>"
                "</body></html>",
                content_type="text/html",
            )

    if dry:
        # Run dry in a thread
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
            target=run_trading_pipeline, kwargs={"force": True, "model_filter": model_filter}, daemon=True
        ).start()

    model_label = f" ({model_filter.upper()})" if model_filter else " (All)"
    return Response(
        "<html><body style='background:#0f1117;color:#fff;padding:40px;font-family:sans-serif'>"
        f"<h2>{'Dry run' if dry else 'Pipeline'} triggered{model_label}!</h2>"
        "<p>Refresh the dashboard to see progress. "
        "<a href='/' style='color:#3b82f6'>Back to dashboard</a></p>"
        "<script>setTimeout(()=>window.location='/',3000)</script>"
        "</body></html>",
        content_type="text/html",
    )


@app.route("/health")
def health():
    """Health check for Railway."""
    return jsonify({"status": "ok", "uptime_since": bot_status["started_at"]})


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

    # Initialize per-model status
    _initialize_model_status()

    # Start scheduler — runs pipeline at 9:35 AM ET every weekday
    scheduler = BackgroundScheduler(timezone=TZ)
    scheduler.add_job(
        run_trading_pipeline,
        CronTrigger(hour=9, minute=35, day_of_week="mon-fri", timezone=TZ),
        id="trading_pipeline",
        name="Daily ML Trading Pipeline",
    )
    scheduler.start()

    next_run = scheduler.get_job("trading_pipeline").next_run_time
    bot_status["next_run_at"] = str(next_run)
    logger.info(f"Scheduler started. Next run: {next_run}")
    logger.info(f"Dashboard: http://0.0.0.0:{PORT}")
    logger.info(f"Active models: {[m.name for m in pipeline.get_active_models()]}")

    # Start Flask
    app.run(host="0.0.0.0", port=PORT, debug=False)
