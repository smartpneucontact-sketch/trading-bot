#!/usr/bin/env python3
"""Live dashboard for the ML trading bot.

Runs 24/7 on Railway. Serves a web UI showing:
- Bot status (last run, next run, portfolio)
- Full run logs
- Run history with P&L tracking
- Manual trigger button

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
STATE_PATH = DATA_DIR / "state" / "pipeline_state.json"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
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
}
status_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER (called by scheduler)
# ═══════════════════════════════════════════════════════════════════════════

def run_trading_pipeline(force=False):
    """Execute the pipeline and update status."""
    global bot_status

    with status_lock:
        if bot_status["state"] == "running":
            logging.getLogger("dashboard").warning("Pipeline already running, skipping")
            return
        bot_status["state"] = "running"
        bot_status["current_step"] = "starting"

    start_time = time.time()

    try:
        pipeline.run_pipeline(dry_run=False, force=force)
        elapsed = time.time() - start_time

        with status_lock:
            bot_status["state"] = "idle"
            bot_status["last_run_at"] = datetime.now().isoformat()
            bot_status["last_run_status"] = "success"
            bot_status["last_run_duration"] = round(elapsed, 1)
            bot_status["last_error"] = None
            bot_status["current_step"] = None
            bot_status["total_runs"] += 1

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

        logging.getLogger("dashboard").error(f"Pipeline failed: {error_msg}")


# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


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


@app.route("/")
def index():
    """Main dashboard page."""
    state = _load_state()
    log_files = _get_log_files()

    # Build portfolio table
    history = state.get("history", [])
    latest = history[-1] if history else None

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

    # Build history table
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

    # Build log selector
    log_options = ""
    for lf in log_files[:20]:
        log_options += f'<option value="{lf.name}">{lf.name}</option>'

    # Status color
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
    </style>
</head>
<body>
    <h1>ML Trading Bot</h1>
    <p class="subtitle">H{pipeline.HORIZON}_LongOnly{pipeline.TOP_N} &middot; Paper Trading &middot; Auto-refreshes every 30s</p>

    <div class="grid">
        <div class="card">
            <h2>Status</h2>
            <div class="kv">
                <span class="label">State</span>
                <span class="status-badge" style="background:{status_color}">{s['state'].upper()}</span>
            </div>
            <div class="kv">
                <span class="label">Last run</span>
                <span class="value">{s['last_run_at'][:16] if s['last_run_at'] else 'never'}</span>
            </div>
            <div class="kv">
                <span class="label">Last status</span>
                <span class="value {'green' if s['last_run_status']=='success' else 'red' if s['last_run_status']=='error' else ''}">{s['last_run_status'] or '-'}</span>
            </div>
            <div class="kv">
                <span class="label">Duration</span>
                <span class="value">{f"{s['last_run_duration']}s" if s['last_run_duration'] else '-'}</span>
            </div>
            <div class="kv">
                <span class="label">Total runs</span>
                <span class="value">{s['total_runs']}</span>
            </div>
            <div class="kv">
                <span class="label">Pipeline runs</span>
                <span class="value">{state.get('run_count', 0)}</span>
            </div>
            <div class="kv">
                <span class="label">Bot uptime since</span>
                <span class="value">{s['started_at'][:16]}</span>
            </div>
            <div class="actions">
                <a class="btn" href="/run?force=1">Run Now (Force)</a>
                <a class="btn secondary" href="/run?dry=1">Dry Run</a>
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
    state = _load_state()
    s["pipeline_state"] = state
    return jsonify(s)


@app.route("/api/portfolio")
def api_portfolio():
    """JSON API for current portfolio."""
    state = _load_state()
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
    """JSON API for run history."""
    state = _load_state()
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
    """Manually trigger a pipeline run."""
    force = "force" in (os.environ.get("QUERY_STRING", "") + str(dict(
        __import__("flask").request.args)))
    dry = "dry" in str(dict(__import__("flask").request.args))

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
                pipeline.run_pipeline(dry_run=True, force=True)
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
            target=run_trading_pipeline, kwargs={"force": True}, daemon=True
        ).start()

    return Response(
        "<html><body style='background:#0f1117;color:#fff;padding:40px;font-family:sans-serif'>"
        f"<h2>{'Dry run' if dry else 'Pipeline'} triggered!</h2>"
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

    # Start Flask
    app.run(host="0.0.0.0", port=PORT, debug=False)
