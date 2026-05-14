"""Per-model run state — JSON-on-disk, one file per model.

State persists between runs (cron-driven daily pipeline). Tracks the last
rebalance time, the run counter, and the rolling history of past run
summaries. The cutloss scanner also stores its peak-prices and
portfolio-stop trip flag here under separate keys.

Schema is intentionally untyped (`dict`) — both this module and the
cutloss scanner add keys to it freely, and a strict schema would just
ratchet up coupling for no real safety win.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid circular import at runtime
    from pipeline import ModelConfig


def load_state(mc: "ModelConfig") -> dict:
    """Load pipeline state from JSON for a specific model.

    Returns a fresh empty-history dict if the state file doesn't exist
    (first run for this slot).
    """
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    if mc.state_path.exists():
        return json.loads(mc.state_path.read_text())
    return {"last_rebalance": None, "last_run": None, "run_count": 0, "history": []}


def save_state(state: dict, mc: "ModelConfig") -> None:
    """Save pipeline state to JSON for a specific model. Atomic write would
    be nicer but a partial-write on this small file is recoverable."""
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    mc.state_path.write_text(json.dumps(state, indent=2, default=str))


def trading_days_between(start: datetime, end: datetime) -> int:
    """Count Mon–Fri days strictly between `start` and `end`.

    Holidays aren't subtracted — overshooting by a day on the rebalance
    cadence is harmless and avoids importing the holiday calendar here.
    Same-day returns 0.
    """
    if end <= start:
        return 0
    days = 0
    cur = start.date() + timedelta(days=1)
    end_date = end.date()
    while cur <= end_date:
        if cur.weekday() < 5:
            days += 1
        cur = cur + timedelta(days=1)
    return days


def should_rebalance(state: dict, horizon_days: int, force: bool = False) -> bool:
    """Return True if `horizon_days` trading days have passed since the
    last rebalance (or if `force` is set, or if it's never run).

    `horizon_days` is passed in (instead of imported from pipeline.HORIZON)
    so this module stays decoupled from the pipeline-level config.
    """
    if force:
        return True
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    return trading_days_between(last_date, datetime.now()) >= horizon_days
