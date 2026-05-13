"""Persistent trade journal — JSON Lines + CSV per model.

One record per submitted Alpaca order. Records are append-only across
rebalances and cutloss events; concurrent appends are serialized by a
per-model lock so the JSONL/CSV files never interleave or grow a partial
header. Read back via `TradeJournal.get_trades(...)` for analysis.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# Persistent data dir is the same env var the rest of the pipeline reads.
# Resolved at import time; if DATA_DIR changes mid-process we deliberately
# don't notice (Railway containers are recycled, not edited in place).
_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
_TRADE_DIR = _DATA_DIR / "trades"


@dataclass
class TradeRecord:
    """A single trade event — one per order submitted."""
    # Identity
    trade_id: str                   # unique: {model}_{timestamp}_{symbol}_{side}
    run_id: str                     # links all trades from the same rebalance
    model: str                      # "v4", "v5"
    timestamp: str                  # ISO-8601 UTC when order was submitted

    # Order details
    symbol: str
    side: str                       # "buy" | "sell"
    action: str                     # "new_position" | "exit_position" | "rebalance_up" | "rebalance_down"
    order_type: str                 # "market"
    time_in_force: str              # "day"

    # Amounts
    notional_usd: float             # dollar amount of the order
    shares: Optional[float] = None
    fill_price: Optional[float] = None

    # Model context
    predicted_return_pct: Optional[float] = None
    rank: Optional[int] = None
    target_weight_usd: Optional[float] = None

    # Position context (for exits / rebalances)
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl_usd: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    holding_period_days: Optional[int] = None
    position_value_before: Optional[float] = None

    # Alpaca response
    order_id: Optional[str] = None
    order_status: str = "pending"   # "submitted" | "filled" | "failed"
    error_message: Optional[str] = None

    # Portfolio context
    portfolio_value: Optional[float] = None
    cash_before: Optional[float] = None
    total_positions: Optional[int] = None
    rebalance_turnover_pct: Optional[float] = None


class TradeJournal:
    """Persistent per-model trade log.

    Concurrency: rebalance and cut-loss can fire on different threads.
    A per-model lock makes the JSONL/CSV appends atomic so neither file
    interleaves nor grows a half-written CSV header.
    """

    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    @classmethod
    def _lock_for(cls, model_name: str) -> threading.Lock:
        with cls._locks_guard:
            if model_name not in cls._locks:
                cls._locks[model_name] = threading.Lock()
            return cls._locks[model_name]

    def __init__(self, model_name: str):
        self.model_name = model_name
        _TRADE_DIR.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = _TRADE_DIR / f"trades_{model_name}.jsonl"
        self.csv_path = _TRADE_DIR / f"trades_{model_name}.csv"

    def log_trade(self, record: TradeRecord) -> None:
        """Append a trade record to both JSONL and CSV (lock-protected)."""
        data = asdict(record)
        with self._lock_for(self.model_name):
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")

            write_header = (
                not self.csv_path.exists()
                or self.csv_path.stat().st_size == 0
            )
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(data)

    def get_trades(self, symbol: Optional[str] = None,
                   since: Optional[str] = None) -> list[dict]:
        """Read trades back from the journal (for analysis endpoints)."""
        trades: list[dict] = []
        if not self.jsonl_path.exists():
            return trades
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if symbol and rec.get("symbol") != symbol:
                    continue
                if since and rec.get("timestamp", "") < since:
                    continue
                trades.append(rec)
        return trades
