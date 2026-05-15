"""US equity market hours and holiday calendar.

Half-days (early closes — typically the day before Thanksgiving and
Christmas Eve) are deliberately treated as full open: orders submitted
within 09:30–16:00 ET will fill before the early close at 13:00, and the
later cron ticks no-op naturally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo


US_MARKET_HOLIDAYS: frozenset[str] = frozenset({
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17",
    "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
})


_NY = ZoneInfo("America/New_York")


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if US equities are open at `now` (defaults to now in ET).

    Closed on weekends, US market holidays, and outside 09:30–16:00 ET.
    """
    if now is None:
        now = datetime.now(_NY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NY)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    if now.strftime("%Y-%m-%d") in US_MARKET_HOLIDAYS:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t
