"""Per-run report — collects step timings, warnings, errors, and a final
formatted summary printed at the end of each model run."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional


class RunReport:
    """Collects all run data for a final summary.

    Step timing is wall-clock; warnings/errors are buffered then printed at
    the end so the per-step output stays clean. Arbitrary K/V data goes
    through `set()` / `get()` and is consumed by `format_summary()`.
    """

    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.steps: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.timings: dict[str, float] = {}
        self.data: dict[str, Any] = {}

    def start_step(self, name: str) -> None:
        self.steps[name] = {"status": "running", "start": time.time()}

    def end_step(self, name: str, status: str = "ok") -> None:
        if name in self.steps:
            elapsed = time.time() - self.steps[name]["start"]
            self.steps[name]["status"] = status
            self.steps[name]["elapsed"] = elapsed
            self.timings[name] = elapsed

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.data.get(key, default)

    def format_summary(self) -> str:
        total_time = (datetime.now() - self.start_time).total_seconds()
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  RUN SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  Date:         {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total time:   {total_time:.1f}s")
        lines.append("")

        # Step timings
        lines.append("  STEPS:")
        for name, info in self.steps.items():
            elapsed = info.get("elapsed", 0)
            status = info["status"]
            icon = "OK" if status == "ok" else "SKIP" if status == "skipped" else "FAIL"
            lines.append(f"    [{icon:4s}] {name:30s} {elapsed:6.1f}s")
        lines.append("")

        # Data quality
        if "universe_size" in self.data:
            lines.append("  DATA:")
            lines.append(f"    Universe requested:   {self.data.get('universe_size', '?')} symbols")
            lines.append(f"    Stocks downloaded:    {self.data.get('stocks_downloaded', '?')} symbols")
            lines.append(f"    Stocks failed:        {self.data.get('stocks_failed', '?')} symbols")
            lines.append(f"    Macro downloaded:     {self.data.get('macro_downloaded', '?')}/{self.data.get('macro_total', '?')} tickers")
            lines.append(f"    Macro missing:        {', '.join(self.data.get('macro_missing', [])) or 'none'}")
            lines.append(f"    Feature columns:      {self.data.get('n_features', '?')}")
            lines.append(f"    Latest data date:     {self.data.get('latest_data_date', '?')}")
            lines.append("")

        # Predictions
        if "n_predictions" in self.data:
            lines.append("  PREDICTIONS:")
            lines.append(f"    Stocks predicted:     {self.data.get('n_predictions', '?')}")
            lines.append(f"    Prediction failures:  {self.data.get('prediction_failures', '?')}")
            pred_range = self.data.get("prediction_range", {})
            if pred_range:
                lines.append(f"    Pred range:           {pred_range.get('min', 0):.2f}% to {pred_range.get('max', 0):.2f}%")
                lines.append(f"    Pred mean:            {pred_range.get('mean', 0):.2f}%")
                lines.append(f"    Pred std:             {pred_range.get('std', 0):.2f}%")
            lines.append("")

        # Portfolio
        if "target_portfolio" in self.data:
            lines.append("  TARGET PORTFOLIO (top 20):")
            for i, (sym, pred) in enumerate(self.data["target_portfolio"]):
                lines.append(f"    {i+1:2d}. {sym:6s}  pred={pred:+6.2f}%")
            lines.append("")

        # Rebalance
        if "rebalance" in self.data:
            rb = self.data["rebalance"]
            lines.append("  REBALANCE:")
            lines.append(f"    Mode:                 {'DRY RUN' if rb.get('dry_run') else 'LIVE'}")
            lines.append(f"    Portfolio value:       ${rb.get('portfolio_value', 0):,.2f}")
            lines.append(f"    Cash available:        ${rb.get('cash', 0):,.2f}")
            lines.append(f"    Target weight/stock:   ${rb.get('target_weight', 0):,.2f}")
            lines.append(f"    Positions before:      {rb.get('positions_before', '?')}")
            lines.append(f"    Sells (exit):          {rb.get('n_sells', 0)}")
            lines.append(f"    Buys (new):            {rb.get('n_buys', 0)}")
            lines.append(f"    Rebalanced (adjust):   {rb.get('n_rebalanced', 0)}")
            lines.append(f"    Held (no change):      {rb.get('n_held', 0)}")
            lines.append(f"    Turnover:              {rb.get('turnover', 0):.0%}")
            if not rb.get("dry_run"):
                lines.append(f"    Orders submitted:      {rb.get('executed', 0)}")
                lines.append(f"    Orders failed:         {rb.get('failed', 0)}")

            if rb.get("sells_detail"):
                lines.append(f"    Sold:   {', '.join(rb['sells_detail'])}")
            if rb.get("buys_detail"):
                lines.append(f"    Bought: {', '.join(rb['buys_detail'])}")
            lines.append("")

        # Trade journal summary
        if "trade_log_summary" in self.data:
            tls = self.data["trade_log_summary"]
            lines.append("  TRADE JOURNAL:")
            lines.append(f"    Trades logged:         {tls.get('count', 0)}")
            lines.append(f"    Total notional:        ${tls.get('total_notional', 0):,.2f}")
            lines.append(f"    Buy notional:          ${tls.get('buy_notional', 0):,.2f}")
            lines.append(f"    Sell notional:         ${tls.get('sell_notional', 0):,.2f}")
            lines.append(f"    Journal file:          {tls.get('file', '?')}")
            lines.append("")

        # Warnings / Errors
        if self.warnings:
            lines.append(f"  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:20]:
                lines.append(f"    - {w}")
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings) - 20} more")
            lines.append("")

        if self.errors:
            lines.append(f"  ERRORS ({len(self.errors)}):")
            for e in self.errors[:20]:
                lines.append(f"    - {e}")
            lines.append("")

        status = "SUCCESS" if not self.errors else "COMPLETED WITH ERRORS"
        lines.append(f"  STATUS: {status}")
        lines.append("=" * 70)
        return "\n".join(lines)
