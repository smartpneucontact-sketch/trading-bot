"""Order placement, rebalancing, and post-submission polling.

Three entry points:

  `fetch_inactive_assets(symbols, mc, logger, top_n)` — Alpaca asset
  status check on the top N candidates so we don't try to buy delisted
  or non-tradeable names.

  `poll_order_status(order_id, mc, logger)` — wait up to ~8s for a
  market order to reach a terminal state (filled/rejected/etc) so the
  trade journal records real fill prices/quantities.

  `rebalance_portfolio(...)` — the daily rebalance flow:
  fetches account + positions, computes per-symbol deltas, places
  sell/buy notional orders for each non-zero delta, polls for fills,
  appends to the trade journal. Driven by `target_symbols` and
  `target_weights` (conviction sizing) — equal-weight is just the
  fallback if `target_weights` is None.

All Alpaca I/O goes through `core.alpaca.alpaca_request`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.alpaca import (
    _make_alpaca_headers, _to_alpaca_symbol, alpaca_request,
    get_account, get_positions,
)
from core.journal import TradeRecord

if TYPE_CHECKING:
    from core.config import ModelConfig
    from core.journal import TradeJournal
    from core.run_report import RunReport


def fetch_inactive_assets(symbols: list[str], mc: "ModelConfig",
                          logger, top_n: int = 20) -> set[str]:
    """Check which of the top `2*top_n` candidate symbols are inactive on
    Alpaca. Returns the set to drop before order placement.

    Only the top candidates are checked to keep API calls bounded — a
    delisted name 50 ranks down doesn't matter, since we won't buy it.
    """
    import requests
    inactive: set[str] = set()
    headers = _make_alpaca_headers(mc)
    check_count = min(len(symbols), top_n * 2)
    checked = 0
    for sym in symbols[:check_count]:
        try:
            url = f"{mc.alpaca_base_url}/v2/assets/{_to_alpaca_symbol(sym)}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                asset = resp.json()
                if asset.get("status") != "active" or not asset.get("tradable", True):
                    inactive.add(sym)
                    logger.info(
                        f"    Filtered {sym}: status={asset.get('status')}, "
                        f"tradable={asset.get('tradable')}"
                    )
            elif resp.status_code == 404:
                inactive.add(sym)
                logger.info(f"    Filtered {sym}: not found on Alpaca")
            # else: API error — keep the symbol, it will fail at order time
            # with proper handling.
            checked += 1
        except Exception as e:
            logger.warning(f"    Asset check failed for {sym}: {e}")
            # On timeout/error, keep the symbol rather than blocking
            checked += 1
    if inactive:
        logger.info(
            f"  Inactive asset filter: removed {len(inactive)} of {checked} "
            f"checked ({', '.join(sorted(inactive))})"
        )
    else:
        logger.info(
            f"  Inactive asset filter: all {checked} checked symbols are active"
        )
    return inactive


def poll_order_status(order_id: str, mc: "ModelConfig", logger,
                      max_wait: float = 8.0, interval: float = 0.5) -> dict:
    """Poll Alpaca for final order status (filled/rejected/etc).

    Most market orders fill in <1s but secondary-venue routing during
    volatile minutes can take several seconds. We wait up to 8s before
    giving up so the trade journal records real fill prices/quantities.
    Polling backs off after the first 4 attempts to limit API calls.
    """
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
        time.sleep(sleep_for)
        elapsed += sleep_for
    try:
        return alpaca_request("GET", f"v2/orders/{order_id}", mc)
    except Exception:
        return {"status": "unknown", "id": order_id}


def rebalance_portfolio(
    target_symbols: list[str],
    rankings: list[tuple[str, float]],
    mc: "ModelConfig",
    journal: "TradeJournal",
    logger,
    report: "RunReport",
    dry_run: bool = False,
    target_weights: dict[str, float] | None = None,
):
    """Rebalance the live portfolio to match `target_symbols` (+ optional
    conviction weights), placing Alpaca orders and logging each trade.

    If `target_weights` is given, use those allocations. Otherwise fall
    back to equal-weight across `target_symbols`.

    Side effects: writes to the run report's `rebalance` dict, places
    Alpaca orders (unless `dry_run`), appends to the trade journal.
    """
    report.start_step("rebalance")
    rb_data = report.data.setdefault("rebalance", {})
    rb_data["dry_run"] = dry_run

    # Build prediction lookup: symbol -> (predicted_return, rank)
    pred_lookup: dict[str, tuple[float, int]] = {}
    for rank_idx, (sym, pred) in enumerate(rankings):
        pred_lookup[sym] = (pred, rank_idx + 1)

    # Microsecond run_id avoids collisions when two clicks land in the
    # same second (e.g. cron + manual trigger).
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
    #
    # `target_leverage` (default 1.0) multiplies the total book size.
    # 2.0 means we target 2× equity in long positions via Reg-T margin —
    # Alpaca's buying_power must cover this or orders will reject. The
    # buying_power pre-flight below warns before submission.
    leverage = float(getattr(mc, "target_leverage", 1.0) or 1.0)
    if target_weights:
        sym_allocations = {
            sym: portfolio_value * w * leverage
            for sym, w in target_weights.items()
        }
        avg_weight = portfolio_value * leverage / len(target_symbols)
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "conviction"
        lev_tag = f" × {leverage}x leverage" if leverage != 1.0 else ""
        logger.info(
            f"  Sizing: CONVICTION-WEIGHTED "
            f"(exposure={sum(target_weights.values()):.0%}{lev_tag})"
        )
    else:
        avg_weight = portfolio_value * leverage / len(target_symbols)
        sym_allocations = {sym: avg_weight for sym in target_symbols}
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "equal" if leverage == 1.0 else f"equal × {leverage}x"

    rb_data["target_leverage"] = leverage

    # Pre-flight check: at higher leverage, Alpaca's buying_power must
    # cover the total target notional. If not, log warning and continue
    # (Alpaca will reject overage orders; better to surface this clearly).
    total_target_notional = sum(sym_allocations.values())
    bp = float(acct.get("buying_power", 0))
    if total_target_notional > bp:
        logger.warning(
            f"  [REBALANCE] target notional ${total_target_notional:,.2f} "
            f"exceeds buying_power ${bp:,.2f} (leverage={leverage}x, "
            f"equity=${portfolio_value:,.2f}). "
            f"Alpaca will likely reject some orders — consider reducing "
            f"target_leverage to {bp / max(portfolio_value, 1):.2f}x or lower."
        )
        report.add_warning(
            f"target notional ${total_target_notional:,.0f} > buying_power "
            f"${bp:,.0f}; orders may be rejected"
        )

    total_positions = max(len(target_set) + len(current_set), 1)
    turnover = (len(to_sell) + len(to_buy)) / total_positions

    orders: list[dict] = []
    n_rebalanced = 0
    n_held_unchanged = 0

    # -- 1. Sell positions not in target ----------------------------------
    logger.info(f"\n  SELLS - {len(to_sell)} positions to exit:")
    for sym in to_sell:
        pos = current_positions[sym]
        qty = pos["qty"]
        mv = pos["market_value"]
        pl = pos["unrealized_pl"]
        pl_pct = pos["unrealized_pl_pct"]
        entry = pos["avg_entry"]
        price = pos["current_price"]

        logger.info(
            f"    EXIT  {sym:6s}: {qty:>8.2f} shares, "
            f"entry=${entry:.2f} -> now=${price:.2f}, "
            f"val=${mv:,.2f}, P&L=${pl:+,.2f} ({pl_pct:+.1f}%)"
        )

        if not dry_run:
            orders.append({
                "action": "sell", "symbol": sym, "qty": qty,
                "notional": mv, "side": "sell",
                "trade_action": "exit_position",
                "entry_price": entry, "current_price": price,
                "position_value_before": mv,
                "unrealized_pnl_usd": pl, "unrealized_pnl_pct": pl_pct,
            })

    # -- 2. Held positions: rebalance up/down if drift > 10% --------------
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

            logger.info(
                f"    REBAL {sym:6s}: ${current_value:,.0f} -> ${sym_target:,.0f} "
                f"(drift {drift_pct:.0f}%, {direction} ${abs(diff):,.0f}{pred_str})"
            )
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
            logger.info(
                f"    HOLD  {sym:6s}: ${current_value:,.0f} "
                f"(drift {drift_pct:.0f}% < 10%, no action{pred_str})"
            )
            n_held_unchanged += 1

    # -- 3. Buy new positions ---------------------------------------------
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

    logger.info(
        f"\n  Summary: {len(to_sell)} exits, {len(to_buy)} new buys, "
        f"{n_rebalanced} rebalanced, {n_held_unchanged} held, "
        f"turnover: {turnover:.0%}"
    )

    if dry_run:
        logger.info("  MODE: DRY RUN - no orders sent, no trades logged")
        report.end_step("rebalance")
        return rb_data

    # -- Execute orders + log each trade ----------------------------------
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
                resp = alpaca_request(
                    "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger
                )
                order_status = resp.get("status", "accepted") if resp else "accepted"
                trade.order_id = resp.get("id") if resp else None
                trade.order_status = order_status
                trade.shares = order["qty"]
                if order_status in ("rejected", "canceled", "expired"):
                    logger.error(
                        f"    REJECTED EXIT {sym:6s}: status={order_status}, "
                        f"reason: {resp.get('reject_reason', 'unknown')}"
                    )
                    trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                    report.add_error(
                        f"Order {order_status}: EXIT {sym} - {resp.get('reject_reason', '')}"
                    )
                    failed += 1
                else:
                    order_ok = True
                    logger.info(
                        f"    OK  EXIT  {sym:6s}: closed {order['qty']:.2f} shares, "
                        f"P&L=${order.get('unrealized_pnl_usd', 0):+,.2f} "
                        f"({order.get('unrealized_pnl_pct', 0):+.1f}%)"
                    )

            elif order["action"] in ("buy_notional", "sell_notional"):
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
                    report.add_error(
                        f"Order {order_status}: {order['trade_action']} {sym} - "
                        f"{resp.get('reject_reason', '')}"
                    )
                    failed += 1
                else:
                    order_ok = True
                    filled_qty = resp.get("filled_qty")
                    filled_str = (
                        f", filled_qty={filled_qty}"
                        if filled_qty and filled_qty != "0" else ""
                    )
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
        if trade.order_id and trade.order_status not in (
            "failed", "rejected", "canceled", "expired"
        ):
            try:
                final = poll_order_status(trade.order_id, mc, logger)
                trade.order_status = final.get("status", trade.order_status)
                filled_qty = final.get("filled_qty")
                filled_price = final.get("filled_avg_price")
                if filled_qty and filled_qty != "0":
                    trade.shares = float(filled_qty)
                if filled_price and filled_price != "0":
                    trade.fill_price = float(filled_price)
            except Exception:
                pass

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
    logger.info(
        f"  Notional: ${total_notional:,.2f} total "
        f"(${buy_notional:,.2f} buys, ${sell_notional:,.2f} sells)"
    )

    report.end_step("rebalance")
    return rb_data
