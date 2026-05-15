"""Intraday cut-loss scanner — runs every minute during market hours.

Three stop layers, applied in order per slot every 60s:

  1. Portfolio-level **soft tiered** stop (replaces the legacy hard
     "liquidate all at -3% DD" rule):
       Tier 1 (DD ≤ pstop):       scale to 60% gross exposure
       Tier 2 (DD ≤ pstop·5/3):   scale to 30%
       Tier 3 (DD ≤ pstop·7/3):   liquidate to 0% + trip flag

  2. Per-position **hard stop**: sell if down `cutloss_hard_stop`%
     from average entry price.

  3. Per-position **trailing stop**: sell if down `cutloss_trailing_stop`%
     from the position's peak price since entry.

After hard/trailing stops fire, `_redistribute_after_cutloss` replaces
sold positions with the model's next-best picks. The "topup" pattern
(adding freed cash pro-rata to surviving positions, which caused the
2026-05-01 / 2026-05-07 redistribute death spirals) is deliberately
removed — leftover cash now sits as cash until the next scheduled
rebalance.

State (peak prices, daily anchor, trip flag) is persisted per-slot in
`mc.state_path` so it survives Railway redeploys.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.alpaca import _to_alpaca_symbol, alpaca_request
from core.config import get_active_models
from core.journal import TradeJournal, TradeRecord
from core.logging_setup import get_cutloss_logger
from core.market import is_market_open
from core.orders import poll_order_status
from core.state import load_state, save_state

if TYPE_CHECKING:
    from core.config import ModelConfig


# A single lock serialises read-modify-write windows inside cutloss_scan
# so back-to-back scheduler ticks don't race on the same state file.
_cutloss_state_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
# Top-level scan entry point — called every 60s by the dashboard scheduler.
# ═══════════════════════════════════════════════════════════════════════════

def cutloss_scan() -> None:
    """Scan all cutloss-enabled models and execute stops if triggered.

    No-ops outside market hours. Each model's exceptions are caught
    individually so one slot's API failure doesn't disable scans for the
    others.
    """
    if not is_market_open():
        return

    logger = get_cutloss_logger()
    models = get_active_models()
    cutloss_models = [mc for mc in models if mc.enable_cutloss]
    if not cutloss_models:
        return

    for mc in cutloss_models:
        try:
            _cutloss_scan_model(mc, logger)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: scan error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scan: portfolio-tier scaler + per-position stops + redistribute
# ═══════════════════════════════════════════════════════════════════════════

def _cutloss_scan_model(mc: "ModelConfig", logger, top_n: int = 20) -> None:
    """Run cut-loss checks for a single model.

    `top_n` is used only in the redistribute summary log and is passed
    through to `_redistribute_after_cutloss`.
    """
    today_iso = datetime.now().strftime("%Y-%m-%d")

    with _cutloss_state_lock:
        state = load_state(mc)

        # Trip-flag short-circuit: portfolio_stop already fired today.
        if state.get("portfolio_stop_tripped_date") == today_iso:
            return

        try:
            positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
        except Exception as e:
            logger.warning(f"[CUTLOSS] {mc.name}: failed to fetch positions: {e}")
            return

        if not positions:
            return

        n_positions = len(positions)

        try:
            acct = alpaca_request("GET", "v2/account", mc, logger=logger)
            current_equity = float(acct.get("equity", 0))
            last_equity = float(acct.get("last_equity", 0))
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: failed to fetch account: {e}")
            return

        # Initialise / refresh the daily anchor on date change. last_equity
        # is yesterday's close from Alpaca and is stable all day, so any
        # intraday restart still measures drawdown from the same baseline.
        state_dirty = False
        if state.get("daily_portfolio_start_date") != today_iso:
            anchor = last_equity if last_equity > 0 else current_equity
            if anchor > 0:
                state["daily_portfolio_start"] = anchor
                state["daily_portfolio_start_date"] = today_iso
                state_dirty = True
        start_equity = state.get("daily_portfolio_start", last_equity)

        # ── Portfolio-level soft tiered scaler ───────────────────────
        if start_equity > 0 and current_equity > 0:
            daily_drawdown_pct = (current_equity / start_equity - 1) * 100
            pstop = float(mc.cutloss_portfolio_stop)  # negative, e.g. -3.0

            if daily_drawdown_pct <= pstop * (7.0 / 3.0):
                # Tier 3 — hard liquidation, preserves original behavior.
                logger.warning(
                    f"[CUTLOSS] {mc.name}: PORTFOLIO STOP TIER 3! "
                    f"Daily drawdown: {daily_drawdown_pct:.2f}% <= "
                    f"{pstop * 7.0 / 3.0:.2f}%. "
                    f"Liquidating ALL {n_positions} positions "
                    f"(equity=${current_equity:,.2f})."
                )
                state["portfolio_stop_tripped_date"] = today_iso
                state["peak_prices"] = {}
                state["last_rebalance"] = None
                save_state(state, mc)
                _liquidate_all(mc, positions, "portfolio_stop", logger)
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SUMMARY — liquidated all "
                    f"{n_positions} positions, 0/{n_positions} remaining, "
                    f"equity=${current_equity:,.2f}"
                )
                return

            elif daily_drawdown_pct <= pstop * (5.0 / 3.0):
                # Tier 2 — scale to 30% gross exposure. No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, current_equity,
                    target_exposure=0.30,
                    tier="Tier2", dd_pct=daily_drawdown_pct, logger=logger,
                )
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 2 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop * 5.0 / 3.0:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 30% exposure. "
                    f"Trip flag NOT set; trailing stops still active."
                )

            elif daily_drawdown_pct <= pstop:
                # Tier 1 — scale to 60% gross exposure. No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, current_equity,
                    target_exposure=0.60,
                    tier="Tier1", dd_pct=daily_drawdown_pct, logger=logger,
                )
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 1 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 60% exposure. "
                    f"Trip flag NOT set."
                )

        # ── Per-position hard + trailing stops ───────────────────────
        peak_prices = state.setdefault("peak_prices", {})
        held_symbols = {p["symbol"] for p in positions}
        # Prune peaks for positions no longer held.
        for sym in list(peak_prices.keys()):
            if sym not in held_symbols:
                del peak_prices[sym]
                state_dirty = True

        symbols_to_sell: list[tuple[str, float, str, float]] = []

        for p in positions:
            sym = p["symbol"]
            qty = float(p.get("qty", 0))
            avg_entry = float(p.get("avg_entry_price", 0))
            current_price = float(p.get("current_price", 0))

            if avg_entry <= 0 or current_price <= 0 or qty <= 0:
                continue

            prev_peak = peak_prices.get(sym, avg_entry)
            current_peak = max(prev_peak, current_price)
            if peak_prices.get(sym) != current_peak:
                peak_prices[sym] = current_peak
                state_dirty = True

            # Hard stop: down X% from entry
            pct_from_entry = (current_price / avg_entry - 1) * 100
            if pct_from_entry <= mc.cutloss_hard_stop:
                logger.warning(
                    f"[CUTLOSS] {mc.name}: HARD STOP on {sym}! "
                    f"{pct_from_entry:.2f}% from entry (threshold: {mc.cutloss_hard_stop}%)"
                )
                symbols_to_sell.append((sym, qty, "hard_stop", pct_from_entry))
                continue

            # Trailing stop: down X% from peak
            pct_from_peak = (current_price / current_peak - 1) * 100
            if pct_from_peak <= mc.cutloss_trailing_stop:
                logger.warning(
                    f"[CUTLOSS] {mc.name}: TRAILING STOP on {sym}! "
                    f"{pct_from_peak:.2f}% from peak ${current_peak:.2f} "
                    f"(threshold: {mc.cutloss_trailing_stop}%)"
                )
                symbols_to_sell.append((sym, qty, "trailing_stop", pct_from_peak))
                continue

        # Execute the sells
        sold_symbols: list[str] = []
        for sym, qty, reason, pct in symbols_to_sell:
            try:
                _execute_cutloss_sell(mc, sym, qty, reason, pct, logger)
                sold_symbols.append(sym)
                if peak_prices.pop(sym, None) is not None:
                    state_dirty = True
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to sell {sym}: {e}")

        if state_dirty:
            save_state(state, mc)

    # Summary + redistribute (outside the lock — pure logging + new API calls)
    if sold_symbols:
        remaining = n_positions - len(sold_symbols)
        logger.info(
            f"[CUTLOSS] {mc.name}: SUMMARY — sold {len(sold_symbols)} "
            f"({', '.join(sold_symbols)}), {remaining}/{n_positions} positions remaining, "
            f"equity=${current_equity:,.2f}"
        )
        try:
            _redistribute_after_cutloss(mc, sold_symbols, logger, top_n=top_n)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: redistribution failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Redistribute freed cash into model's next-best picks (NOT into survivors).
# ═══════════════════════════════════════════════════════════════════════════

def _redistribute_after_cutloss(mc: "ModelConfig", sold_symbols: list[str],
                                logger, top_n: int = 20) -> None:
    """Replace sold positions with next-best stocks from the model's latest
    predictions.

    Flow:
      1. Wait briefly for sells to settle.
      2. Skip-guard: if drawdown is already within 1.5pp of portfolio_stop,
         do nothing — the next 60s scan will likely liquidate anyway.
      3. Look up the model's last predictions from state.
      4. Pick replacement stocks (top-ranked not currently held / just sold).
      5. Buy replacements with an equal share of freed cash.
      6. Any leftover cash sits as cash until the next scheduled rebalance
         (the legacy topup-into-survivors path was removed after the
         2026-05-01 / 2026-05-07 cascades).
    """
    time.sleep(2)  # let sells settle on Alpaca's side

    try:
        positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
        account = alpaca_request("GET", "v2/account", mc, logger=logger)
    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to fetch positions/account: {e}")
        return

    # Skip-guard — see docstring. Widened from 1.0pp to 1.5pp after the
    # cascades showed the brake firing too late.
    try:
        current_eq = float(account.get("equity", 0))
        last_eq = float(account.get("last_equity", 0))
    except (TypeError, ValueError):
        current_eq = last_eq = 0
    if current_eq > 0 and last_eq > 0:
        drawdown_pct = (current_eq / last_eq - 1) * 100
        skip_threshold = mc.cutloss_portfolio_stop + 1.5
        if drawdown_pct <= skip_threshold:
            logger.warning(
                f"[REDISTRIBUTE] {mc.name}: SKIPPED — daily drawdown {drawdown_pct:+.2f}% "
                f"is within 1.5pp of portfolio_stop ({mc.cutloss_portfolio_stop:+.1f}%); "
                f"funnelling cash into a sinking portfolio is counterproductive."
            )
            return

    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    available = min(cash, buying_power)
    equity = float(account.get("equity", 0))
    buffer = equity * 0.01  # keep 1% of equity in cash to avoid over-allocating
    available = max(0, available - buffer)

    if available < 50:
        logger.info(f"[REDISTRIBUTE] {mc.name}: only ${available:.2f} available, skipping")
        return

    held_symbols: set[str] = set()
    for p in positions:
        sym = p["symbol"]
        if sym not in sold_symbols:
            held_symbols.add(sym)

    n_held = len(held_symbols)
    n_to_replace = len(sold_symbols)

    state = load_state(mc)
    history = state.get("history", [])
    replacements: list[tuple[str, float]] = []

    if history:
        latest = history[-1]
        predictions = latest.get("predictions", {})
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        target_syms = set(latest.get("target_symbols", []))

        sold_set = set(sold_symbols)
        for sym, score in ranked:
            if sym in held_symbols or sym in sold_set:
                continue
            replacements.append((sym, score))
            if len(replacements) >= n_to_replace:
                break

        if len(replacements) < n_to_replace:
            for sym in target_syms:
                if sym in held_symbols or sym in sold_set:
                    continue
                if sym not in [r[0] for r in replacements]:
                    replacements.append((sym, 0))
                    if len(replacements) >= n_to_replace:
                        break

    if replacements:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: replacing {n_to_replace} sold position(s) "
            f"with {len(replacements)} new: {', '.join(r[0] for r in replacements)}"
        )
    else:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: no replacement candidates found in predictions, "
            f"distributing into {n_held} existing positions"
        )

    journal = TradeJournal(mc.name)
    n_bought = 0

    if replacements:
        n_targets = len(replacements) + n_held
        replacement_alloc = available / max(n_targets, 1)

        for sym, score in replacements:
            alloc = round(replacement_alloc, 2)
            if alloc < 10:
                continue
            n_bought += _place_redistribute_buy(
                mc, sym, alloc, "replacement", score, journal, logger
            )

        used = replacement_alloc * len(replacements)
        leftover = available - used
    else:
        leftover = available

    # NOTE: prior versions topped up surviving positions with leftover.
    # That created the redistribute death spiral. Leftover cash now sits
    # idle until the next scheduled rebalance.
    deployed = available - leftover
    if leftover > 50:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: ${leftover:.2f} held as cash until next "
            f"rebalance (topup-existing disabled to prevent procyclical averaging "
            f"down on losers)."
        )

    logger.info(
        f"[REDISTRIBUTE] {mc.name}: completed — {n_bought} buys placed, "
        f"${deployed:.2f} deployed (${leftover:.2f} held cash), "
        f"{n_held + len(replacements)}/{top_n} target positions"
    )


def _place_redistribute_buy(mc: "ModelConfig", symbol: str, alloc: float,
                            action: str, score: float,
                            journal, logger) -> int:
    """Place a single redistribution buy order. Returns 1 on success, 0 on failure."""
    order_data = {
        "symbol": _to_alpaca_symbol(symbol),
        "notional": str(alloc),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    try:
        result = alpaca_request("POST", "v2/orders", mc, data=order_data, logger=logger)
        order_id = result.get("id", "?")
        order_status = result.get("status", "submitted")
        label = f"NEW {symbol}" if action == "replacement" else f"TOP-UP {symbol}"
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: {label} ${alloc:.2f}"
            f"{f' (score={score:.4f})' if score else ''} → {order_status}"
        )

        fill_price = None
        filled_qty = 0
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
            except Exception:
                pass

        record = TradeRecord(
            trade_id=f"{mc.name}_redist_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="buy",
            action=f"redistribute_{action}",
            order_type="market",
            time_in_force="day",
            notional_usd=alloc,
            order_status=order_status,
            order_id=order_id,
            shares=filled_qty,
            fill_price=fill_price,
        )
        journal.log_trade(record)
        return 1

    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to buy {symbol}: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Per-position sell + portfolio-level liquidation + soft pro-rata scaling.
# ═══════════════════════════════════════════════════════════════════════════

def _execute_cutloss_sell(mc: "ModelConfig", symbol: str, qty: float,
                          reason: str, pct: float, logger) -> None:
    """Execute a market sell to close a position triggered by a cut-loss rule.

    Uses `DELETE /v2/positions/<symbol>` which closes the entire position
    atomically — handles fractional shares cleanly and never half-fills.
    Caller owns peak-price tracking; we deliberately do not touch state here.
    """
    logger.info(
        f"[CUTLOSS] {mc.name}: SELLING {symbol} qty={qty:.2f} "
        f"reason={reason} ({pct:.2f}%)"
    )

    alpaca_sym = _to_alpaca_symbol(symbol)
    try:
        result = alpaca_request(
            "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger,
        ) or {}
        order_id = result.get("id", "?")
        order_status = result.get("status", "accepted")

        if order_status in ("rejected", "canceled", "expired"):
            logger.error(
                f"[CUTLOSS] {mc.name}: {symbol} close order {order_status}: "
                f"{result.get('reject_reason', 'unknown')}"
            )
        else:
            logger.info(
                f"[CUTLOSS] {mc.name}: {symbol} close order placed: {order_id}, "
                f"status={order_status}"
            )

        fill_price = None
        filled_qty = qty
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
                logger.info(
                    f"[CUTLOSS] {mc.name}: {symbol} final status={order_status}, "
                    f"filled_qty={filled_qty}, fill_price={fill_price}"
                )
            except Exception:
                pass

        # Notional for the journal: Alpaca's `DELETE /v2/positions/<sym>`
        # doesn't return a notional, so we compute it ourselves from the
        # post-poll fill data. Falls back to 0 if either is unknown.
        notional = round(filled_qty * fill_price, 2) if (filled_qty and fill_price) else 0.0

        journal = TradeJournal(mc.name)
        record = TradeRecord(
            trade_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="sell",
            action=reason,
            order_type="market",
            time_in_force="day",
            notional_usd=notional,
            order_status=order_status,
            order_id=order_id,
            error_message=result.get("reject_reason") if order_status in ("rejected", "canceled", "expired") else None,
            shares=filled_qty,
            fill_price=fill_price,
        )
        journal.log_trade(record)

    except Exception as e:
        logger.error(f"[CUTLOSS] {mc.name}: order failed for {symbol}: {e}")
        raise


def _liquidate_all(mc: "ModelConfig", positions: list, reason: str, logger) -> None:
    """Liquidate every position for a model (Tier 3 portfolio stop).

    Caller is responsible for wiping `state["peak_prices"]` and persisting
    the trip flag *before* calling this — so the flag + empty peak dict
    are committed atomically even if a sell fails midway through.
    """
    logger.warning(
        f"[CUTLOSS] {mc.name}: LIQUIDATING ALL {len(positions)} positions ({reason})"
    )
    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("qty", 0))
        if qty > 0:
            try:
                _execute_cutloss_sell(mc, sym, qty, reason, 0.0, logger)
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to liquidate {sym}: {e}")


def _soft_scale_portfolio(mc: "ModelConfig", positions: list,
                          current_equity: float, target_exposure: float,
                          tier: str, dd_pct: float, logger) -> int:
    """Scale gross exposure down to `target_exposure` of equity by selling
    pro-rata across positions. Used by Tier 1 and Tier 2 of the soft
    portfolio stop.

    Each position is partial-sold using `notional` market orders so the
    portfolio's relative weights stay roughly intact (no "let the winners
    run" bias against the model's intended shape).

    Returns the count of positions partially-or-fully sold.
    """
    if not positions:
        return 0
    total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
    if total_mv <= 0:
        return 0
    target_mv = current_equity * target_exposure
    excess_mv = total_mv - target_mv
    if excess_mv <= 0:
        return 0

    fraction_to_sell = min(max(excess_mv / total_mv, 0.0), 1.0)
    journal = TradeJournal(mc.name)
    n_sold = 0
    reason = f"soft_scale_{tier.lower()}"

    for p in positions:
        sym = p["symbol"]
        mv = float(p.get("market_value", 0) or 0)
        if mv <= 0:
            continue
        sell_notional = round(mv * fraction_to_sell, 2)
        if sell_notional < 5.0:
            continue  # skip dust slices
        alpaca_sym = _to_alpaca_symbol(sym)
        try:
            resp = alpaca_request(
                "POST", "v2/orders", mc,
                data={
                    "symbol": alpaca_sym,
                    "notional": sell_notional,
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                },
                logger=logger,
            ) or {}
            status = resp.get("status", "submitted")
            logger.info(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell {sym} "
                f"${sell_notional:.2f} ({fraction_to_sell*100:.1f}% of "
                f"position) reason={reason} status={status}"
            )
            try:
                ts = datetime.now(timezone.utc)
                journal.log_trade(TradeRecord(
                    trade_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                    run_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d')}",
                    model=mc.name,
                    timestamp=ts.isoformat(),
                    symbol=sym,
                    side="sell",
                    action=reason,
                    order_type="market",
                    time_in_force="day",
                    notional_usd=sell_notional,
                    order_id=resp.get("id"),
                    order_status=status,
                    current_price=float(p.get("current_price", 0) or 0),
                ))
            except Exception:
                pass  # journal failure shouldn't block the sell
            n_sold += 1
        except Exception as e:
            logger.error(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell failed for {sym}: {e}"
            )

    logger.info(
        f"[CUTLOSS] {mc.name}: {tier} scaled {n_sold} positions pro-rata "
        f"(sold ~{fraction_to_sell*100:.1f}% of each), DD was {dd_pct:+.2f}%, "
        f"target_exposure={target_exposure*100:.0f}%."
    )
    return n_sold
