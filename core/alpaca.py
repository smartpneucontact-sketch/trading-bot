"""Alpaca API primitives — auth headers, request wrapper, account/positions.

`alpaca_request` is the single chokepoint for every Alpaca call so:
- Per-model credentials get applied (each slot has its own key/secret)
- Logging is consistent (`API [v7]: GET v2/account`)
- HTTP errors raise with a uniform message

`_to_alpaca_symbol` handles the BRK-B → BRK.B conversion: our universe
uses yfinance's dash form, Alpaca's trading API wants the dot. Class-share
detection is heuristic (ends with `-X` where X is uppercase ASCII), which
matches every share-class ticker in the SP500+NDX100+R1K universe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import ModelConfig
    from core.run_report import RunReport


def _to_alpaca_symbol(sym: str) -> str:
    """Convert yfinance-style class-share ticker (BRK-B) to Alpaca form (BRK.B).

    The universe scrape stores symbols with dashes for yfinance
    compatibility; Alpaca's trading and data APIs expect a dot.
    """
    if (
        len(sym) >= 3
        and sym[-2] == "-"
        and sym[-1].isascii()
        and sym[-1].isalpha()
        and sym[-1].isupper()
    ):
        return sym[:-2] + "." + sym[-1]
    return sym


def _make_alpaca_headers(mc: "ModelConfig") -> dict:
    """Build Alpaca auth headers for a specific model's credentials."""
    return {
        "APCA-API-KEY-ID": mc.alpaca_key,
        "APCA-API-SECRET-KEY": mc.alpaca_secret,
        "Content-Type": "application/json",
    }


def alpaca_request(method: str, endpoint: str, mc: "ModelConfig",
                   data=None, logger=None):
    """Make an Alpaca API request with per-model credentials.

    Returns parsed JSON on 2xx, raises `Exception` on anything else.
    204 responses (common on `DELETE /v2/positions/{sym}`) come back
    as `{}` since they have no body.
    """
    import requests
    url = f"{mc.alpaca_base_url}/{endpoint}"
    headers = _make_alpaca_headers(mc)

    if logger:
        logger.info(f"    API [{mc.name}]: {method} {endpoint}")

    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=15)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data, timeout=15)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=15)
    else:
        raise ValueError(f"Unknown method: {method}")

    if resp.status_code not in (200, 204, 207):
        err_msg = f"Alpaca {method} {endpoint}: {resp.status_code} {resp.text}"
        if logger:
            logger.error(f"    {err_msg}")
        raise Exception(err_msg)

    if resp.status_code == 204:
        return {}
    return resp.json()


def get_account(mc: "ModelConfig", logger, report: "RunReport"):
    """Fetch Alpaca account info and stash portfolio_value/cash/buying_power
    into the run report's rebalance section."""
    acct = alpaca_request("GET", "v2/account", mc, logger=logger)
    pv = float(acct["portfolio_value"])
    cash = float(acct["cash"])
    bp = float(acct["buying_power"])
    logger.info(
        f"  Account [{mc.name}]: portfolio=${pv:,.2f}, "
        f"cash=${cash:,.2f}, buying_power=${bp:,.2f}"
    )
    report.data.setdefault("rebalance", {}).update({
        "portfolio_value": pv, "cash": cash, "buying_power": bp,
    })
    return acct


def get_positions(mc: "ModelConfig", logger) -> dict[str, dict]:
    """Get current positions as `{symbol: {qty, market_value, P&L, ...}}`."""
    positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
    pos_dict: dict[str, dict] = {}
    total_value = 0.0
    total_pl = 0.0

    for p in positions:
        mv = float(p["market_value"])
        pl = float(p["unrealized_pl"])
        pos_dict[p["symbol"]] = {
            "qty": float(p["qty"]),
            "market_value": mv,
            "unrealized_pl": pl,
            "unrealized_pl_pct": float(p.get("unrealized_plpc", 0)) * 100,
            "avg_entry": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "side": p["side"],
        }
        total_value += mv
        total_pl += pl

    logger.info(
        f"  Positions [{mc.name}]: {len(pos_dict)} stocks, "
        f"value=${total_value:,.2f}, unrealized P&L=${total_pl:,.2f}"
    )

    if pos_dict:
        for sym, info in sorted(pos_dict.items()):
            logger.info(
                f"    {sym:6s}: {info['qty']:>8.2f} shares @ "
                f"${info['avg_entry']:>8.2f} -> ${info['current_price']:>8.2f}, "
                f"val=${info['market_value']:>10,.2f}, "
                f"P&L=${info['unrealized_pl']:>+8,.2f} "
                f"({info['unrealized_pl_pct']:>+5.1f}%)"
            )

    return pos_dict
