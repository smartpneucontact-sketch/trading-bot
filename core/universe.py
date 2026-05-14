"""Tradeable universe — S&P 500 + Nasdaq 100 + Russell 1000.

Scrapes Wikipedia (which is the cheapest reliable source for the
constituent lists), caches the result on the Railway persistent volume,
and falls back to the cache if every scrape fails. The cache is the
*entire* reason the bot can still run on a day Wikipedia is being
slow/blocked.

Symbols are normalized to Alpaca's dash form (BRK.B → BRK-B). Tickers
in `EXCLUDED_SYMBOLS` are blacklisted from the final set as a safety net
(very illiquid / known data-quality issues).
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.run_report import RunReport


_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
SYMBOL_CACHE_PATH = _DATA_DIR / "universe_cache.json"


# Symbols excluded from trading. Re-evaluate annually — names that have
# accumulated 4+ years of post-IPO history can be removed. Kept here as
# a safety net for the pipeline; not authoritative.
EXCLUDED_SYMBOLS: set[str] = {
    "VFS",   # very illiquid, halted often
    "SMCI",  # data quality / restated financials
}


def _wiki_read_html(url: str, logger, max_attempts: int = 3) -> list:
    """Read HTML tables from Wikipedia with retry + exponential backoff."""
    import requests as _req
    headers = {
        "User-Agent": "MLTradingBot/1.0 (educational paper trading project)",
    }
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _req.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return pd.read_html(io.StringIO(resp.text))
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    f"  scrape attempt {attempt}/{max_attempts} failed for {url}: "
                    f"{e} — retrying in {wait}s"
                )
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _load_symbol_cache(logger) -> list[str]:
    """Load cached symbol list from the last successful scrape."""
    if SYMBOL_CACHE_PATH.exists():
        try:
            cache = json.loads(SYMBOL_CACHE_PATH.read_text())
            syms = cache.get("symbols", [])
            cached_at = cache.get("cached_at", "unknown")
            try:
                cached_dt = datetime.fromisoformat(cached_at)
                age_days = (datetime.now() - cached_dt).days
                logger.info(
                    f"  Loaded {len(syms)} cached symbols "
                    f"(from {cached_at}, {age_days}d old)"
                )
            except Exception:
                logger.info(f"  Loaded {len(syms)} cached symbols (from {cached_at})")
            return syms
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}")
    return []


def _save_symbol_cache(symbols: list[str], logger) -> None:
    """Save the symbol list to cache for future scrape-failure fallback."""
    try:
        SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYMBOL_CACHE_PATH.write_text(json.dumps({
            "symbols": symbols,
            "cached_at": datetime.now().isoformat(),
            "count": len(symbols),
        }, indent=2))
        logger.info(f"  Saved {len(symbols)} symbols to cache")
    except Exception as e:
        logger.warning(f"  Cache save failed: {e}")


def get_tradeable_symbols(logger, report: "RunReport") -> list[str]:
    """Get S&P 500 + Nasdaq 100 + Russell 1000 symbols from Wikipedia.

    Uses proper User-Agent to avoid 403 blocks. Falls back to cached
    symbol list if all scrapes fail.
    """
    report.start_step("get_universe")
    sp500: list[str] = []
    ndx_syms: list[str] = []
    russell_syms: list[str] = []

    try:
        tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", logger
        )
        sp500 = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"  S&P 500: {len(sp500)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  S&P 500 scrape failed: {e}")
        report.add_warning(f"S&P 500 scrape failed: {e}")

    try:
        ndx = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100", logger
        )
        for table in ndx:
            if "Ticker" in table.columns:
                ndx_syms = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                break
            elif "Symbol" in table.columns:
                ndx_syms = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                break
        logger.info(f"  Nasdaq 100: {len(ndx_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Nasdaq 100 scrape failed: {e}")
        report.add_warning(f"Nasdaq 100 scrape failed: {e}")

    # Russell 1000 — adds ~400-500 mid-cap stocks not in S&P 500
    try:
        r1k_tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Russell_1000_Index", logger
        )
        for table in r1k_tables:
            if "Ticker" in table.columns:
                russell_syms = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                break
            elif "Symbol" in table.columns:
                russell_syms = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                break
        logger.info(f"  Russell 1000: {len(russell_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Russell 1000 scrape failed (non-critical): {e}")
        report.add_warning(f"Russell 1000 scrape failed: {e}")

    all_syms = sorted(set(sp500 + ndx_syms + russell_syms) - EXCLUDED_SYMBOLS)

    # If scraping failed completely, fall back to cache
    if not all_syms:
        logger.warning("  All scrapes failed — falling back to cached symbol list")
        all_syms = _load_symbol_cache(logger)
    else:
        _save_symbol_cache(all_syms, logger)

    logger.info(
        f"  Total universe: {len(all_syms)} unique symbols "
        f"(excluded {len(EXCLUDED_SYMBOLS)} blacklisted)"
    )
    report.set("universe_size", len(all_syms))
    report.end_step("get_universe")
    return all_syms
