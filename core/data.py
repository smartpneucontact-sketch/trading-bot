"""Daily bar + macro data download via yfinance.

Both downloads use yfinance's batch endpoint when possible (50 tickers
per request for stocks; one-by-one for macro because the macro tickers
include `^VIX` etc. that don't batch cleanly with regular stocks).

`MIN_HISTORY_DAYS` filters out names whose price history isn't long
enough for the feature pipeline (the model needs 252-day rolling stats).
A symbol with < MIN days lands in `failed_syms` and is dropped silently.

`_normalize_columns` handles yfinance's MultiIndex column layouts which
differ across versions.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from core.run_report import RunReport


MIN_HISTORY_DAYS = 250  # need at least this many days for the 252-day features

# Macro tickers needed for features. `^VIX` is renamed to `VIX` for
# downstream consistency; the rest keep their yfinance names.
MACRO_TICKERS: list[str] = [
    "^VIX", "SPY", "QQQ", "IWM", "TLT", "SHY", "HYG",
    "GLD", "USO", "UUP",
    # All 11 GICS sector SPDRs. XLB / XLC / XLRE added in 2026-05 with the
    # combo_v1 deploy — it allocates to the full sector set via the
    # time-series momentum sleeve. Existing v6/v8 features only consume
    # the first 8; the extras are harmless to features.py (it ignores
    # unrecognised macro columns when computing sector_rel_*).
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

MACRO_RENAME: dict[str, str] = {"^VIX": "VIX"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance column names (handles MultiIndex from newer versions)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def download_bars(symbols: list[str], days: int, logger,
                  report: "RunReport") -> dict[str, pd.DataFrame]:
    """Download recent daily bars from Yahoo Finance.

    Batched 50 at a time with one retry on failure. Returns a dict
    keyed by symbol — entries missing from the dict either failed to
    download or had < `MIN_HISTORY_DAYS` of history.
    """
    report.start_step("download_stocks")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    logger.info(f"  Date range: {start.date()} -> {end.date()}")
    logger.info(f"  Symbols to download: {len(symbols)}")

    data: dict[str, pd.DataFrame] = {}
    failed_syms: list[str] = []
    batch_size = 50
    n_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(symbols), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch = symbols[batch_idx:batch_idx + batch_size]
        tickers_str = " ".join(batch)
        batch_start = time.time()

        df = None
        last_err: object = None
        for attempt in range(1, 3):  # 1 try + 1 retry
            try:
                df = yf.download(
                    tickers_str, start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    group_by="ticker", auto_adjust=True, progress=False,
                    threads=True,
                )
                if df is not None and len(df) > 0:
                    break
                last_err = "empty dataframe"
            except Exception as e:
                last_err = e
            if attempt < 2:
                logger.warning(
                    f"  Batch {batch_num}/{n_batches} attempt {attempt} "
                    f"failed ({last_err}); retrying in 2s"
                )
                time.sleep(2)

        if df is None or len(df) == 0:
            logger.warning(f"  Batch {batch_num}/{n_batches} FAILED after retry: {last_err}")
            report.add_warning(f"Stock batch {batch_num} failed: {last_err}")
            failed_syms.extend(batch)
            time.sleep(0.3)
            continue

        batch_ok = 0
        for sym in batch:
            try:
                if len(batch) == 1:
                    sym_df = df.copy()
                else:
                    sym_df = df[sym].copy()
                sym_df = _normalize_columns(sym_df)
                sym_df = sym_df.dropna(subset=["close"])
                if len(sym_df) >= MIN_HISTORY_DAYS:
                    data[sym] = sym_df
                    batch_ok += 1
                else:
                    failed_syms.append(sym)
            except Exception:
                failed_syms.append(sym)

        batch_time = time.time() - batch_start
        logger.info(
            f"  Batch {batch_num}/{n_batches}: "
            f"{batch_ok}/{len(batch)} ok ({batch_time:.1f}s)"
        )
        time.sleep(0.3)

    logger.info(f"  Downloaded: {len(data)} symbols, failed: {len(failed_syms)} symbols")

    if data:
        sample_sym = next(iter(data))
        sample_df = data[sample_sym]
        logger.info(
            f"  Sample ({sample_sym}): {len(sample_df)} days, "
            f"{sample_df.index[0].date()} -> {sample_df.index[-1].date()}"
        )
        report.set("latest_data_date", str(sample_df.index[-1].date()))

    report.set("stocks_downloaded", len(data))
    report.set("stocks_failed", len(failed_syms))
    if failed_syms and len(failed_syms) <= 20:
        report.add_warning(f"Failed stocks: {', '.join(failed_syms)}")
    report.end_step("download_stocks")
    return data


def download_macro(days: int, logger, report: "RunReport") -> dict[str, pd.DataFrame]:
    """Download macro tickers, one at a time to avoid mixed-ticker batching issues."""
    report.start_step("download_macro")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    macro: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            df = _normalize_columns(df)
            name = MACRO_RENAME.get(ticker, ticker)
            df_clean = df.dropna(subset=["close"])
            if len(df_clean) > 0:
                macro[name] = df_clean
                logger.info(
                    f"  {name:6s}: {len(df_clean)} days, "
                    f"latest close: {df_clean['close'].iloc[-1]:.2f}"
                )
            else:
                missing.append(ticker)
                logger.warning(f"  {ticker}: no data returned")
        except Exception as e:
            missing.append(ticker)
            logger.warning(f"  {ticker}: FAILED - {e}")
            report.add_warning(f"Macro {ticker} failed: {e}")
        time.sleep(0.1)

    logger.info(f"  Macro: {len(macro)}/{len(MACRO_TICKERS)} downloaded")
    if missing:
        logger.warning(f"  Missing: {', '.join(missing)}")

    report.set("macro_downloaded", len(macro))
    report.set("macro_total", len(MACRO_TICKERS))
    report.set("macro_missing", missing)
    report.end_step("download_macro")
    return macro
