"""Model inference orchestrator — features + macro → ranked predictions.

`predict_rankings` is the entry point. It:

  1. Computes per-stock features (via the version-specific function from
     `core.features.get_feature_func`)
  2. Joins macro features by date (forward-filling holidays)
  3. Adds cross-sectional rank features (`cs_*`) across the cross-section
  4. Filters stocks whose feature coverage drops below 95% or contains
     NaN in any present column
  5. Zero-fills macro features that are entirely absent (handles old
     `SP500_*` vs new `SPY_*` naming drift without dropping every stock)
  6. Calls `model.predict(...)` and returns a list sorted descending by
     predicted return
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.features import add_cross_sectional_ranks, get_feature_func

if TYPE_CHECKING:
    from core.run_report import RunReport


# Min fraction of expected features that must be present per-stock. If a
# stock is missing more than 5% of its features we drop it (data is too
# stale or incomplete to trust). Macro features that are *uniformly*
# missing across the whole cross-section get zero-filled instead — a
# constant offset doesn't break ranking, dropping every stock would.
MIN_FEATURE_COVERAGE = 0.95


def _is_keepable(sym: str) -> bool:
    """Filter for genuine tradeable tickers.

    Keeps BRK-B and BF-B (dashes get converted to dots at order time).
    Drops non-ASCII, slashes (warrant/unit shares), or >8 chars.
    """
    if not sym or not sym.isascii() or "/" in sym or len(sym) > 8:
        return False
    cleaned = sym.replace("-", "").replace(".", "")
    return cleaned.isalpha()


def predict_rankings(
    stock_data: dict,
    macro_features: pd.DataFrame,
    model,
    feature_cols: list,
    logger,
    report: "RunReport",
    feature_version: str = "v4",
) -> list[tuple[str, float]]:
    """Generate predictions for all stocks, return sorted rankings.

    Returns `[(symbol, predicted_return_pct), ...]` sorted descending.
    """
    report.start_step("predict")
    predictions: dict[str, float] = {}
    failures: list[tuple[str, str]] = []
    feat_func = get_feature_func(feature_version)

    has_cs_features = any(c.startswith("cs_") for c in feature_cols)

    # --- Step 1: per-stock features, joined with macro ----------------
    latest_rows: dict[str, pd.Series] = {}
    for sym, df in stock_data.items():
        try:
            stock_feats = feat_func(df)
            if not macro_features.empty:
                merged = stock_feats.join(macro_features, how="left")
                macro_cols = macro_features.columns
                merged[macro_cols] = merged[macro_cols].ffill()
            else:
                merged = stock_feats
            latest_rows[sym] = merged.iloc[-1]
        except Exception as e:
            failures.append((sym, str(e)))

    logger.info(f"  Features computed: {len(latest_rows)} ok, {len(failures)} failed")

    # --- Step 2: cross-sectional ranks --------------------------------
    if has_cs_features and latest_rows:
        logger.info(
            f"  Adding cross-sectional ranks across {len(latest_rows)} stocks..."
        )
        latest_rows = add_cross_sectional_ranks(latest_rows, feature_cols)
        dropped_cs = latest_rows.pop("__cs_dropped__", []) if isinstance(latest_rows, dict) else []
        for sym in dropped_cs:
            failures.append((sym, "missing cross-sectional inputs"))
        if dropped_cs:
            logger.info(f"  Dropped {len(dropped_cs)} stocks lacking CS inputs")

    # --- Step 3: per-stock inference ----------------------------------
    missing_warned = False
    for sym, row_series in latest_rows.items():
        try:
            row = row_series.to_frame().T
            avail_cols = [c for c in feature_cols if c in row.columns]

            if len(avail_cols) < len(feature_cols) * MIN_FEATURE_COVERAGE:
                missing_count = len(feature_cols) - len(avail_cols)
                failures.append((sym, f"too many missing features ({missing_count}/{len(feature_cols)})"))
                continue

            row = row[avail_cols]
            if row.isna().any(axis=1).iloc[0]:
                # NaN inside a feature that IS present — per-stock data
                # quality issue. Drop rather than impute.
                nan_cols = [c for c in row.columns if row[c].isna().iloc[0]]
                failures.append((sym, f"NaN in {len(nan_cols)} features"))
                continue

            # Macro feature naming drift: training script used SP500_*,
            # new yfinance code uses SPY_* (or vice versa). Zero-fill
            # absent columns rather than failing every stock.
            absent = [c for c in feature_cols if c not in row.columns]
            if absent:
                if not missing_warned:
                    logger.info(
                        f"  Zero-filling {len(absent)} feature(s) absent "
                        f"from live data: {absent[:8]}"
                        + ("…" if len(absent) > 8 else "")
                    )
                    missing_warned = True
                for col in absent:
                    row[col] = 0.0

            row = row[feature_cols]  # column order must match training
            pred = model.predict(row)[0]
            if np.isfinite(pred):
                predictions[sym] = float(pred)
            else:
                failures.append((sym, "non-finite prediction"))
        except Exception as e:
            failures.append((sym, str(e)))

    # --- Step 4: filter untradeable tickers, sort ---------------------
    tradeable_preds = {
        sym: pred for sym, pred in predictions.items() if _is_keepable(sym)
    }
    skipped = len(predictions) - len(tradeable_preds)
    if skipped > 0:
        logger.info(f"  Filtered {skipped} untradeable tickers (non-ASCII / malformed)")

    ranked = sorted(tradeable_preds.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"  Predictions: {len(ranked)} ok, {len(failures)} failed")
    if failures:
        reason_counts = Counter(reason for _, reason in failures)
        for reason, count in reason_counts.most_common(5):
            logger.info(f"    Failure: {count}x — {reason}")
        if len(failures) <= 10:
            for sym, reason in failures:
                logger.info(f"    SKIP {sym}: {reason}")

    if predictions:
        preds_arr = np.array(list(predictions.values()))
        report.set("prediction_range", {
            "min": float(np.min(preds_arr)),
            "max": float(np.max(preds_arr)),
            "mean": float(np.mean(preds_arr)),
            "std": float(np.std(preds_arr)),
            "median": float(np.median(preds_arr)),
        })
        logger.info(
            f"  Prediction range: {np.min(preds_arr):.2f}% to "
            f"{np.max(preds_arr):.2f}% "
            f"(mean: {np.mean(preds_arr):.2f}%, std: {np.std(preds_arr):.2f}%)"
        )

    report.set("n_predictions", len(ranked))
    report.set("prediction_failures", len(failures))

    if ranked:
        logger.info(f"  Top 5:    {[(s, f'{p:+.2f}%') for s, p in ranked[:5]]}")
        logger.info(f"  Bottom 5: {[(s, f'{p:+.2f}%') for s, p in ranked[-5:]]}")

    report.end_step("predict")
    return ranked
