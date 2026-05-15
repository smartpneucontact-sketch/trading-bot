"""Portfolio construction — regime detection and conviction-weighted sizing.

`compute_live_regime_score` produces a composite signal in [-1, +1]:
VIX (low → favorable), SPY trend vs 200-SMA, SPY 20d momentum, and
HYG credit health. Weights and clipping are tuned to keep the score
intuitive (0 ≈ neutral, +0.3 ≈ unambiguously risk-on, -0.3 ≈ risk-off).

`regime_to_exposure` maps that score onto a 0.4–1.0 exposure multiplier,
which is then applied to position weights via `conviction_weights` or
its sector-neutral sibling.

`conviction_weights` allocates capital proportionally to prediction
strength, capped at 2× equal-weight to prevent any single name dominating.
`sector_neutral_weights` adds a sector cap on top of that for the V8
model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))


# ═══════════════════════════════════════════════════════════════════════════
# Regime detection
# ═══════════════════════════════════════════════════════════════════════════

def compute_live_regime_score(macro_data: dict[str, pd.DataFrame]) -> float:
    """Composite market regime score in [-1, +1].

    -1 = hostile (high VIX, weak trend, weak credit) → exposure cut.
    +1 = favorable → full exposure.

    Returns 0.0 if no macro inputs are available (neutral fallback).
    """
    signals: list[tuple[str, float, float]] = []

    if "VIX" in macro_data:
        vix = macro_data["VIX"]["close"]
        if len(vix) >= 120:
            vix_pct = vix.rank(pct=True).iloc[-1]
            signals.append(("vix", 0.30, 1 - 2 * vix_pct))

    spy_data = macro_data.get("SPY")
    if spy_data is not None:
        spy_close = spy_data["close"]
        if len(spy_close) >= 200:
            sma_200 = spy_close.rolling(200).mean().iloc[-1]
            dist_pct = (spy_close.iloc[-1] - sma_200) / (sma_200 + 1e-8) * 100
            signals.append(("trend", 0.30, max(-1, min(1, dist_pct / 5))))

            if len(spy_close) >= 20:
                ret_20 = (spy_close.iloc[-1] / spy_close.iloc[-20] - 1) * 100
                signals.append(("momentum", 0.20, max(-1, min(1, ret_20 / 5))))

    if "HYG" in macro_data:
        hyg = macro_data["HYG"]["close"]
        if len(hyg) >= 50:
            hyg_sma50 = hyg.rolling(50).mean().iloc[-1]
            hyg_dist = (hyg.iloc[-1] - hyg_sma50) / (hyg_sma50 + 1e-8) * 100
            signals.append(("credit", 0.20, max(-1, min(1, hyg_dist / 2))))

    if not signals:
        return 0.0

    total_weight = sum(w for _, w, _ in signals)
    score = sum(w * s for _, w, s in signals) / total_weight
    return max(-1.0, min(1.0, score))


def regime_to_exposure(regime_score: float) -> float:
    """Map regime score [-1, +1] to portfolio exposure multiplier [0.4, 1.0].

      score ≥ +0.3 → 1.00 (full exposure)
      score =  0.0 → 0.85
      score = -0.3 → 0.70
      score ≤ -1.0 → 0.40 (floor)
    """
    if regime_score >= 0.3:
        return 1.0
    if regime_score >= 0.0:
        return 0.85 + (regime_score / 0.3) * 0.15
    if regime_score >= -0.3:
        return 0.7 + ((regime_score + 0.3) / 0.3) * 0.15
    return max(0.4, 0.7 + (regime_score + 0.3) / 0.7 * 0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Sizing — conviction-weighted, optionally sector-neutral
# ═══════════════════════════════════════════════════════════════════════════

def conviction_weights(
    predictions: list[tuple[str, float]],
    top_n: int = 20,
    max_weight_multiple: float = 2.0,
    regime_exposure: float = 1.0,
) -> dict[str, float]:
    """Convert ranked predictions into conviction-weighted allocations.

    Returns `{symbol: weight}` where weights sum to `regime_exposure`.
    Each weight is capped at `max_weight_multiple × equal-weight` to
    prevent any single high-conviction pick from dominating the book.
    """
    top = predictions[:top_n]
    if not top:
        return {}

    preds = np.array([p for _, p in top])
    preds_shifted = preds - preds.min() + 0.01
    denom = preds_shifted.sum()
    raw_weights = preds_shifted / denom if denom > 0 else np.ones(len(preds)) / len(preds)

    equal_weight = 1.0 / top_n
    max_weight = equal_weight * max_weight_multiple
    capped = np.minimum(raw_weights, max_weight)
    capped_sum = capped.sum()
    if capped_sum > 0:
        capped = capped / capped_sum * regime_exposure
    else:
        capped = np.ones(len(capped)) / len(capped) * regime_exposure

    return {sym: round(float(w), 6) for (sym, _), w in zip(top, capped)}


def sector_neutral_weights(
    predictions: list[tuple[str, float]],
    sector_map: dict[str, str],
    top_n: int = 20,
    max_per_sector: int = 3,
    max_weight_multiple: float = 2.0,
    regime_exposure: float = 1.0,
) -> dict[str, float]:
    """Sector-constrained conviction-weighted portfolio (V8).

    Walks the predictions in order and accepts each name only if its
    sector has fewer than `max_per_sector` picks already. Stocks from
    overrepresented sectors get skipped — net effect is to spread the
    book across sectors while still preferring higher-conviction names
    within each sector.
    """
    if not predictions:
        return {}

    selected: list[tuple[str, float]] = []
    sector_counts: dict[str, int] = {}

    for sym, pred in predictions:
        if len(selected) >= top_n:
            break
        sector = sector_map.get(sym, "Unknown")
        count = sector_counts.get(sector, 0)
        if count < max_per_sector:
            selected.append((sym, pred))
            sector_counts[sector] = count + 1

    if not selected:
        return {}

    preds = np.array([p for _, p in selected])
    preds_shifted = preds - preds.min() + 0.01
    denom = preds_shifted.sum()
    raw_weights = preds_shifted / denom if denom > 0 else np.ones(len(preds)) / len(preds)

    equal_weight = 1.0 / len(selected)
    max_weight = equal_weight * max_weight_multiple
    capped = np.minimum(raw_weights, max_weight)
    capped_sum = capped.sum()
    if capped_sum > 0:
        capped = capped / capped_sum * regime_exposure
    else:
        capped = np.ones(len(capped)) / len(capped) * regime_exposure

    return {sym: round(float(w), 6) for (sym, _), w in zip(selected, capped)}


def load_sector_map_for_pipeline() -> dict[str, str]:
    """Load the cached `sector_map.json` used by the V8 sector-neutral model.

    Returns an empty dict if the cache is missing (in which case
    `sector_neutral_weights` falls back to treating every stock as the
    "Unknown" sector and the per-sector cap effectively caps the whole
    book at `max_per_sector` stocks).
    """
    sector_cache = _DATA_DIR / "sector_map.json"
    if sector_cache.exists():
        try:
            with open(sector_cache) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
