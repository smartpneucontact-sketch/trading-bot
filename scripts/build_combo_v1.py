"""Build the combo_v1 model bundle and save to model/combo_v1/model.pkl.

The bundle is a direct-weights strategy (not an ML model). It carries:

  - model: ComboStrategy instance with embedded ComboConfig
  - strategy_type: "direct_weights" (selects runner's compute_weights branch)
  - horizon: 5 (rebalance every 5 trading days — but the strategy itself
             uses 12-mo / 6-mo / 3-mo / 21-day momentum lookbacks)
  - feature_cols: empty (this strategy doesn't go through predict_rankings)
  - version: "combo_v1"

Run from the deploy directory:

    cd trading_system/deploy
    python scripts/build_combo_v1.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `core` importable when running from deploy/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.combo_strategy import ComboConfig, ComboStrategy

OUT = Path(__file__).resolve().parent.parent / "model" / "combo_v1" / "model.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = ComboConfig(
        # Sleeve allocations (must sum to 1.0; __post_init__ asserts)
        mom_weight=0.30, mom_top_n=30,
        fast_weight=0.25, fast_top_n=20,
        dual_weight=0.20, dual_top_n=30,
        ts_weight=0.25,
        # Vol-targeting (per sleeve, captured in bundle for full reproducibility)
        dual_vol_target=0.15,
        ts_vol_target=0.20,
        # Risk overlay
        spy_dd_lookback=60,
        spy_full_dd=0.08,
        spy_cash_dd=0.18,
        max_gross_exposure=1.0,
        min_position_weight=0.005,
    )
    strategy = ComboStrategy(config=config)

    bundle = {
        "model": strategy,
        "strategy_type": "direct_weights",
        "feature_cols": [],   # not used; direct-weights bypasses predict_rankings
        "horizon": 5,
        "version": "combo_v1",
        "tag": "Combo 4-signal multi-horizon multi-asset momentum",
        # Carry the config explicitly for inspection-friendly bundles
        "combo_config": config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        # Backtest reference (from /Traiding 11/REPORT.md, 10-yr Apr-2016 → Mar-2026)
        "backtest_reference": {
            "window": "2016-04-01..2026-03-27",
            "universe_size": 1040,
            "tc_bps_per_side": 5,
            "leverage_for_reference": 2.5,
            "mean_monthly_return": 0.0417,
            "median_monthly_return": 0.0295,
            "sharpe": 1.00,
            "max_drawdown": -0.595,
            "calmar": 0.76,
            "hit_rate_monthly": 0.65,
            "hit_rate_3pct_monthly": 0.50,
            "worst_year_pct": -0.386,
            "worst_year": 2022,
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved bundle → {OUT}")
    print(f"  size: {OUT.stat().st_size / 1024:.1f} KB")
    print(f"  strategy_type: {bundle['strategy_type']}")
    print(f"  version: {bundle['version']}")
    print(f"  saved_at: {bundle['saved_at']}")


if __name__ == "__main__":
    main()
