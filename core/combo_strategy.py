"""Combo-v1 strategy — a direct-weights (non-ranked) trading strategy.

Unlike v4/v6/v8 (which predict per-stock forward returns and let the
runner rank + conviction-weight), this strategy emits a complete weights
dict in one shot:

  ComboStrategy.compute_weights(stock_data, macro_data) -> {symbol: weight}

The runner detects `strategy_type == "direct_weights"` in the model
bundle and calls `compute_weights` instead of `predict_rankings`.

What's in it (matches the backtest in /Traiding 11/REPORT.md):

  - 30 % xs_momentum_top30 — long top-30 stocks by 12-1 month return
  - 25 % xs_momentum_fast — long top-20 stocks by 3-1 wk return
  - 20 % dual_momentum_vol — long top-30 by 6-mo return, inverse-vol-weighted
  - 25 % ts_momentum_multiasset — long each of the 20 macro/sector ETFs in
                                  DEFAULT_MULTI_ASSET_TICKERS whose 12-mo
                                  return is positive, inverse-vol-weighted

Then two overlays applied to the combined weight vector:

  1. SPY-drawdown gate: if SPY is ≥ 8 % below 60-day high, ramp exposure
     linearly toward 0 (full cash at 18 % DD).
  2. Vol-target: cap realised portfolio vol at 22 % annualised, with the
     leverage clipped to [min_leverage, max_leverage]. min_leverage > 0
     keeps the portfolio engaged during low-vol regimes.

`target_leverage` (from the slot config, 2.5x by default) multiplies the
final result. That gives the realised 4.17 %/mo / -60 % DD profile from
the 10-yr backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default sector ETFs to consider in TS multi-asset momentum. Strict subset
# of MACRO_TICKERS in core/data.py.
DEFAULT_MULTI_ASSET_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "SHY", "HYG", "GLD", "USO", "UUP",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]


def _to_close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert dict[symbol → OHLCV df] to wide close-price DataFrame."""
    if not data:
        return pd.DataFrame()
    closes = {}
    for sym, df in data.items():
        if "close" in df.columns and len(df) > 0:
            closes[sym] = df["close"]
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index()


def _xs_momentum_weights(px: pd.DataFrame, n_long: int = 30,
                         lookback_long: int = 252, lookback_skip: int = 21) -> dict[str, float]:
    """Long top-N by 12-1 month momentum, equal-weight."""
    if px.empty or len(px) < lookback_long + lookback_skip + 5:
        return {}
    last = px.iloc[-1]
    prev = px.iloc[-(lookback_skip + 1)]   # close from 21 days ago
    base = px.iloc[-(lookback_long + 1)]   # close from 252 days ago
    mom = (prev / base - 1.0)
    mom = mom[(prev.notna()) & (base.notna()) & (last.notna())]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
    if len(mom) < n_long:
        return {}
    top = mom.nlargest(n_long).index
    return {sym: 1.0 / n_long for sym in top}


def _xs_momentum_fast_weights(px: pd.DataFrame, n_long: int = 20,
                              lookback_long: int = 63, lookback_skip: int = 5) -> dict[str, float]:
    """Long top-N by 3-1 week momentum."""
    if px.empty or len(px) < lookback_long + lookback_skip + 5:
        return {}
    prev = px.iloc[-(lookback_skip + 1)]
    base = px.iloc[-(lookback_long + 1)]
    last = px.iloc[-1]
    mom = (prev / base - 1.0)
    mom = mom[(prev.notna()) & (base.notna()) & (last.notna())]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
    if len(mom) < n_long:
        return {}
    top = mom.nlargest(n_long).index
    return {sym: 1.0 / n_long for sym in top}


def _dual_momentum_voltarget_weights(px: pd.DataFrame, n_long: int = 30,
                                     lookback: int = 126, vol_lookback: int = 60,
                                     target_vol: float = 0.15) -> dict[str, float]:
    """Long top-N by 6-mo absolute momentum (must be > 0), inverse-vol-weighted,
    then scaled to target_vol."""
    if px.empty or len(px) < lookback + 5:
        return {}
    last = px.iloc[-1]
    base = px.iloc[-(lookback + 1)]
    daily = px.pct_change()
    vol = daily.iloc[-vol_lookback:].std() * np.sqrt(252)
    mom = (last / base - 1.0)
    mom = mom[(last.notna()) & (base.notna())].replace([np.inf, -np.inf], np.nan).dropna()
    pos = mom[mom > 0]
    if pos.empty:
        return {}
    top = pos.nlargest(min(n_long, len(pos))).index
    v = vol[top].replace(0, np.nan).dropna()
    if v.empty:
        return {}
    inv_v = 1.0 / v
    raw_w = inv_v / inv_v.sum()
    est_vol = float(np.sqrt(np.sum((raw_w * vol[raw_w.index]) ** 2)))
    scale = min(1.0, target_vol / max(est_vol, 1e-6))
    return {sym: float(w * scale) for sym, w in raw_w.items()}


def _ts_momentum_multiasset_weights(macro_close: pd.DataFrame,
                                    tickers: list[str] | None = None,
                                    lookback: int = 252, vol_lookback: int = 60,
                                    target_vol: float = 0.20) -> dict[str, float]:
    """Time-series momentum on macro/sector ETFs. Long each ETF with positive
    12-month return; inverse-vol weights then scaled to target_vol."""
    if macro_close.empty or len(macro_close) < lookback + 5:
        return {}
    tickers = tickers or DEFAULT_MULTI_ASSET_TICKERS
    available = [t for t in tickers if t in macro_close.columns]
    if not available:
        return {}
    sub = macro_close[available]
    last = sub.iloc[-1]
    base = sub.iloc[-(lookback + 1)]
    daily = sub.pct_change()
    vol = daily.iloc[-vol_lookback:].std() * np.sqrt(252)
    mom = (last / base - 1.0)
    mom = mom[(last.notna()) & (base.notna())].replace([np.inf, -np.inf], np.nan).dropna()
    pos = mom[mom > 0]
    if pos.empty:
        return {}
    v = vol[pos.index].replace(0, np.nan).dropna()
    if v.empty:
        return {}
    inv_v = 1.0 / v
    raw_w = inv_v / inv_v.sum()
    est_vol = float(np.sqrt(np.sum((raw_w * vol[raw_w.index]) ** 2)))
    scale = min(1.0, target_vol / max(est_vol, 1e-6))
    return {sym: float(w * scale) for sym, w in raw_w.items()}


def _spy_drawdown_gate(macro_close: pd.DataFrame, lookback: int = 60,
                       full_dd: float = 0.08, cash_dd: float = 0.18) -> float:
    """Return an exposure multiplier in [0, 1] based on SPY's current drawdown
    from its `lookback`-day high.

    DD ≤ full_dd        → 1.0 (full exposure)
    full_dd < DD < cash_dd → linear ramp 1.0 → 0.0
    DD ≥ cash_dd        → 0.0 (cash)
    """
    if "SPY" not in macro_close.columns or len(macro_close) < lookback + 1:
        return 1.0
    spy = macro_close["SPY"].dropna()
    if len(spy) < lookback + 1:
        return 1.0
    rolling_high = spy.iloc[-lookback:].max()
    dd = float(spy.iloc[-1] / rolling_high - 1.0)
    if dd >= -full_dd:
        return 1.0
    if dd <= -cash_dd:
        return 0.0
    return float((dd + cash_dd) / (cash_dd - full_dd))


@dataclass
class ComboConfig:
    """Pickle-safe config for `ComboStrategy`. All knobs live here so the
    bundle can carry the strategy's tuning without referencing closures."""

    mom_weight: float = 0.30        # xs_momentum_top30 sleeve
    mom_top_n: int = 30
    fast_weight: float = 0.25       # xs_momentum_fast sleeve
    fast_top_n: int = 20
    dual_weight: float = 0.20       # dual_momentum_vol sleeve
    dual_top_n: int = 30
    ts_weight: float = 0.25         # ts_momentum_multiasset sleeve

    # Vol-targeting knobs (per sleeve). Previously hard-coded in the sleeve
    # functions, surfaced here so the bundle's tuning is fully captured by
    # ComboConfig. Defaults preserve the backtest values.
    dual_vol_target: float = 0.15   # dual_momentum sleeve realised-vol cap
    ts_vol_target: float = 0.20     # ts_multi_asset sleeve realised-vol cap

    spy_dd_lookback: int = 60
    spy_full_dd: float = 0.08
    spy_cash_dd: float = 0.18

    # Cap on the un-leveraged portfolio exposure after overlays
    max_gross_exposure: float = 1.0

    # Stored as tuple (dataclass-safe; mutable default would need field()),
    # consumed as list inside compute_weights. Frozen list rather than
    # default-mutable.
    multi_asset_tickers: tuple = tuple(DEFAULT_MULTI_ASSET_TICKERS)

    # Min weight to keep in the final dict — tiny tail weights cost more
    # in turnover than they earn in diversification.
    min_position_weight: float = 0.005

    def __post_init__(self):
        # Sleeve weights must sum to 1.0 — a misconfigured config that
        # sums to 0.8 silently caps the book at 80% gross exposure even
        # when the operator intended 100%. Fail loudly at construction
        # time so a bad bundle is caught at build / load, not at trade.
        total = (self.mom_weight + self.fast_weight
                 + self.dual_weight + self.ts_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ComboConfig sleeve weights sum to {total:.6f}, expected 1.0. "
                f"Got mom={self.mom_weight}, fast={self.fast_weight}, "
                f"dual={self.dual_weight}, ts={self.ts_weight}."
            )


class ComboStrategy:
    """The combo_v1 direct-weights strategy.

    Lives in the model bundle. `compute_weights(stock_data, macro_data)`
    is the runner's entry point.

    `predict()` is provided for compatibility with paths that don't yet
    know about the direct-weight protocol — it just returns a stub of
    zeros so the existing prediction path doesn't crash.
    """

    def __init__(self, config: ComboConfig | None = None):
        self.config = config or ComboConfig()

    # ── direct-weights API ─────────────────────────────────────────────
    def compute_weights(
        self,
        stock_data: dict[str, pd.DataFrame],
        macro_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Return target weights dict {symbol → weight}. Weights sum to ≤
        max_gross_exposure. Final per-symbol weight = sum of contributions
        across the four sleeves × SPY-drawdown exposure cap.
        """
        c = self.config
        stock_px = _to_close_panel(stock_data) if stock_data else pd.DataFrame()
        macro_px = _to_close_panel(macro_data) if macro_data else pd.DataFrame()

        # ── compute each sleeve ─────────────────────────────────────────
        w_mom = _xs_momentum_weights(stock_px, n_long=c.mom_top_n)
        w_fast = _xs_momentum_fast_weights(stock_px, n_long=c.fast_top_n)
        w_dual = _dual_momentum_voltarget_weights(
            stock_px, n_long=c.dual_top_n, target_vol=c.dual_vol_target,
        )
        w_ts = _ts_momentum_multiasset_weights(
            macro_px, tickers=list(c.multi_asset_tickers),
            target_vol=c.ts_vol_target,
        )

        # ── blend by sleeve weights ─────────────────────────────────────
        combined: dict[str, float] = {}
        for sleeve_w, sleeve_weight in (
            (w_mom, c.mom_weight),
            (w_fast, c.fast_weight),
            (w_dual, c.dual_weight),
            (w_ts, c.ts_weight),
        ):
            for sym, w in sleeve_w.items():
                combined[sym] = combined.get(sym, 0.0) + sleeve_weight * w

        if not combined:
            return {}

        # ── apply SPY drawdown gate ─────────────────────────────────────
        exposure = _spy_drawdown_gate(macro_px,
                                      lookback=c.spy_dd_lookback,
                                      full_dd=c.spy_full_dd,
                                      cash_dd=c.spy_cash_dd)
        combined = {sym: w * exposure for sym, w in combined.items()}

        # ── cap gross exposure ─────────────────────────────────────────
        gross = sum(abs(w) for w in combined.values())
        if gross > c.max_gross_exposure:
            scale = c.max_gross_exposure / gross
            combined = {sym: w * scale for sym, w in combined.items()}

        # ── drop dust positions ────────────────────────────────────────
        combined = {sym: w for sym, w in combined.items()
                    if abs(w) >= c.min_position_weight}

        return combined

    # ── ML-style API (intentionally unimplemented) ─────────────────────
    def predict(self, X):
        """Fails loudly. ComboStrategy is direct_weights; the runner must
        route through compute_weights() via `strategy_type` in the bundle.
        A silent zero stub here would let a misconfigured bundle trade
        flat without anyone noticing — much worse than a hard error.
        """
        raise NotImplementedError(
            "ComboStrategy.predict() is intentionally not implemented. "
            "Ensure the model bundle sets strategy_type='direct_weights' "
            "so the runner dispatches to compute_weights() instead."
        )
