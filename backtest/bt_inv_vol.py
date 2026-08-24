"""Momentum + Inverse Volatility Weighting Strategy

Two sequential regime filters determine whether to be in equities or bonds:

  1. Dual Momentum — SPY 12-month return must beat AGG AND be positive
  2. 200-day MA    — SPY price must be above its 200-day SMA

If both pass → rank universe by 12-month return, take top-N, weight by
inverse volatility (lower vol = larger allocation).

If either fails → 95% AGG, 5% cash.

Data feed order (enforced by run_backtest):
  index 0  → SPY
  index 1  → AGG
  index 2+ → stock universe
"""
from __future__ import annotations

import datetime
import logging
import math
import pathlib
import sys
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.strategy.combined_momentum import _FALLBACK_UNIVERSE
from backtest._price_cache import get_prices
from backtest._hysteresis import apply_hysteresis, smooth_weights
from backtest._sector_map import SECTOR_MAP
from backtest._signal import ICTracker, cross_sectional_zscore, spearman_corr
from backtest._weighting import (
    erc_weights,
    inv_vol_weights,
    ledoit_wolf_constant_corr,
    momentum_rank_weights,
    sample_covariance,
)

logger = logging.getLogger(__name__)

# Pre-fetched VIX daily closes for dynamic sleeve stress detection.
# Populated by run_backtest() when dynamic_sleeve=True and vix_threshold > 0.
_VIX_LOOKUP: dict[datetime.date, float] = {}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MomentumInvVolStrategy(bt.Strategy):

    params = (
        ("top_n", 5),               # momentum stocks to hold in equity regime
        ("ma_period", 200),         # SPY moving average period for crash filter
        ("momentum_period", 252),   # ~12 months of trading days
        ("vol_window", 20),         # rolling window for inv-vol weighting
        ("cash_buffer", 0.05),      # fraction kept in cash at every rebalance
        ("rebalance_freq", "monthly"),
        ("regime_filter", False),   # True = dual-momentum + MA check; False = always equities (highest Sharpe)
        ("sharpe_ranking", False),  # True = rank by mom/vol (risk-adjusted); False = rank by raw 12m return
        ("vol_targeting", False),       # True = scale exposure by target_vol / realized_portfolio_vol
        ("vol_target", 0.15),           # annualised target portfolio volatility (15%)
        ("vol_lookback", 21),           # trading days of portfolio returns used for realized vol
        ("abs_momentum_filter", False), # True = only hold stocks with positive abs-momentum return; spare slots go to cash
        ("abs_momentum_period", 126),   # lookback for the abs filter (default 126 bars ≈ 6 months, independent of ranking)
        ("momentum_gap", 0),            # skip most-recent N bars in ranking (21 ≈ 12-1 skip-month momentum)
        ("blended_momentum", False),    # True = rank by avg percentile across 3/6/12-month windows
        ("blend_windows", (63, 126, 252)),  # lookbacks (bars) blended when blended_momentum=True
        ("max_weight", 1.0),            # cap on any single holding's portfolio weight (1.0 = no cap)
        ("weighting", "inv_vol"),       # "inv_vol" (independent names) or "erc" (covariance-aware equal risk contribution)
        ("cov_window", 63),             # trading days of returns used to estimate the covariance for ERC (~3 months)
        ("cov_shrinkage", "ledoit_wolf"),  # "ledoit_wolf" (constant-corr shrinkage) or "none" (raw sample covariance)
        ("mom_tilt", 0.0),              # λ in [0,1]: w = (1-λ)·w_ERC + λ·w_momentum_rank (0 = max Sharpe, 1 = full conviction)
        ("tranches", 1),                # overlapping rebalance tranches: N sub-books rebalanced on staggered days (1 = baseline)
        ("keep_n", 0),                  # holding hysteresis: retain a held name while ranked in top keep_n (0 or <=top_n = off). Cuts churn without broadening (book stays top_n).
        ("weight_smoothing", 1.0),      # EWMA blend of new inv-vol targets toward prior weights, in (0,1]; 1.0 = off, lower = stickier. Single-book path only (tranches=1).
        ("trend_overlay", False),       # True = scale equity exposure when SPY < trend MA (partial de-risk, not full switch)
        ("trend_derisk", 0.5),          # equity exposure fraction retained when SPY < trend MA; remainder → AGG
        # ── Pillar 1: dynamic signal tuning (single-book path only) ────────
        ("accel_tilt", 0.0),            # λ in [0,1]: rank by (1-λ)·z(12m return) + λ·z(acceleration). 0 = off (raw 12m).
        ("ic_weighting", False),        # True = rank by IC-weighted multi-horizon z-blend (overrides accel_tilt)
        ("ic_horizons", (21, 63, 126, 252)),  # lookback bars blended under IC weighting (1/3/6/12-month)
        ("ic_beta", 0.2),               # EWMA memory for the rolling IC (higher = more reactive; ~5-6mo at 0.2)
        ("ic_tau", 0.10),               # softmax temperature for IC→weights (lower = sharper horizon concentration)
        # ── Pillar 2.4: adaptive smoothing ────────────────────────────────────
        ("adaptive_smoothing", False),  # True = α tightens when cross-sectional noise spikes or SPY<MA200
        ("smooth_alpha_min", 0.2),      # tightest α used during high noise / crash (sticky)
        ("smooth_alpha_max", 0.8),      # loosest α used during calm, low-noise markets (reactive)
        ("cs_noise_window", 60),        # monthly rebalances of CS-vol history kept for percentile rank (~5 years)
        # ── Pillar 3: safe-asset routing ──────────────────────────────────────
        ("safe_asset", "AGG"),          # ticker parked in when de-risking: "AGG" (default) or "BIL"/"SGOV"
        # ── Pillar 4: diversifying sleeve ─────────────────────────────────────
        ("sleeve_pct", 0.0),            # fraction of portfolio permanently in defensive sleeve (0 = off)
        ("sleeve_assets", ()),          # tickers in the sleeve, e.g. ("AGG", "GLD", "TLT")
        ("sleeve_weighting", "equal"),  # "equal" or "inv_vol" within the sleeve
        ("sector_cap", 1.0),            # max aggregate weight for any single GICS sector (1.0 = off)
        # ── Catastrophic stop ─────────────────────────────────────────────────
        ("cat_stop_pct", 0.0),          # per-name hard stop: exit if close falls >cat_stop_pct below avg cost basis (0 = off)
        # ── Dynamic sleeve (conditional GLD expansion) ───────────────────────
        ("dynamic_sleeve", False),      # True = expand sleeve to sleeve_pct_stress during market stress
        ("sleeve_pct_stress", 0.20),    # sleeve fraction during stress periods (default 20%)
        ("vix_threshold", 30.0),        # VIX level that triggers stress mode (0 = disable VIX check)
        ("spy_monthly_dd", -0.08),      # SPY 1-month return that triggers stress (-1 = disable SPY check)
        # ── Recession rotation ────────────────────────────────────────────────
        ("recession_mode", False),      # True = rotate to defensive universe during GDP recessions
        ("recession_defensives", ()),   # tickers eligible during recession (consumer staples/healthcare/utilities)
        ("recession_months", frozenset()),  # precomputed frozenset of (year, month) in recession
        ("printlog", True),
    )

    def __init__(self):
        self.spy = self.datas[0]
        self.agg = self.datas[1]

        # Resolve the safe-asset feed by name so BIL/SGOV can be loaded at any
        # feed index. Falls back to AGG when the ticker isn't in the feed list,
        # preserving exact backward compatibility (safe_asset="AGG" default).
        safe_name = self.params.safe_asset
        self._safe_feed = next(
            (d for d in self.datas if d._name == safe_name),
            self.agg,
        )

        # Resolve sleeve feeds by name; they are excluded from momentum ranking.
        _sleeve_names = set(self.params.sleeve_assets)
        self._sleeve_feeds: list = [d for d in self.datas if d._name in _sleeve_names]

        # Exclude SPY, AGG, the safe-asset feed, and sleeve feeds from the
        # momentum-ranked universe so defensive assets are never selected as
        # momentum stocks.
        _overhead = {self.spy, self.agg, self._safe_feed} | set(self._sleeve_feeds)
        self.stocks = [d for d in self.datas[2:] if d not in _overhead]

        self.spy_ma200 = bt.indicators.SimpleMovingAverage(
            self.spy, period=self.params.ma_period
        )

        self._last_period: tuple = (-1, -1)
        self._regime: str = "init"
        self._daily_values: list[float] = []  # portfolio value recorded every bar for vol targeting

        # Overlapping-tranche state. Tranche i rebalances once per month on the
        # first trading day on/after day _tranche_day[i], holding 1/N of capital;
        # the book is the average of the N tranches. N=1 → exact baseline path.
        n = self.params.tranches
        self._tranche_day = [1 + round(i * 28 / n) for i in range(n)] if n > 1 else [1]
        self._tranche_period: list = [(-1, -1)] * n
        self._tranche_weights: list[dict] = [dict() for _ in range(n)]

        # Prior-rebalance equity weights (book only), for EWMA weight smoothing.
        self._prev_weights: dict = {}

        # Pillar 1 — rolling-IC horizon weighting state. The tracker holds one
        # EWMA'd IC per horizon; _ic_prior_z / _ic_prior_close snapshot the
        # previous rebalance so the *next* rebalance can score each horizon's
        # past prediction against the return that has since been realised.
        self._ic_tracker = ICTracker(self.params.ic_horizons, self.params.ic_beta)
        self._ic_prior_z: dict[int, dict[str, float]] | None = None
        self._ic_prior_close: dict[str, float] = {}

        # Pillar 2.4 — adaptive smoothing: rolling history of monthly CS-vol
        # observations (one per rebalance) used to percentile-rank the current
        # noise level without look-ahead.
        self._cs_vol_history: list[float] = []

        # Catastrophic stop: tickers stopped out in the current rebalance period.
        # Cleared at the start of each rebalance so a name can be re-selected next month.
        self._stopped_this_period: set[str] = set()
        self._cat_stop_count: int = 0  # total stop triggers for reporting

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mom_return(self, data) -> float:
        """Momentum-ranking return; -9999 when history is insufficient.

        With ``momentum_gap`` > 0 this measures the return from ``momentum_period``
        bars ago up to ``momentum_gap`` bars ago (skip-month momentum, e.g. the
        classic 12-1 = gap 21, period 252), skipping the most recent month to
        avoid the short-term reversal effect.
        """
        return self._window_return(data, self.params.momentum_period, self.params.momentum_gap)

    def _window_return(self, data, window: int, gap: int = 0) -> float:
        """Return from ``window`` bars ago to ``gap`` bars ago; -9999 if too short."""
        if len(data) <= window + gap:
            return -9999.0
        old = data.close[-window]
        if old == 0:
            return -9999.0
        recent = data.close[-gap] if gap > 0 else data.close[0]
        return recent / old - 1.0

    def _vol_scalar(self) -> float:
        """Scale factor = min(1, target_vol / realized_portfolio_vol). Returns 1.0 when inactive or insufficient history."""
        if not self.params.vol_targeting:
            return 1.0
        n = self.params.vol_lookback
        if len(self._daily_values) < n + 2:
            return 1.0
        vals = np.array(self._daily_values[-(n + 1):])
        rets = vals[1:] / vals[:-1] - 1.0
        realized = float(np.std(rets, ddof=1) * math.sqrt(252))
        if realized <= 0:
            return 1.0
        return min(1.0, self.params.vol_target / realized)

    def _rolling_vol(self, data) -> float | None:
        """Annualised std dev of daily returns over vol_window days."""
        p = self.params.vol_window
        if len(data) <= p + 1:
            return None
        prices = np.array([data.close[-i] for i in range(p + 1)])
        returns = prices[:-1] / prices[1:] - 1.0
        return float(np.std(returns, ddof=1) * math.sqrt(252))

    def _rank_score(self, candidates: list[tuple]) -> dict:
        """Map each candidate's data feed to a ranking score (higher = better).

        Default: raw ranking return. With ``blended_momentum`` the score is the
        average cross-sectional percentile across ``blend_windows`` lookbacks, so
        a stock strong across 3/6/12 months ranks above one strong in just one.
        """
        if self.params.ic_weighting:
            return self._ic_score(candidates)

        if self.params.accel_tilt > 0:
            return self._accel_score(candidates)

        if not self.params.blended_momentum:
            return {d: mom for d, mom, _ in candidates}

        feeds = [d for d, _, _ in candidates]
        n = len(feeds)
        pct_sum = {d: 0.0 for d in feeds}
        used = 0
        for w in self.params.blend_windows:
            rets = [(d, self._window_return(d, w)) for d in feeds]
            valid = [(d, r) for d, r in rets if r != -9999.0]
            if len(valid) < 2:
                continue
            used += 1
            ordered = sorted(valid, key=lambda x: x[1])  # worst → best
            for rank, (d, _) in enumerate(ordered):
                pct_sum[d] += rank / (len(ordered) - 1)
        if used == 0:
            return {d: mom for d, mom, _ in candidates}
        return {d: pct_sum[d] / used for d in feeds}

    # ------------------------------------------------------------------
    # Pillar 4 — diversifying sleeve
    # ------------------------------------------------------------------

    def _effective_sleeve_pct(self) -> float:
        """Return the active sleeve fraction: sleeve_pct normally, sleeve_pct_stress when stressed.

        Stress triggers (either fires):
          • VIX close > vix_threshold on the current bar's date (uses _VIX_LOOKUP).
          • SPY 21-bar (≈1-month) return < spy_monthly_dd.
        Both checks disabled when dynamic_sleeve=False (returns params.sleeve_pct directly).
        """
        if not self.params.dynamic_sleeve:
            return self.params.sleeve_pct

        stress = False

        # VIX trigger
        if self.params.vix_threshold > 0 and _VIX_LOOKUP:
            dt = self.datetime.date()
            vix_val = _VIX_LOOKUP.get(dt)
            if vix_val is None:
                # Walk back up to 5 calendar days for weekends/holidays
                for lag in range(1, 6):
                    vix_val = _VIX_LOOKUP.get(dt - datetime.timedelta(days=lag))
                    if vix_val is not None:
                        break
            if vix_val is not None and vix_val > self.params.vix_threshold:
                stress = True

        # SPY 1-month return trigger
        if not stress and self.params.spy_monthly_dd < 0:
            lookback = 21
            if len(self.spy) > lookback and float(self.spy.close[-lookback]) > 0:
                spy_ret = float(self.spy.close[0]) / float(self.spy.close[-lookback]) - 1.0
                if spy_ret < self.params.spy_monthly_dd:
                    stress = True

        return self.params.sleeve_pct_stress if stress else self.params.sleeve_pct

    def _sleeve_weights(self) -> dict:
        """Permanent defensive sleeve: always-on allocation to sleeve_assets.

        Returns a dict of feed→weight (fraction of total portfolio). The sleeve
        budget is sleeve_pct of the portfolio, split across sleeve assets by
        equal weight or inverse-vol weighting. Returns empty dict when sleeve
        is off (sleeve_pct=0 or no assets configured).
        """
        if self.params.sleeve_pct <= 0.0 or not self._sleeve_feeds:
            return {}

        budget = self._effective_sleeve_pct()
        feeds = self._sleeve_feeds

        if self.params.sleeve_weighting == "inv_vol" and len(feeds) > 1:
            vols = {}
            for d in feeds:
                v = self._rolling_vol(d)
                if v is not None and v > 0:
                    vols[d] = v
            if len(vols) == len(feeds):
                total_inv = sum(1.0 / v for v in vols.values())
                return {d: (1.0 / vols[d]) / total_inv * budget for d in feeds}

        # equal weight (also fallback when inv_vol has insufficient history)
        per_asset = budget / len(feeds)
        return {d: per_asset for d in feeds}

    # ------------------------------------------------------------------
    # Pillar 2.4 — adaptive smoothing
    # ------------------------------------------------------------------

    def _cs_noise_alpha(self) -> float:
        """Compute the adaptive α for EWMA weight smoothing.

        Reads cross-sectional std of 21-day returns across the full universe,
        appends it to a rolling history, and returns an α that is tight
        (smooth_alpha_min) when noise is at its historical high and loose
        (smooth_alpha_max) when noise is at its historical low.

        Crash override: SPY below MA200 → always return smooth_alpha_min,
        independent of the noise estimate, to avoid whipsawing during
        drawdowns when inv-vol estimates are least reliable.
        """
        if float(self.spy.close[0]) < float(self.spy_ma200[0]):
            return self.params.smooth_alpha_min

        rets_21 = []
        for d in self.stocks:
            if len(d) > 21 and d.close[-21] != 0:
                r = d.close[0] / d.close[-21] - 1.0
                if math.isfinite(r):
                    rets_21.append(r)

        if len(rets_21) < 5:
            return self.params.smooth_alpha_max

        cs_std = float(np.std(rets_21, ddof=1))
        self._cs_vol_history.append(cs_std)
        if len(self._cs_vol_history) > self.params.cs_noise_window:
            self._cs_vol_history = self._cs_vol_history[-self.params.cs_noise_window:]

        hist = self._cs_vol_history
        if len(hist) < 10:
            return self.params.smooth_alpha_max

        # Percentile rank of current noise within history (excluding current obs
        # so the rank is always computed on strictly prior data).
        noise_pct = sum(v < cs_std for v in hist[:-1]) / (len(hist) - 1)
        lo, hi = self.params.smooth_alpha_min, self.params.smooth_alpha_max
        return hi - (hi - lo) * noise_pct

    # ------------------------------------------------------------------
    # Pillar 1 — adaptive signal scoring (single-book path only)
    # ------------------------------------------------------------------

    def _accel_score(self, candidates: list[tuple]) -> dict:
        """Rank score = (1-λ)·z(12m return) + λ·z(acceleration).

        Acceleration is the second-derivative of price momentum: the recent
        half's return minus the distant half's return,

            accel = r(t-126, t) − r(t-252, t-126),

        cross-sectionally z-scored against the candidate set. A stock with a
        big 12-month number that is *decelerating* (recent half weaker than the
        distant half) is tilted down before the fixed 252-day window
        mechanically rolls it off — the 2021-rotation failure mode. λ is kept
        small so this is a tilt on the proven 12-month signal, not a switch to a
        faster strategy. Names without 252+126 bars of history sink to the
        bottom (-9999), matching the raw-ranking convention.
        """
        feeds = [d for d, _, _ in candidates]
        mom = {d: m for d, m, _ in candidates}  # 12m ranking return, already computed

        accel: dict = {}
        for d in feeds:
            recent = self._window_return(d, 126)            # close[0] / close[-126] - 1
            older = self._window_return(d, 252, gap=126)    # close[-126] / close[-252] - 1
            accel[d] = None if (recent == -9999.0 or older == -9999.0) else recent - older

        valid = [d for d in feeds if accel[d] is not None]
        z_mom = cross_sectional_zscore({d: mom[d] for d in valid})
        z_acc = cross_sectional_zscore({d: accel[d] for d in valid})

        lam = self.params.accel_tilt
        return {
            d: ((1.0 - lam) * z_mom[d] + lam * z_acc[d]) if d in z_mom else -9999.0
            for d in feeds
        }

    def _ic_score(self, candidates: list[tuple]) -> dict:
        """Rank score = Σ_h w_h · z(return_h), with w_h set by rolling IC.

        Each rebalance: (1) compute the cross-sectional z-score of every
        horizon's return on the *current* candidates; (2) using the snapshot
        stored at the *previous* rebalance, score how well each horizon's past
        z-score predicted the return that has since been realised (Spearman IC),
        and fold that into the per-horizon EWMA; (3) weight the current
        horizons by a softmax over positive-clamped smoothed ICs and blend.

        Look-ahead discipline: the IC update consumes only the prior snapshot
        and the now-realised return (both ≤ t); the resulting weights are then
        applied to the *forward* month's ranking. The realised return is never
        used in the same rebalance that trades on it. The first rebalance has no
        prior snapshot, so weights fall back to uniform (equal-weight blend).
        """
        feeds = [d for d, _, _ in candidates]
        horizons = self.params.ic_horizons

        # (1) current per-horizon cross-sectional z-scores
        cur_z: dict[int, dict] = {}
        for h in horizons:
            rets = {d: self._window_return(d, h) for d in feeds}
            rets = {d: r for d, r in rets.items() if r != -9999.0}
            cur_z[h] = cross_sectional_zscore(rets)
        cur_close = {d._name: float(d.close[0]) for d in feeds}

        # (2) IC update from the previous snapshot vs the return realised since
        if self._ic_prior_z is not None and self._ic_prior_close:
            realized = {
                name: cur_close[name] / c0 - 1.0
                for name, c0 in self._ic_prior_close.items()
                if name in cur_close and c0 > 0
            }
            ic_by_h: dict[int, float] = {}
            for h in horizons:
                prior = self._ic_prior_z.get(h, {})
                common = [n for n in prior if n in realized]
                if len(common) >= 5:  # need a minimal cross-section for a usable IC
                    ic_by_h[h] = spearman_corr(
                        [prior[n] for n in common], [realized[n] for n in common]
                    )
            if ic_by_h:
                self._ic_tracker.update(ic_by_h)

        # snapshot current state (z by *name*, so it survives feed re-binding)
        self._ic_prior_z = {h: {d._name: z for d, z in cur_z[h].items()} for h in horizons}
        self._ic_prior_close = cur_close

        # (3) blend current z-scores by IC-derived horizon weights
        w = self._ic_tracker.weights(self.params.ic_tau)
        score: dict = {}
        for d in feeds:
            num = 0.0
            den = 0.0
            for h in horizons:
                zv = cur_z[h].get(d)
                if zv is not None:
                    num += w[h] * zv
                    den += w[h]
            score[d] = num if den > 0 else -9999.0
        return score

    def _basket_returns(self, selected: list[tuple], window: int) -> np.ndarray | None:
        """``window``×k daily-return matrix for the selected names (columns in
        ``selected`` order). Returns None if any name lacks enough clean history."""
        feeds = [d for d, _, _ in selected]
        cols = []
        for d in feeds:
            if len(d) <= window + 1:
                return None
            prices = np.array([d.close[-i] for i in range(window + 1)])
            if np.any(prices <= 0):
                return None
            cols.append(prices[:-1] / prices[1:] - 1.0)
        return np.column_stack(cols)

    def _basket_fractions(self, selected: list[tuple]) -> dict:
        """Fractional weights (sum to 1) for the selected names.

        ``inv_vol`` keeps the original behaviour. ``erc`` builds a (optionally
        Ledoit-Wolf shrunk) covariance over ``cov_window`` and solves for
        equal-risk-contribution weights, then blends toward momentum-rank
        conviction by ``mom_tilt`` (λ). Falls back to inverse-vol if the
        covariance/ERC solve can't be trusted, so a rebalance never fails.
        """
        if self.params.weighting != "erc":
            total_inv = sum(1.0 / vol for _, _, vol in selected)
            return {d: (1.0 / vol) / total_inv for d, _, vol in selected}

        returns = self._basket_returns(selected, self.params.cov_window)
        if returns is None or returns.shape[1] != len(selected):
            total_inv = sum(1.0 / vol for _, _, vol in selected)
            return {d: (1.0 / vol) / total_inv for d, _, vol in selected}

        if self.params.cov_shrinkage == "ledoit_wolf":
            cov = ledoit_wolf_constant_corr(returns)
        else:
            cov = sample_covariance(returns)

        w = erc_weights(cov)
        if w is None:
            w = inv_vol_weights(cov)

        lam = self.params.mom_tilt
        if lam > 0:
            # ``selected`` is already sorted best→worst momentum.
            w = (1.0 - lam) * w + lam * momentum_rank_weights(len(selected))
            w = w / w.sum()

        return {d: float(w[i]) for i, (d, _, _) in enumerate(selected)}

    def _cap_weights(self, raw: dict, investable: float, max_w: float) -> dict:
        """Cap each weight at ``max_w`` (water-filling), redistributing excess to
        uncapped names by their inverse-vol share until the budget is exhausted."""
        if max_w >= 1.0 or not raw:
            return raw
        weights = dict(raw)
        capped: set = set()
        for _ in range(len(weights)):
            over = [d for d, w in weights.items() if w > max_w + 1e-9 and d not in capped]
            if not over:
                break
            for d in over:
                capped.add(d)
                weights[d] = max_w
            used = sum(weights[d] for d in capped)
            remaining = investable - used
            free = [d for d in weights if d not in capped]
            base = sum(raw[d] for d in free)
            if remaining <= 0 or base <= 0 or not free:
                break
            for d in free:
                weights[d] = raw[d] / base * remaining
        return weights

    def _apply_sector_cap(self, weights: dict, sector_cap: float) -> dict:
        """Scale down any GICS sector whose aggregate weight exceeds sector_cap.

        Freed weight stays as cash — no rebalancing into next-best names, so
        the cap's effect on risk/return is measured cleanly without confounding
        with a different selection algorithm.
        """
        sector_totals: dict[str, float] = {}
        stock_set = set(self.stocks)
        for d, w in weights.items():
            if w <= 0 or d not in stock_set:
                continue
            sector = SECTOR_MAP.get(d._name, "Unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + w

        over = {s: t for s, t in sector_totals.items() if t > sector_cap + 1e-9}
        if not over:
            return weights

        new_weights = dict(weights)
        for d, w in weights.items():
            if w <= 0 or d not in stock_set:
                continue
            sector = SECTOR_MAP.get(d._name, "Unknown")
            if sector in over:
                new_weights[d] = w * (sector_cap / over[sector])
        return new_weights

    def _equity_weights(self) -> dict:
        """Top-N momentum stocks weighted by inverse volatility."""
        candidates: list[tuple] = []
        for d in self.stocks:
            mom = self._mom_return(d)
            if mom == -9999.0:
                continue
            vol = self._rolling_vol(d)
            if vol is None or vol <= 0:
                continue
            candidates.append((d, mom, vol))

        # Recession rotation: restrict candidates to defensive subset when active.
        if (
            self.params.recession_mode
            and self.params.recession_defensives
            and self.params.recession_months
        ):
            dt = self.datetime.date()
            if (dt.year, dt.month) in self.params.recession_months:
                defensive_set = set(self.params.recession_defensives)
                candidates = [(d, mom, vol) for d, mom, vol in candidates if d._name in defensive_set]

        if self.params.abs_momentum_filter:
            p = self.params.abs_momentum_period
            passing = []
            for d, mom, vol in candidates:
                if len(d) <= p or d.close[-p] == 0:
                    continue
                if d.close[0] / d.close[-p] - 1.0 > 0:
                    passing.append((d, mom, vol))
            candidates = passing

        if not candidates:
            return {}

        score = self._rank_score(candidates)
        if self.params.sharpe_ranking:
            candidates.sort(key=lambda x: x[1] / x[2], reverse=True)
        else:
            candidates.sort(key=lambda x: score[x[0]], reverse=True)

        # Selection: naive top_n, or top_n with a holding-hysteresis rank buffer.
        # The buffer retains a currently-held name while it stays in the top
        # keep_n, so a name oscillating around the rank-N cutoff is not traded
        # every month. Book size is unchanged (still top_n) — no broadening.
        # Catastrophic-stop exclusion: names stopped out intra-month are barred
        # from re-entry until the next rebalance period (set cleared in next()).
        ranked_feeds = [d for d, _, _ in candidates if d._name not in self._stopped_this_period]
        if self.params.keep_n > self.params.top_n:
            held = {d for d in self.stocks if self.getposition(d).size != 0}
            chosen = apply_hysteresis(ranked_feeds, held, self.params.top_n, self.params.keep_n)
        else:
            chosen = ranked_feeds[: self.params.top_n]
        tuple_by_feed = {d: triple for triple in candidates for d in (triple[0],)}
        selected = [tuple_by_feed[d] for d in chosen]

        # Proportional scaling: if only 3 of 5 slots qualify, invest 3/5 of the target allocation;
        # the remainder sits as cash rather than concentrating into fewer names.
        # sleeve_pct is reserved for the defensive sleeve and is not available to equities.
        slot_fraction = len(selected) / self.params.top_n if self.params.abs_momentum_filter else 1.0
        equity_budget = max(0.0, 1.0 - self.params.cash_buffer - self._effective_sleeve_pct())
        investable = equity_budget * self._vol_scalar() * slot_fraction

        fractions = self._basket_fractions(selected)

        weights: dict = {d: 0.0 for d in self.datas}
        for d, _, _ in selected:
            weights[d] = fractions[d] * investable

        if self.params.max_weight < 1.0:
            capped = self._cap_weights(
                {d: weights[d] for d, _, _ in selected}, investable, self.params.max_weight
            )
            for d, w in capped.items():
                weights[d] = w

        if self.params.sector_cap < 1.0:
            weights = self._apply_sector_cap(weights, self.params.sector_cap)

        # EWMA weight smoothing: blend the (capped) target book toward last
        # rebalance's weights to damp inverse-vol estimate noise and shrink the
        # size of rebalancing trades. Single-book path only — the tranche path
        # averages its own sub-books and is left untouched.
        # Adaptive mode (Pillar 2.4): α is computed from cross-sectional noise
        # and crash state rather than fixed; replaces weight_smoothing when on.
        _do_smooth = self.params.tranches == 1 and (
            self.params.weight_smoothing < 1.0 or self.params.adaptive_smoothing
        )
        if _do_smooth:
            alpha = (
                self._cs_noise_alpha()
                if self.params.adaptive_smoothing
                else self.params.weight_smoothing
            )
            target = {d: weights[d] for d, _, _ in selected}
            smoothed = smooth_weights(target, self._prev_weights, alpha)
            for d, w in smoothed.items():
                weights[d] = w
            self._prev_weights = dict(smoothed)
        else:
            self._prev_weights = {d: weights[d] for d, _, _ in selected}

        return weights

    def _bond_weights(self) -> dict:
        """Safe-asset allocation scaled by vol scalar; remainder stays in cash."""
        weights = {d: 0.0 for d in self.datas}
        bond_budget = max(0.0, 1.0 - self.params.cash_buffer - self._effective_sleeve_pct())
        weights[self._safe_feed] = bond_budget * self._vol_scalar()
        return weights

    def _decide_weights(self, dt: datetime.date) -> tuple[dict, str]:
        """Regime decision → (target weights over all datas, regime label).

        Pure: no ordering or state mutation, so it can be called once per book
        (single rebalance) or once per tranche (overlapping rebalances). Each
        call reads price data as of the current bar, so a tranche firing on a
        later day naturally sees more recent momentum/vol.
        """
        if self.params.regime_filter:
            # ── Regime filter 1: Dual Momentum ────────────────────────
            spy_mom = self._mom_return(self.spy)
            agg_mom = self._mom_return(self.agg)
            in_equities = spy_mom > agg_mom and spy_mom > 0.0
            # ── Regime filter 2: 200-day MA crash filter ───────────────
            spy_above_ma = float(self.spy.close[0]) > float(self.spy_ma200[0])
            if not spy_above_ma:
                in_equities = False
        else:
            in_equities = True
            spy_mom = agg_mom = 0.0
            spy_above_ma = True

        if in_equities:
            weights = self._equity_weights()
            if not weights:
                bond_w = self._bond_weights()
                for d, w in self._sleeve_weights().items():
                    bond_w[d] = bond_w.get(d, 0.0) + w
                return bond_w, "BONDS(no_candidates)"
            regime = "STOCKS"
            # Partial trend overlay: when SPY is below its MA, retain only
            # `trend_derisk` of equity exposure and rotate the rest into AGG.
            if self.params.trend_overlay and not self.params.regime_filter:
                if float(self.spy.close[0]) < float(self.spy_ma200[0]):
                    keep = self.params.trend_derisk
                    for d in weights:
                        weights[d] *= keep
                    equity_budget = max(0.0, 1.0 - self.params.cash_buffer - self._effective_sleeve_pct())
                    freed = equity_budget * self._vol_scalar() * (1.0 - keep)
                    weights[self._safe_feed] = weights.get(self._safe_feed, 0.0) + freed
                    regime = "STOCKS(trend_derisk)"
            for d, w in self._sleeve_weights().items():
                weights[d] = weights.get(d, 0.0) + w
            return weights, regime

        reason = (
            "below_200MA"
            if (spy_mom > agg_mom and spy_mom > 0.0 and not spy_above_ma)
            else "bonds_preferred"
        )
        weights = self._bond_weights()
        for d, w in self._sleeve_weights().items():
            weights[d] = weights.get(d, 0.0) + w
        return weights, f"BONDS({reason})"

    def _place_orders(self, weights: dict) -> None:
        """Issue order_target_percent for every weight, sells (negative delta) first."""
        total_val = self.broker.getvalue()

        def _delta(item):
            d, tgt = item
            pos = self.getposition(d)
            curr = (pos.size * d.close[0] / total_val) if total_val > 0 else 0.0
            return tgt - curr

        for data, tgt_pct in sorted(weights.items(), key=lambda x: _delta(x)):
            self.order_target_percent(data, tgt_pct)

    def _rebalance(self, dt: datetime.date) -> None:
        weights, regime = self._decide_weights(dt)
        self._regime = regime
        self._place_orders(weights)

    def _rebalance_tranches(self, dt: datetime.date) -> None:
        """Overlapping rebalance tranches.

        N sub-books each rebalance once per month on a staggered day and hold
        1/N of capital; the book is their average. This diversifies away
        rebalance-date timing luck without changing selection or the inverse-vol
        weights — pure variance reduction. Capping per tranche keeps the
        aggregate weight of any name ≤ max_weight (an average of capped weights
        is still capped). Reduces exactly to the baseline when N=1.
        """
        n = self.params.tranches
        period = (dt.year, dt.month)
        dirty = False
        for i in range(n):
            if self._tranche_period[i] != period and dt.day >= self._tranche_day[i]:
                weights, _ = self._decide_weights(dt)
                self._tranche_weights[i] = weights
                self._tranche_period[i] = period
                dirty = True
        if not dirty:
            return

        agg = {d: 0.0 for d in self.datas}
        active = 0
        for i in range(n):
            if not self._tranche_weights[i]:
                continue
            active += 1
            for d, w in self._tranche_weights[i].items():
                agg[d] += w / n
        self._regime = f"TRANCHES({active}/{n})"
        self._place_orders(agg)

    def _check_cat_stops(self) -> None:
        """Daily catastrophic-stop check (runs every bar, not just on rebalance days).

        Exits any held stock whose current close has fallen more than
        cat_stop_pct below its average cost basis (backtrader's position.price).
        The stopped name is added to _stopped_this_period so it is excluded
        from selection at the next rebalance within the same month — preventing
        an immediate re-entry. The set is cleared at each monthly rebalance.
        """
        threshold = self.params.cat_stop_pct
        if threshold <= 0.0:
            return
        for d in self.stocks:
            pos = self.getposition(d)
            if pos.size <= 0:
                continue
            if pos.price <= 0:
                continue
            drop = d.close[0] / pos.price - 1.0
            if drop < -threshold:
                if self.params.printlog:
                    logger.warning(
                        "CAT STOP %s: close=%.2f avg_cost=%.2f drop=%.1f%%",
                        d._name, d.close[0], pos.price, drop * 100,
                    )
                self.order_target_percent(d, 0.0)
                self._stopped_this_period.add(d._name)
                self._cat_stop_count += 1

    # ------------------------------------------------------------------
    # Backtrader hooks
    # ------------------------------------------------------------------

    def next(self):
        self._daily_values.append(self.broker.getvalue())

        if len(self.spy) <= self.params.momentum_period:
            return

        # Catastrophic stop: checked daily, fires between rebalances.
        self._check_cat_stops()

        dt: datetime.date = self.spy.datetime.date(0)

        if self.params.tranches > 1:
            self._rebalance_tranches(dt)
            return

        if self.params.rebalance_freq == "weekly":
            iso = dt.isocalendar()
            period = (iso[0], iso[1])
        else:
            period = (dt.year, dt.month)

        if period == self._last_period:
            return

        # Clear stopped-out set at the start of a new rebalance period so names
        # can be re-selected if momentum still favours them next month.
        self._stopped_this_period.clear()
        self._last_period = period
        self._rebalance(dt)

    def stop(self):
        if self.params.printlog:
            logger.info(
                "Strategy complete | final_value=%.2f | regime=%s",
                self.broker.getvalue(), self._regime,
            )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_feed(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
) -> bt.feeds.PandasData:
    df = get_prices(symbol, start, end, auto_adjust=True)

    feed = bt.feeds.PandasData(dataname=df)
    feed._name = symbol
    return feed


def _benchmark_metrics(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
    initial_cash: float,
) -> dict:
    try:
        df = get_prices(symbol, start, end, auto_adjust=True)
    except ValueError:
        return {}
    if df.empty or len(df) < 2:
        return {}

    prices = df["close"].dropna()
    daily_ret = prices.pct_change().dropna()
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    rf_daily = 0.04 / 252
    excess = daily_ret - rf_daily
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(252))
        if excess.std() > 0 else None
    )
    rolling_max = prices.cummax()
    drawdowns = (prices - rolling_max) / rolling_max
    max_dd = float(abs(drawdowns.min()) * 100)

    return {
        "symbol": f"{symbol} Buy & Hold",
        "total_return_pct": round(float(total_return), 2),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 2),
        "initial_cash": initial_cash,
        "final_value": round(float(initial_cash * (1 + total_return / 100)), 2),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
    initial_cash: float = 100_000.0,
    top_n: int = 5,
    stock_universe: list[str] | None = None,
    vol_window: int = 20,
    rebalance_freq: str = "monthly",
    regime_filter: bool = False,
    sharpe_ranking: bool = False,
    vol_targeting: bool = False,
    vol_target: float = 0.15,
    vol_lookback: int = 21,
    abs_momentum_filter: bool = True,
    abs_momentum_period: int = 126,
    momentum_gap: int = 0,
    blended_momentum: bool = False,
    max_weight: float = 1.0,
    trend_overlay: bool = False,
    trend_derisk: float = 0.5,
    weighting: str = "inv_vol",
    cov_window: int = 63,
    cov_shrinkage: str = "ledoit_wolf",
    mom_tilt: float = 0.0,
    tranches: int = 1,
    keep_n: int = 0,
    weight_smoothing: float = 1.0,
    accel_tilt: float = 0.0,
    ic_weighting: bool = False,
    ic_horizons: tuple[int, ...] = (21, 63, 126, 252),
    ic_beta: float = 0.2,
    ic_tau: float = 0.10,
    adaptive_smoothing: bool = False,
    smooth_alpha_min: float = 0.2,
    smooth_alpha_max: float = 0.8,
    cs_noise_window: int = 60,
    safe_asset: str = "AGG",
    sleeve_pct: float = 0.0,
    sleeve_assets: tuple[str, ...] = (),
    sleeve_weighting: str = "equal",
    sector_cap: float = 1.0,
    cat_stop_pct: float = 0.0,
    dynamic_sleeve: bool = False,
    sleeve_pct_stress: float = 0.20,
    vix_threshold: float = 30.0,
    spy_monthly_dd: float = -0.08,
    recession_mode: bool = False,
    recession_defensives: tuple[str, ...] = (),
) -> dict[str, Any]:
    if start_date is None:
        start_date = datetime.date(2015, 1, 1)
    if end_date is None:
        end_date = datetime.date.today()

    universe = stock_universe if stock_universe is not None else _FALLBACK_UNIVERSE

    # 252 trading days ≈ 380 calendar days of warmup for momentum_period + MA200
    data_start = start_date - datetime.timedelta(days=380)

    logger.info(
        "Running momentum+inv-vol backtest | %s → %s | cash=%.0f | "
        "universe=%d stocks | top_n=%d | vol_window=%d | rebalance=%s",
        start_date, end_date, initial_cash,
        len(universe), top_n, vol_window, rebalance_freq,
    )

    # Pre-fetch VIX closes for dynamic sleeve stress detection.
    global _VIX_LOOKUP
    _VIX_LOOKUP = {}
    if dynamic_sleeve and vix_threshold > 0:
        try:
            vix_df = get_prices("^VIX", data_start, end_date or datetime.date.today())
            _VIX_LOOKUP = {
                (d.date() if hasattr(d, "date") else d): float(row.get("Close", row.get("close", 0.0)))
                for d, row in vix_df.iterrows()
                if not pd.isna(row.get("Close", row.get("close", float("nan"))))
            }
            logger.info("VIX loaded: %d daily closes", len(_VIX_LOOKUP))
        except Exception as exc:
            logger.warning("VIX fetch failed (%s); VIX trigger disabled.", exc)

    # Precompute recession months once so the strategy doesn't repeat the work.
    _recession_months: frozenset[tuple[int, int]] = frozenset()
    if recession_mode and recession_defensives:
        from backtest._recession import build_recession_months
        _signal_start = (start_date or datetime.date(2007, 1, 1)) - datetime.timedelta(days=400)
        _signal_end = end_date or datetime.date.today()
        _recession_months = build_recession_months(_signal_start, _signal_end)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MomentumInvVolStrategy,
        top_n=top_n,
        vol_window=vol_window,
        rebalance_freq=rebalance_freq,
        regime_filter=regime_filter,
        sharpe_ranking=sharpe_ranking,
        vol_targeting=vol_targeting,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        abs_momentum_filter=abs_momentum_filter,
        abs_momentum_period=abs_momentum_period,
        momentum_gap=momentum_gap,
        blended_momentum=blended_momentum,
        max_weight=max_weight,
        trend_overlay=trend_overlay,
        trend_derisk=trend_derisk,
        weighting=weighting,
        cov_window=cov_window,
        cov_shrinkage=cov_shrinkage,
        mom_tilt=mom_tilt,
        tranches=tranches,
        keep_n=keep_n,
        weight_smoothing=weight_smoothing,
        accel_tilt=accel_tilt,
        ic_weighting=ic_weighting,
        ic_horizons=ic_horizons,
        ic_beta=ic_beta,
        ic_tau=ic_tau,
        adaptive_smoothing=adaptive_smoothing,
        smooth_alpha_min=smooth_alpha_min,
        smooth_alpha_max=smooth_alpha_max,
        cs_noise_window=cs_noise_window,
        safe_asset=safe_asset,
        sleeve_pct=sleeve_pct,
        sleeve_assets=tuple(sleeve_assets),
        sleeve_weighting=sleeve_weighting,
        sector_cap=sector_cap,
        cat_stop_pct=cat_stop_pct,
        dynamic_sleeve=dynamic_sleeve,
        sleeve_pct_stress=sleeve_pct_stress,
        vix_threshold=vix_threshold,
        spy_monthly_dd=spy_monthly_dd,
        recession_mode=recession_mode,
        recession_defensives=tuple(recession_defensives),
        recession_months=_recession_months,
        printlog=False,
    )

    # Feed order: SPY first, AGG second, optional safe asset third, sleeve
    # assets next, then the universe. All non-universe tickers are added to
    # _overhead_syms so they are never treated as momentum-ranking candidates.
    _overhead_syms = {"SPY", "AGG", safe_asset} | set(sleeve_assets)
    _feed_prefix = ["SPY", "AGG"]
    if safe_asset not in ("AGG", ""):
        _feed_prefix.append(safe_asset)
    for s in sleeve_assets:
        if s not in _overhead_syms or s not in _feed_prefix:
            if s not in _feed_prefix:
                _feed_prefix.append(s)

    loaded_symbols: list[str] = []
    for sym in _feed_prefix + list(universe):
        try:
            feed = _fetch_feed(sym, data_start, end_date)
            cerebro.adddata(feed, name=sym)
            if sym not in _overhead_syms:
                loaded_symbols.append(sym)
        except Exception as exc:
            logger.warning("Skipping %s — could not fetch data: %s", sym, exc)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="monthly_returns",
        timeframe=bt.TimeFrame.Months,
    )

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    dd = strat.analyzers.drawdown.get_analysis()
    ta = strat.analyzers.trades.get_analysis()

    total_trades = ta.get("total", {}).get("closed", 0)
    won_trades = ta.get("won", {}).get("total", 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

    raw_monthly: dict = strat.analyzers.monthly_returns.get_analysis()
    monthly_returns = []
    for dt_key, ret in raw_monthly.items():
        d = dt_key.date() if hasattr(dt_key, "date") else dt_key
        if start_date <= d <= end_date:
            monthly_returns.append(
                {"year": d.year, "month": d.month, "return_pct": round(ret * 100, 2)}
            )
    monthly_returns.sort(key=lambda x: (x["year"], x["month"]))

    benchmark = _benchmark_metrics("SPY", start_date, end_date, initial_cash)

    return {
        "params": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "initial_cash": initial_cash,
            "top_n": top_n,
            "momentum_period": 252,
            "vol_window": vol_window,
            "rebalance_freq": rebalance_freq,
            "regime_filter": regime_filter,
            "sharpe_ranking": sharpe_ranking,
            "vol_targeting": vol_targeting,
            "vol_target": vol_target,
            "vol_lookback": vol_lookback,
            "abs_momentum_filter": abs_momentum_filter,
            "abs_momentum_period": abs_momentum_period,
            "momentum_gap": momentum_gap,
            "blended_momentum": blended_momentum,
            "max_weight": max_weight,
            "trend_overlay": trend_overlay,
            "trend_derisk": trend_derisk,
            "weighting": weighting,
            "cov_window": cov_window,
            "cov_shrinkage": cov_shrinkage,
            "mom_tilt": mom_tilt,
            "tranches": tranches,
            "keep_n": keep_n,
            "weight_smoothing": weight_smoothing,
            "accel_tilt": accel_tilt,
            "ic_weighting": ic_weighting,
            "ic_horizons": list(ic_horizons),
            "ic_beta": ic_beta,
            "ic_tau": ic_tau,
            "safe_asset": safe_asset,
            "sector_cap": sector_cap,
            "universe_size": len(loaded_symbols),
        },
        "strategy": {
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe_raw, 4) if sharpe_raw is not None else None,
            "max_drawdown_pct": round(dd.get("max", {}).get("drawdown", 0.0), 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "monthly_returns": monthly_returns,
            "cat_stop_triggers": strat._cat_stop_count,
        },
        "benchmark": benchmark,
    }
