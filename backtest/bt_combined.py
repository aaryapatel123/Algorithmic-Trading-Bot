"""Combined 3-strategy momentum system:
  1. Dual Momentum (monthly)  — SPY vs AGG 12-month return decides stocks vs bonds
  2. 200-day MA filter        — SPY below 200MA → move to AGG
  3. Monthly stock selection  — top-N stocks by 12-month momentum, equal-weight
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

STOCK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
ALL_SYMBOLS = ["SPY", "AGG"] + STOCK_UNIVERSE


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class CombinedMomentumStrategy(bt.Strategy):
    """
    Data feed order (enforced by run_backtest):
      index 0  → SPY
      index 1  → AGG
      index 2+ → STOCK_UNIVERSE in order
    """

    params = (
        ("top_n", 3),
        ("ma_period", 200),
        ("momentum_period", 252),   # ~12 months of trading days
        ("printlog", True),
    )

    def __init__(self):
        self.spy = self.datas[0]
        self.agg = self.datas[1]
        self.stocks = list(self.datas[2:])

        # SPY is also a valid stock candidate
        self.stock_candidates = [self.spy] + self.stocks

        self.spy_ma200 = bt.indicators.SimpleMovingAverage(
            self.spy, period=self.params.ma_period
        )

        self._last_month: int = -1
        self._regime: str = "init"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mom_return(self, data) -> float:
        """12-month price return; returns -9999 when not enough history."""
        p = self.params.momentum_period
        if len(data) <= p:
            return -9999.0
        old = data.close[-p]
        if old == 0:
            return -9999.0
        return data.close[0] / old - 1.0

    def _rebalance(self, dt: datetime.date) -> None:
        spy_mom = self._mom_return(self.spy)
        agg_mom = self._mom_return(self.agg)

        # ── Step 1: Dual momentum regime ──────────────────────────────
        in_stocks = spy_mom > agg_mom and spy_mom > 0.0

        # ── Step 2: 200-day MA crash filter ───────────────────────────
        spy_above_ma = float(self.spy.close[0]) > float(self.spy_ma200[0])
        if not spy_above_ma:
            in_stocks = False

        all_feeds = list(self.datas)

        # ── Step 3: Build target allocations ──────────────────────────
        if in_stocks:
            ranked = sorted(
                self.stock_candidates,
                key=lambda d: self._mom_return(d),
                reverse=True,
            )
            top = ranked[: self.params.top_n]
            tgt = 0.95 / self.params.top_n
            targets = {d: (tgt if d in top else 0.0) for d in all_feeds}

            regime = "STOCKS"
            if self.params.printlog:
                names = [d._name for d in top]
                logger.info(
                    "%s | %s | top-%d=%s | spy_mom=%.1f%% agg_mom=%.1f%%"
                    " | spy=%.2f ma200=%.2f",
                    dt, regime, self.params.top_n, names,
                    spy_mom * 100, agg_mom * 100,
                    self.spy.close[0], self.spy_ma200[0],
                )
        else:
            reason = (
                "below_200MA"
                if (spy_mom > agg_mom and spy_mom > 0.0 and not spy_above_ma)
                else "bonds_preferred"
            )
            targets = {d: (0.95 if d is self.agg else 0.0) for d in all_feeds}
            regime = f"BONDS({reason})"
            if self.params.printlog:
                logger.info(
                    "%s | %s | spy_mom=%.1f%% agg_mom=%.1f%%",
                    dt, regime, spy_mom * 100, agg_mom * 100,
                )

        self._regime = regime

        # ── Place orders: sells first (ascending target), then buys ───
        total_val = self.broker.getvalue()
        def _delta(item):
            d, tgt_pct = item
            pos = self.getposition(d)
            curr_pct = (pos.size * d.close[0] / total_val) if total_val > 0 else 0.0
            return tgt_pct - curr_pct  # negative = sell, positive = buy

        for data, tgt_pct in sorted(targets.items(), key=lambda x: _delta(x)):
            self.order_target_percent(data, tgt_pct)

    # ------------------------------------------------------------------
    # Backtrader hooks
    # ------------------------------------------------------------------

    def next(self):
        if len(self.spy) <= self.params.momentum_period:
            return

        dt: datetime.date = self.spy.datetime.date(0)
        if dt.month == self._last_month:
            return

        self._last_month = dt.month
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
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol}")

    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low",
                 "Close": "close", "Volume": "volume"}
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    feed = bt.feeds.PandasData(dataname=df)
    feed._name = symbol
    return feed


def _benchmark_metrics(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
    initial_cash: float,
) -> dict:
    """Compute buy-and-hold metrics for a symbol using raw yfinance data."""
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty or len(df) < 2:
        return {}

    # Handle MultiIndex (happens when downloading single ticker with some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices = df["Close"].dropna()
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
    top_n: int = 3,
) -> dict[str, Any]:
    if start_date is None:
        start_date = datetime.date(2015, 1, 1)
    if end_date is None:
        end_date = datetime.date.today()

    # Download extra history so the strategy has 252 bars before start_date
    data_start = start_date - datetime.timedelta(days=380)

    logger.info(
        "Running combined backtest | %s → %s | cash=%.0f | top_n=%d",
        start_date, end_date, initial_cash, top_n,
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        CombinedMomentumStrategy,
        top_n=top_n,
        printlog=True,
    )

    # Feed order matters: SPY first, AGG second, then stocks
    for sym in ALL_SYMBOLS:
        feed = _fetch_feed(sym, data_start, end_date)
        cerebro.adddata(feed, name=sym)

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

    # Monthly returns — filter to [start_date, end_date]
    raw_monthly: dict = strat.analyzers.monthly_returns.get_analysis()
    monthly_returns = []
    for dt_key, ret in raw_monthly.items():
        if hasattr(dt_key, "date"):
            d = dt_key.date()
        else:
            d = dt_key
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
        },
        "strategy": {
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe_raw, 4) if sharpe_raw is not None else None,
            "max_drawdown_pct": round(
                dd.get("max", {}).get("drawdown", 0.0), 2
            ),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "monthly_returns": monthly_returns,
        },
        "benchmark": benchmark,
    }
