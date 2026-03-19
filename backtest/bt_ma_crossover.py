from __future__ import annotations

import datetime
import logging

import backtrader as bt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MACrossoverBT(bt.Strategy):
    params = (
        ("short_period", 20),
        ("long_period", 50),
        ("printlog", True),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_period
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
                if self.params.printlog:
                    logger.info(
                        "BUY @ %.2f | short_ma=%.4f long_ma=%.4f",
                        self.data.close[0],
                        self.short_ma[0],
                        self.long_ma[0],
                    )
        elif self.crossover < 0:
            self.sell()
            if self.params.printlog:
                logger.info(
                    "SELL @ %.2f | short_ma=%.4f long_ma=%.4f",
                    self.data.close[0],
                    self.short_ma[0],
                    self.long_ma[0],
                )

    def stop(self):
        if self.params.printlog:
            logger.info(
                "Strategy complete | final_value=%.2f",
                self.broker.getvalue(),
            )


def _fetch_yfinance_feed(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
) -> bt.feeds.PandasData:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")

    # Normalize column names for backtrader
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    return bt.feeds.PandasData(dataname=df)


def run_backtest(
    symbol: str = "AAPL",
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    short_period: int = 20,
    long_period: int = 50,
    initial_cash: float = 100_000.0,
) -> dict:
    if start_date is None:
        start_date = (
            datetime.date.today() - datetime.timedelta(days=730)  # ~2 years
        )
    if end_date is None:
        end_date = datetime.date.today()

    logger.info(
        "Running backtest: %s %s→%s (short=%d, long=%d, cash=%.0f)",
        symbol,
        start_date,
        end_date,
        short_period,
        long_period,
        initial_cash,
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MACrossoverBT,
        short_period=short_period,
        long_period=long_period,
        printlog=True,
    )

    data_feed = _fetch_yfinance_feed(symbol, start_date, end_date)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analysis = strat.analyzers.trades.get_analysis()

    total_trades = trade_analysis.get("total", {}).get("closed", 0)
    won_trades = trade_analysis.get("won", {}).get("total", 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "short_period": short_period,
        "long_period": long_period,
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown_pct": round(drawdown.get("max", {}).get("drawdown", 0), 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
    }

    return summary
