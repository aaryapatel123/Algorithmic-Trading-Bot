from __future__ import annotations

import datetime
import logging

import backtrader as bt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MeanReversionBT(bt.Strategy):
    params = (
        ("ma_period", 20),
        ("drop_threshold", 0.03),   # buy when price is this far below MA
        ("rsi_period", 14),
        ("rsi_buy", 35.0),          # RSI must be below this to buy
        ("rsi_sell", 65.0),         # RSI above this triggers sell
        ("trailing_stop", 0.05),    # trailing stop as a fraction
        ("printlog", True),
    )

    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period
        )
        self.rsi = bt.indicators.RSI(
            self.datas[0],
            period=self.params.rsi_period,
            safediv=True,
        )

        self._highest_price: float | None = None

    def next(self):
        close = self.data.close[0]
        ma = self.ma[0]
        rsi = self.rsi[0]

        if not self.position:
            drop_pct = (ma - close) / ma
            if drop_pct >= self.params.drop_threshold and rsi < self.params.rsi_buy:
                size = int(self.broker.getcash() * 0.95 / close)
                if size < 1:
                    return
                self.buy(size=size)
                self._highest_price = close
                if self.params.printlog:
                    logger.info(
                        "BUY  @ %.2f | ma=%.2f drop=%.2f%% rsi=%.2f",
                        close, ma, drop_pct * 100, rsi,
                    )
        else:
            # Update trailing high-water mark
            if close > self._highest_price:
                self._highest_price = close

            trailing_floor = self._highest_price * (1 - self.params.trailing_stop)
            at_or_above_ma = close >= ma
            rsi_overbought = rsi > self.params.rsi_sell
            trailing_hit = close <= trailing_floor

            if at_or_above_ma or rsi_overbought or trailing_hit:
                reason = (
                    "MA reached" if at_or_above_ma
                    else "RSI overbought" if rsi_overbought
                    else "trailing stop"
                )
                self.sell(size=self.position.size)
                self._highest_price = None
                if self.params.printlog:
                    logger.info(
                        "SELL @ %.2f | ma=%.2f rsi=%.2f reason=%s",
                        close, ma, rsi, reason,
                    )

    def stop(self):
        if self.params.printlog:
            logger.info("Strategy complete | final_value=%.2f", self.broker.getvalue())


def _fetch_yfinance_feed(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
) -> bt.feeds.PandasData:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")

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
    initial_cash: float = 100_000.0,
    ma_period: int = 20,
    drop_threshold: float = 0.03,
    rsi_period: int = 14,
    rsi_buy: float = 35.0,
    rsi_sell: float = 65.0,
    trailing_stop: float = 0.05,
) -> dict:
    if start_date is None:
        start_date = datetime.date.today() - datetime.timedelta(days=730)
    if end_date is None:
        end_date = datetime.date.today()

    logger.info(
        "Running mean-reversion backtest: %s %s→%s "
        "(ma=%d drop=%.1f%% rsi_buy=%.0f rsi_sell=%.0f trail=%.1f%% cash=%.0f)",
        symbol, start_date, end_date,
        ma_period, drop_threshold * 100,
        rsi_buy, rsi_sell, trailing_stop * 100,
        initial_cash,
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MeanReversionBT,
        ma_period=ma_period,
        drop_threshold=drop_threshold,
        rsi_period=rsi_period,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
        trailing_stop=trailing_stop,
        printlog=True,
    )

    data_feed = _fetch_yfinance_feed(symbol, start_date, end_date)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

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

    return {
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown_pct": round(drawdown.get("max", {}).get("drawdown", 0), 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "ma_period": ma_period,
        "drop_threshold": drop_threshold,
        "rsi_period": rsi_period,
        "rsi_buy": rsi_buy,
        "rsi_sell": rsi_sell,
        "trailing_stop": trailing_stop,
    }
