from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timezone

from alpaca.data import StockHistoricalDataClient

from src.broker.alpaca_client import AlpacaBroker
from src.config import load_config
from src.data.market_data import AlpacaDataProvider
from src.models.trade import (
    SignalRecord,
    TradeRecord,
    close_db,
    get_state,
    init_db,
    set_state,
)
from src.risk.position_sizer import PositionSizer
from src.strategy.multi_signal import MultiSignalStrategy
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 60  # check every minute when market is open
_CLOSED_SLEEP_SECONDS = 300  # check every 5 minutes when market is closed


class TradingBot:
    def __init__(self) -> None:
        self._config = load_config()
        setup_logging(self._config.log_level)
        init_db(self._config.db_path)

        self._broker = AlpacaBroker(self._config)
        data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
        )
        self._data = AlpacaDataProvider(data_client)
        self._strategy = MultiSignalStrategy(
            short_period=self._config.short_ma_period,
            long_period=self._config.long_ma_period,
            rsi_period=self._config.rsi_period,
            rsi_overbought=self._config.rsi_overbought,
            rsi_oversold=self._config.rsi_oversold,
            bb_period=self._config.bb_period,
            bb_std_dev=self._config.bb_std_dev,
            macd_fast=self._config.macd_fast,
            macd_slow=self._config.macd_slow,
            macd_signal_period=self._config.macd_signal_period,
            min_confirmations=self._config.min_confirmations,
        )
        self._sizer = PositionSizer(self._config)
        self._running = False

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Trading bot started — symbols=%s", self._config.symbols)
        account = self._broker.get_account()
        logger.info(
            "Account: equity=%.2f cash=%.2f buying_power=%.2f",
            account.equity,
            account.cash,
            account.buying_power,
        )

        while self._running:
            try:
                self._tick()
            except Exception as exc:
                logger.error("Unhandled error in main loop: %s", exc, exc_info=True)
            finally:
                if self._running:
                    time.sleep(_POLL_INTERVAL_SECONDS)

        self._shutdown()

    def _tick(self) -> None:
        if not self._broker.is_market_open():
            clock = self._broker.get_clock()
            logger.info(
                "Market is closed. Next open: %s",
                clock.next_open.isoformat(),
            )
            time.sleep(_CLOSED_SLEEP_SECONDS)
            return

        account = self._broker.get_account()
        positions = self._broker.get_positions()

        for symbol in self._config.symbols:
            try:
                self._process_symbol(symbol, account, positions)
            except Exception as exc:
                logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

    def _process_symbol(self, symbol, account, positions) -> None:
        bars_needed = max(
            self._config.long_ma_period + 1,
            self._config.bb_period,
            self._config.macd_slow + self._config.macd_signal_period,
            self._config.rsi_period + 1,
        ) + 10  # extra buffer
        bars = self._data.get_historical_bars(
            symbol, self._config.bar_timeframe, limit=bars_needed
        )

        signal = self._strategy.compute_signal(symbol, bars)

        # Persist signal
        signal_record = SignalRecord.create(
            symbol=signal.symbol,
            action=signal.action,
            short_ma=signal.short_ma,
            long_ma=signal.long_ma,
            confidence=signal.confidence,
            signal_timestamp=signal.timestamp,
        )

        if signal.action == "HOLD":
            return

        # Avoid duplicate orders: check if we already processed this bar's signal.
        # Use the last bar's index timestamp so restarts don't re-fire the same signal.
        last_bar_ts = str(bars.index[-1]) if not bars.empty else ""
        last_signal_key = f"last_signal_{symbol}"
        last_ts = get_state(last_signal_key)
        current_ts = last_bar_ts
        if last_ts == current_ts:
            logger.debug("Duplicate signal for %s — skipping", symbol)
            return
        set_state(last_signal_key, current_ts)

        current_price = self._data.get_latest_price(symbol)
        side = "buy" if signal.action == "BUY" else "sell"

        order_request = self._sizer.calculate_order(
            symbol=symbol,
            side=side,
            current_price=current_price,
            account=account,
            positions=positions,
        )

        if order_request is None:
            return

        result = self._broker.submit_market_order(
            symbol=order_request.symbol,
            qty=order_request.qty,
            side=order_request.side,
        )

        TradeRecord.create(
            symbol=result.symbol,
            side=result.side,
            qty=result.qty,
            order_id=result.order_id,
            status=result.status,
            signal=signal_record,
            trade_timestamp=result.submitted_at,
        )

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — stopping bot")
        self._running = False

    def _shutdown(self) -> None:
        logger.info("Bot shutting down")
        try:
            account = self._broker.get_account()
            positions = self._broker.get_positions()
            logger.info(
                "Final state: equity=%.2f positions=%d",
                account.equity,
                len(positions),
            )
        except Exception as exc:
            logger.warning("Could not fetch final state: %s", exc)
        close_db()
        logger.info("Shutdown complete")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    TradingBot().run()


if __name__ == "__main__":
    main()
