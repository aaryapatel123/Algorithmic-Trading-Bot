from __future__ import annotations

import logging
import math
import signal
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.broker.alpaca_client import AlpacaBroker
from src.broker.base import PositionInfo
from src.config import load_config
from src.models.trade import (
    SignalRecord,
    TradeRecord,
    close_db,
    get_state,
    init_db,
    set_state,
)
from src.strategy.combined_momentum import (
    CombinedMomentumStrategy,
    MomentumSignal,
    get_universe,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 60       # check every minute when market is open
_CLOSED_SLEEP_SECONDS = 300       # check every 5 minutes when market is closed
_STATE_KEY_REBALANCE = "last_rebalance_yearmonth"

# Sentinel used when a symbol has no current position
_NO_POSITION = PositionInfo(
    symbol="", qty=0, avg_entry_price=0.0, market_value=0.0, unrealized_pl=0.0
)


class TradingBot:
    def __init__(self) -> None:
        self._config = load_config()
        setup_logging(self._config.log_level)
        init_db(self._config.db_path)

        self._broker = AlpacaBroker(self._config)
        self._strategy = CombinedMomentumStrategy(
            top_n=self._config.top_n,
            momentum_days=self._config.momentum_days,
            ma_period=self._config.ma_period,
        )
        self._running = False

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Trading bot started — combined momentum strategy")
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

    # ------------------------------------------------------------------
    # Tick: trigger a rebalance on the first trading day of each month
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if not self._broker.is_market_open():
            clock = self._broker.get_clock()
            logger.info(
                "Market is closed. Next open: %s",
                clock.next_open.isoformat(),
            )
            time.sleep(_CLOSED_SLEEP_SECONDS)
            return

        now = datetime.now(tz=timezone.utc)
        current_yearmonth = now.strftime("%Y-%m")
        last_rebalance = get_state(_STATE_KEY_REBALANCE)

        if last_rebalance == current_yearmonth:
            logger.debug("Already rebalanced for %s — holding", current_yearmonth)
            return

        logger.info(
            "New month detected (%s, last=%s) — triggering rebalance",
            current_yearmonth,
            last_rebalance or "never",
        )

        try:
            custom = self._config.symbols if self._config.symbols else None
            universe = get_universe(custom)
            logger.info("Universe: %d stocks", len(universe))

            mom_signal = self._strategy.compute_signal(universe)
            self._execute_rebalance(mom_signal)

            # Only mark as done after execution completes without raising
            set_state(_STATE_KEY_REBALANCE, current_yearmonth)
            logger.info("Rebalance complete for %s | regime=%s", current_yearmonth, mom_signal.regime)
        except Exception as exc:
            logger.error("Rebalance failed — will retry next tick: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Execute rebalance: sells first, then buys
    # ------------------------------------------------------------------

    def _execute_rebalance(self, mom_signal: MomentumSignal) -> None:
        account = self._broker.get_account()
        positions = self._broker.get_positions()
        portfolio_value = account.portfolio_value

        if portfolio_value <= 0:
            logger.error("Portfolio value is zero or negative — aborting rebalance")
            return

        # Create one signal record for the whole rebalance
        signal_record = SignalRecord.create(
            symbol=mom_signal.regime[:10],
            action="HOLD",
            short_ma=mom_signal.spy_mom,
            long_ma=mom_signal.agg_mom,
            confidence=float(len(mom_signal.targets)),
            signal_timestamp=mom_signal.timestamp,
        )

        pos_by_symbol: Dict[str, PositionInfo] = {p.symbol: p for p in positions}

        # Get current prices for all relevant symbols
        all_symbols = list(set(mom_signal.targets.keys()) | set(pos_by_symbol.keys()))
        prices = self._get_prices(all_symbols)

        # Compute target quantities
        target_qtys: Dict[str, int] = {}
        for sym, weight in mom_signal.targets.items():
            price = prices.get(sym)
            if not price or price <= 0:
                logger.warning("No valid price for %s — skipping", sym)
                continue
            qty = math.floor(portfolio_value * weight / price)
            if qty > 0:
                target_qtys[sym] = qty

        logger.info(
            "Rebalance plan | regime=%s targets=%s",
            mom_signal.regime,
            {k: v for k, v in target_qtys.items()},
        )

        # ── Phase A: SELLS ──────────────────────────────────────────────
        for sym, pos in pos_by_symbol.items():
            target_qty = target_qtys.get(sym, 0)
            sell_qty = pos.qty - target_qty
            if sell_qty <= 0:
                continue
            try:
                result = self._broker.submit_market_order(sym, sell_qty, "sell")
                TradeRecord.create(
                    symbol=result.symbol,
                    side=result.side,
                    qty=result.qty,
                    order_id=result.order_id,
                    status=result.status,
                    signal=signal_record,
                    trade_timestamp=result.submitted_at,
                )
                logger.info("SELL %s x%d | order_id=%s", sym, sell_qty, result.order_id)
            except Exception as exc:
                logger.error("Failed to sell %s x%d: %s", sym, sell_qty, exc, exc_info=True)

        # Brief pause to allow sell orders to settle before buying
        if any(pos.qty > target_qtys.get(sym, 0) for sym, pos in pos_by_symbol.items()):
            logger.info("Waiting 5s for sell orders to settle...")
            time.sleep(5)

        # Refresh account and positions after sells
        account = self._broker.get_account()
        positions = self._broker.get_positions()
        pos_by_symbol = {p.symbol: p for p in positions}

        # ── Phase B: BUYS ───────────────────────────────────────────────
        for sym, qty in target_qtys.items():
            current_qty = pos_by_symbol.get(sym, _NO_POSITION).qty
            buy_qty = qty - current_qty
            if buy_qty <= 0:
                continue
            try:
                result = self._broker.submit_market_order(sym, buy_qty, "buy")
                TradeRecord.create(
                    symbol=result.symbol,
                    side=result.side,
                    qty=result.qty,
                    order_id=result.order_id,
                    status=result.status,
                    signal=signal_record,
                    trade_timestamp=result.submitted_at,
                )
                logger.info("BUY %s x%d | order_id=%s", sym, buy_qty, result.order_id)
            except Exception as exc:
                logger.error("Failed to buy %s x%d: %s", sym, buy_qty, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols.

        Tries Alpaca position data first (no extra API call for held symbols),
        then falls back to yfinance for unpriced symbols.
        """
        prices: Dict[str, float] = {}

        # Use current positions as a free price source for held symbols
        try:
            for pos in self._broker.get_positions():
                if pos.qty > 0 and pos.symbol in symbols:
                    prices[pos.symbol] = pos.market_value / pos.qty
        except Exception as exc:
            logger.warning("Could not fetch positions for pricing: %s", exc)

        # Fetch remaining symbols via yfinance
        unpriced = [s for s in symbols if s not in prices]
        if unpriced:
            try:
                data = yf.download(
                    unpriced,
                    period="2d",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                )
                for sym in unpriced:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            close = data[sym]["Close"].dropna()
                        else:
                            close = data["Close"].dropna()
                        if not close.empty:
                            prices[sym] = float(close.iloc[-1])
                    except (KeyError, TypeError):
                        logger.warning("Could not extract price for %s from yfinance", sym)
            except Exception as exc:
                logger.error("yfinance price fetch failed: %s", exc)

        return prices

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
