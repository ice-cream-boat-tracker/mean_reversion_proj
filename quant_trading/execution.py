"""
Live trading execution utilities for Alpaca.

This module wraps the Alpaca trading and data clients to provide safe
and idempotent access patterns.  It implements retry logic with
exponential backoff and jitter on network failures, prevents
double‑executing trades when the current position cannot be
determined, and exposes a simple polling loop for running the mean
reversion strategy in real time.

The entry point for live trading is `run_live_trading`, which
instantiates the necessary clients using credentials from a
`LiveConfig` and repeatedly fetches market data, computes signals,
manages positions and submits orders.

Note: Live trading carries real financial risk.  This code is for
educational purposes and should be thoroughly tested before being used
with real money.  Use Alpaca paper trading endpoints for testing.
"""

from __future__ import annotations

import logging
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    # alpaca-py may not be installed for users who only backtest
    StockHistoricalDataClient = None  # type: ignore
    StockBarsRequest = None  # type: ignore
    TradingClient = None  # type: ignore
    MarketOrderRequest = None  # type: ignore
    OrderSide = None  # type: ignore
    TimeInForce = None  # type: ignore

from .strategy import MeanReversionStrategy
from .risk import FixedFractionalSizer, compute_stop_price
from .config import LiveConfig, RiskConfig, StrategyConfig


class AlpacaExecutor:
    """Wrapper around Alpaca's trading and data clients with retry logic."""

    def __init__(self, config: LiveConfig) -> None:
        if StockHistoricalDataClient is None or TradingClient is None:
            raise RuntimeError(
                "alpaca-py is not installed. Install alpaca-py to use live trading features."
            )
        if not config.api_key or not config.api_secret:
            raise RuntimeError("API key and secret must be provided for live trading")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = config.symbol
        # Determine paper vs live environment
        self.trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper="paper" in config.base_url.lower(),
            base_url=config.base_url,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.api_secret, base_url=config.base_url
        )
        self.timeframe = config.timeframe

    def _retry(self, fn, *, tries: int = 5, base_sleep: float = 1.0, max_sleep: float = 30.0, label: str = "call"):
        """Generic retry wrapper with exponential backoff and jitter."""
        for i in range(tries):
            try:
                return fn()
            except Exception as e:
                wait = min(max_sleep, base_sleep * (2 ** i)) + random.uniform(0, 0.25)
                self.logger.warning(f"{label} failed ({type(e).__name__}: {e}). retry {i + 1}/{tries} sleeping {wait:.1f}s")
                time.sleep(wait)
        return None

    def get_clock(self):
        return self._retry(lambda: self.trading_client.get_clock(), label="get_clock")

    def get_recent_bars(self, limit: int) -> Optional[pd.DataFrame]:
        def _fn():
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=limit * 2)
            req = StockBarsRequest(
                symbol_or_symbols=[self.symbol], timeframe=self.timeframe, start=start, end=end, limit=limit, feed="iex"
            )
            bars = self.data_client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                raise RuntimeError(f"No bar data returned for {self.symbol}")
            return df.xs(self.symbol).tail(limit)
        return self._retry(_fn, label="get_recent_bars")

    def get_position_qty(self) -> Optional[float]:
        def _fn():
            positions = self.trading_client.get_all_positions()
            for p in positions:
                if p.symbol == self.symbol:
                    return float(p.qty)
            return 0.0
        return self._retry(_fn, label="get_position_qty")

    def place_order(self, side: str, qty: Optional[float] = None, notional: Optional[float] = None) -> None:
        """Submit a market order on Alpaca.

        Parameters
        ----------
        side: str
            'BUY' or 'SELL'.  Will be mapped to the corresponding OrderSide.
        qty: Optional[float]
            Number of shares to trade.  Provide either qty or notional but not both.
        notional: Optional[float]
            Dollar notional to trade.  Alpaca will compute the quantity based on
            the current market price.  Provide either qty or notional but not both.
        """
        if qty is not None and qty <= 0:
            self.logger.warning("Refusing to place order with non‑positive qty")
            return
        if notional is not None and notional <= 0:
            self.logger.warning("Refusing to place order with non‑positive notional")
            return
        if OrderSide is None or MarketOrderRequest is None:
            self.logger.error("alpaca-py is not installed; cannot submit orders")
            return
        order = MarketOrderRequest(
            symbol=self.symbol,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            qty=qty,
            notional=notional,
            time_in_force=TimeInForce.DAY,
        )
        try:
            resp = self.trading_client.submit_order(order)
            self.logger.info(
                f"Submitted {side} order for {self.symbol}. qty={qty} notional={notional} id={resp.id}"
            )
        except Exception as e:
            self.logger.error(f"Error submitting {side} order: {e}")


def run_live_trading(
    strategy: MeanReversionStrategy,
    live_config: LiveConfig,
    risk_config: RiskConfig,
    sizer: Optional[FixedFractionalSizer] = None,
    max_iter: Optional[int] = None,
) -> None:
    """Run the mean reversion strategy in a loop using Alpaca live data.

    This function repeatedly checks whether the market is open, fetches
    recent bars, computes trading signals and manages positions.  It
    sleeps between iterations according to the `sleep_seconds` in the
    live_config.  Setting `max_iter` limits the number of iterations
    (useful for testing); passing None runs indefinitely.
    """
    logger = logging.getLogger("run_live_trading")
    executor = AlpacaExecutor(live_config)
    iteration = 0
    while True:
        if max_iter is not None and iteration >= max_iter:
            logger.info("Reached maximum number of iterations; exiting live loop.")
            break
        iteration += 1
        try:
            clock = executor.get_clock()
            if clock is None:
                time.sleep(live_config.sleep_seconds)
                continue
            if not clock.is_open:
                now = datetime.now(timezone.utc)
                next_open = clock.next_open.replace(tzinfo=timezone.utc) if clock.next_open.tzinfo is None else clock.next_open
                seconds = max(live_config.sleep_seconds, int((next_open - now).total_seconds()))
                seconds = min(seconds, 1800)
                logger.info(f"Market closed. Sleeping {seconds}s until next open (capped at 30m).")
                time.sleep(seconds)
                continue
            bars = executor.get_recent_bars(strategy.lookback + 2)
            if bars is None or bars.empty:
                time.sleep(live_config.sleep_seconds)
                continue
            closes = bars["close"].tail(strategy.lookback + 1)
            closes = closes.iloc[:-1]  # drop potentially incomplete last bar
            closes = closes.tail(strategy.lookback)
            if len(closes) < strategy.lookback:
                logger.info("Not enough data yet; waiting for more bars.")
                time.sleep(live_config.sleep_seconds)
                continue
            current_price = float(closes.iloc[-1])
            history = closes.iloc[:-1].to_list()
            if len(history) < strategy.lookback:
                # Use as much history as available for early bars
                history = ([history[0]] * (strategy.lookback - len(history))) + history
            signal = strategy.generate_signal(history, current_price)
            pos_qty = executor.get_position_qty()
            if pos_qty is None:
                logger.warning("Unable to determine position; skipping trade to avoid duplicates.")
                time.sleep(live_config.sleep_seconds)
                continue
            logger.info(
                f"{datetime.now()} | Price={current_price:.2f} Z={signal.z_score:.2f} Pos={pos_qty} Action={signal.action}"
            )
            # Determine stop price and size for live sizing
            stop_price = compute_stop_price(current_price, risk_config.stop_loss_pct)
            if signal.action == "BUY" and pos_qty == 0:
                if sizer is not None:
                    # Determine current account value (requires equity API)
                    # For simplicity we allocate the configured dollar_position
                    notional = live_config.dollar_position
                    executor.place_order("BUY", notional=notional)
                else:
                    notional = live_config.dollar_position
                    executor.place_order("BUY", notional=notional)
            elif signal.action == "SELL" and pos_qty > 0:
                # Sell the entire existing position
                qty = pos_qty
                executor.place_order("SELL", qty=qty)
        except Exception as e:
            logger.error(f"Error in live trading loop: {type(e).__name__}: {e}")
        time.sleep(live_config.sleep_seconds)