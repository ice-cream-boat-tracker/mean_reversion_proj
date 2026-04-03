"""
Reusable backtesting engine for trading strategies.

This module implements a straightforward event‑driven backtester
supporting long‑only mean reversion strategies.  It accepts a strategy
object (e.g. an instance of `MeanReversionStrategy`), price data,
configuration objects for backtesting and risk, and optional position
sizers.  The engine produces a history of portfolio values and a
record of trades, and can compute basic performance metrics via
`quant_trading.metrics`.

The design intentionally avoids lookahead bias: signals are computed
using only historical bars and the current bar’s price is used solely
for execution.  Transaction costs, slippage and stop losses are
incorporated to approximate real‑world trading conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Iterable

import numpy as np
import pandas as pd

from .strategy import MeanReversionStrategy, Signal
from .risk import FixedFractionalSizer, compute_stop_price
from .config import BacktestConfig, RiskConfig, StrategyConfig
from .metrics import compute_returns, sharpe_ratio, max_drawdown, volatility, win_rate


@dataclass
class Trade:
    """Record of a single trade executed during the backtest.

    Attributes
    ----------
    entry_idx: int
        Index into the price series at which the trade was opened.
    exit_idx: int
        Index into the price series at which the trade was closed.
    entry_price: float
        Execution price on entry, including slippage and transaction cost.
    exit_price: float
        Execution price on exit, including slippage and transaction cost.
    size: float
        Number of shares purchased.  Negative values could be used for
        short positions in future extensions.
    pnl: float
        Profit or loss in dollars (exit value minus entry cost).
    return_pct: float
        Percentage return on the trade relative to entry cost.
    """
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    portfolio_values: List[float]
        Time series of portfolio values at each bar.
    trades: List[Trade]
        List of executed trades.
    metrics: dict
        Dictionary of computed performance statistics including
        cumulative return, Sharpe ratio, max drawdown, volatility and
        win rate.
    """
    portfolio_values: List[float]
    trades: List[Trade]
    metrics: dict


class BacktestEngine:
    """Engine for running a strategy on historical data.

    Parameters
    ----------
    strategy: MeanReversionStrategy
        Strategy instance defining the trading logic.
    prices: pd.Series
        Series of prices indexed by datetime.  Must be sorted in
        ascending chronological order and contain no missing values.
    config: BacktestConfig
        Configuration object specifying initial cash and date range.
    risk_config: RiskConfig
        Configuration object specifying transaction cost, slippage,
        stop loss and drawdown limits.
    sizer: Optional[FixedFractionalSizer]
        Position sizer.  If None, the engine will allocate all
        available cash to each trade (subject to fractional share
        limitations determined by the strategy configuration).
    """

    def __init__(
        self,
        strategy: MeanReversionStrategy,
        prices: pd.Series,
        config: BacktestConfig,
        risk_config: RiskConfig,
        sizer: Optional[FixedFractionalSizer] = None,
        allow_fractional: bool = True,
    ) -> None:
        # Validate and store inputs
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if prices.isna().any():
            raise ValueError("price series contains NaNs; fill or drop missing values before backtesting")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("price series must be sorted in ascending order")
        self.strategy = strategy
        self.prices = prices
        self.config = config
        self.risk_config = risk_config
        self.sizer = sizer
        self.allow_fractional = allow_fractional

    def run(self) -> BacktestResult:
        """Execute the backtest and return a result object."""
        lookback = self.strategy.lookback
        prices = self.prices
        n = len(prices)

        cash = self.config.initial_cash
        position = 0.0  # number of shares currently held
        entry_price: Optional[float] = None
        entry_idx: Optional[int] = None
        portfolio_values: List[float] = []
        trades: List[Trade] = []

        peak_value = cash  # track peak for drawdown calculations

        # Iterate starting after enough history has been observed
        for i in range(lookback, n - 1):
            # history window uses past lookback bars excluding the current bar
            history_slice = prices.iloc[i - lookback : i].to_list()
            current_price = float(prices.iloc[i])

            signal: Signal = self.strategy.generate_signal(history_slice, current_price)

            # Determine stop price for sizing and risk management
            stop_price = compute_stop_price(current_price, self.risk_config.stop_loss_pct)

            # Liquidate on max drawdown
            # Compute hypothetical portfolio value at current price
            current_portfolio = cash + position * current_price
            peak_value = max(peak_value, current_portfolio)
            drawdown = (current_portfolio - peak_value) / peak_value if peak_value > 0 else 0.0
            if drawdown < -self.risk_config.max_drawdown and position > 0:
                # Force exit due to drawdown breach
                exit_price = current_price * (1 - self.risk_config.transaction_cost - self.risk_config.slippage)
                pnl = (exit_price - entry_price) * position if entry_price is not None else 0.0
                trades.append(
                    Trade(
                        entry_idx=entry_idx if entry_idx is not None else i,
                        exit_idx=i,
                        entry_price=entry_price if entry_price is not None else current_price,
                        exit_price=exit_price,
                        size=position,
                        pnl=pnl,
                        return_pct=(pnl / (entry_price * position)) if entry_price and position != 0 else 0.0,
                    )
                )
                cash += exit_price * position
                position = 0.0
                entry_price = None
                entry_idx = None
                # Reset peak after closing
                peak_value = cash
                # continue to next iteration without generating new trade
                portfolio_values.append(current_portfolio)
                continue

            # Check for stop loss within open trade
            if position > 0 and current_price <= stop_price:
                # Stop loss triggered
                exit_price = current_price * (1 - self.risk_config.transaction_cost - self.risk_config.slippage)
                pnl = (exit_price - entry_price) * position if entry_price is not None else 0.0
                trades.append(
                    Trade(
                        entry_idx=entry_idx if entry_idx is not None else i,
                        exit_idx=i,
                        entry_price=entry_price if entry_price is not None else current_price,
                        exit_price=exit_price,
                        size=position,
                        pnl=pnl,
                        return_pct=(pnl / (entry_price * position)) if entry_price and position != 0 else 0.0,
                    )
                )
                cash += exit_price * position
                position = 0.0
                entry_price = None
                entry_idx = None
                peak_value = cash
                portfolio_values.append(cash)
                continue

            # Trading logic based on signal
            if signal.action == "BUY" and position == 0:
                # Determine position size either from sizer or allocate all cash
                if self.sizer is not None:
                    desired_qty = self.sizer.size_position(
                        account_value=cash, entry_price=current_price, stop_price=stop_price
                    )
                else:
                    # If no sizer, invest all cash
                    desired_qty = cash / current_price
                if not self.allow_fractional:
                    desired_qty = np.floor(desired_qty)
                # Apply slippage and transaction cost on buy
                exec_price = current_price * (1 + self.risk_config.transaction_cost + self.risk_config.slippage)
                cost = desired_qty * exec_price
                if cost <= cash and desired_qty > 0:
                    cash -= cost
                    position += desired_qty
                    entry_price = exec_price
                    entry_idx = i
            elif signal.action == "SELL" and position > 0:
                # Sell entire position
                exec_price = current_price * (1 - self.risk_config.transaction_cost - self.risk_config.slippage)
                proceeds = position * exec_price
                cash += proceeds
                pnl = (exec_price - entry_price) * position if entry_price is not None else 0.0
                trades.append(
                    Trade(
                        entry_idx=entry_idx if entry_idx is not None else i,
                        exit_idx=i,
                        entry_price=entry_price if entry_price is not None else current_price,
                        exit_price=exec_price,
                        size=position,
                        pnl=pnl,
                        return_pct=(pnl / (entry_price * position)) if entry_price and position != 0 else 0.0,
                    )
                )
                # Reset position
                position = 0.0
                entry_price = None
                entry_idx = None
                peak_value = cash
            # update portfolio value
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)

        # Liquidate any remaining position at the end of the series
        if position > 0:
            final_price = float(prices.iloc[-1])
            exec_price = final_price * (1 - self.risk_config.transaction_cost - self.risk_config.slippage)
            pnl = (exec_price - entry_price) * position if entry_price is not None else 0.0
            trades.append(
                Trade(
                    entry_idx=entry_idx if entry_idx is not None else n - 1,
                    exit_idx=n - 1,
                    entry_price=entry_price if entry_price is not None else final_price,
                    exit_price=exec_price,
                    size=position,
                    pnl=pnl,
                    return_pct=(pnl / (entry_price * position)) if entry_price and position != 0 else 0.0,
                )
            )
            cash += exec_price * position
            position = 0.0
            entry_price = None
            entry_idx = None
            portfolio_value = cash
            portfolio_values.append(portfolio_value)

        # Compute metrics
        returns = compute_returns(portfolio_values)
        trade_returns = [t.return_pct for t in trades]
        metrics = {
            "cumulative_return": (portfolio_values[-1] / self.config.initial_cash) - 1.0 if portfolio_values else 0.0,
            "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=self.config.risk_free_rate),
            "max_drawdown": max_drawdown(portfolio_values),
            "volatility": volatility(returns),
            "win_rate": win_rate(trade_returns),
            "num_trades": len(trades),
        }

        return BacktestResult(portfolio_values=portfolio_values, trades=trades, metrics=metrics)