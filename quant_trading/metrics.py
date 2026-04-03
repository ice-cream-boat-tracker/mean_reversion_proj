"""
Performance metrics for trading strategies.

This module contains functions to compute common performance statistics
including cumulative returns, Sharpe ratio, maximum drawdown, volatility
and win rate.  These metrics allow objective comparison of trading
strategies and help identify risk/return trade‑offs.

Definitions
-----------
* **Sharpe Ratio**: A measure of risk‑adjusted return defined as
  (mean portfolio return – risk‑free rate) divided by the standard
  deviation of returns.  A higher Sharpe ratio indicates the investor
  is receiving more return per unit of risk【814500412065829†L45-L55】.
* **Maximum Drawdown (MDD)**: The largest percentage decline from a
  portfolio's peak value to its subsequent trough over a period of
  time.  It is calculated as (trough – peak) / peak【152169762865991†L342-L355】.
* **Volatility**: The annualized standard deviation of returns.
* **Win Rate**: The fraction of trades that are profitable.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, List


def compute_returns(values: Sequence[float]) -> np.ndarray:
    """Compute simple period‑to‑period returns from a sequence of portfolio values.

    Returns are computed as `(V_t − V_{t-1}) / V_{t-1}`.  The result
    has length one less than the input sequence.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return np.array([])
    return np.diff(arr) / arr[:-1]


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute the annualized Sharpe ratio of a return series.

    Parameters
    ----------
    returns: Sequence[float]
        Sequence of periodic returns (e.g. daily, minute).  Should not
        contain NaNs.
    risk_free_rate: float
        Annual risk‑free rate expressed as a decimal fraction.
    periods_per_year: int
        Number of return periods per year (252 for daily, 12 for monthly).

    Returns
    -------
    float
        The Sharpe ratio.  If the standard deviation of returns is zero
        the function returns NaN.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    sigma = excess.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return np.sqrt(periods_per_year) * excess.mean() / sigma


def max_drawdown(values: Sequence[float]) -> float:
    """Compute the maximum drawdown of a portfolio value series.

    The maximum drawdown is the minimum of (V_t − peak_t) / peak_t
    across the time series【152169762865991†L342-L355】.  A more negative value
    indicates a larger drawdown.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    # compute cumulative maximum
    cum_max = np.maximum.accumulate(arr)
    drawdowns = (arr - cum_max) / cum_max
    return drawdowns.min()


def volatility(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized volatility of a return series.

    Parameters
    ----------
    returns: Sequence[float]
        Sequence of periodic returns.
    periods_per_year: int
        Number of periods in a year.

    Returns
    -------
    float
        Annualized standard deviation.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def win_rate(trade_returns: Sequence[float]) -> float:
    """Compute the fraction of trades that were profitable.

    A trade is considered profitable if its P&L is strictly greater
    than zero.

    Parameters
    ----------
    trade_returns: Sequence[float]
        Sequence of realized trade returns (profit / entry price or
        absolute dollar profit).  The definition of return must be
        consistent across trades.

    Returns
    -------
    float
        Ratio of profitable trades to total trades.  If there are no
        trades returns 0.
    """
    tr = np.asarray(trade_returns, dtype=float)
    if tr.size == 0:
        return 0.0
    winners = (tr > 0).sum()
    return winners / tr.size