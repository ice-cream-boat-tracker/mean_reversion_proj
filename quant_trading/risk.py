"""
Risk management and position sizing utilities.

Effective risk management is essential for long‑term trading success.  Even
strategies with positive expected value can fail without disciplined
control over trade sizes and drawdowns【624816229230078†L38-L45】.  This module
provides position sizing functions based on the well‑known fixed
fractional method, as well as a simple stop‑loss calculation helper.
Future enhancements could include Kelly criterion sizing or dynamic
volatility scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FixedFractionalSizer:
    """Position sizing based on a fixed fraction of account equity.

    This sizer computes the number of shares to purchase so that the
    maximum dollar loss on the trade (difference between entry and stop
    prices times number of shares) equals a fixed fraction of the
    account value.  Using a small fraction (1–2%) helps preserve
    capital through inevitable losing streaks【624816229230078†L38-L45】.

    Attributes
    ----------
    risk_per_trade: float
        Fraction of account equity to risk on each trade (e.g. 0.01 = 1%).
    """

    risk_per_trade: float = 0.01

    def size_position(self, account_value: float, entry_price: float, stop_price: float) -> float:
        """Compute the share quantity based on account value and stop distance.

        Parameters
        ----------
        account_value: float
            Current total portfolio value (cash + market value of positions).
        entry_price: float
            The expected execution price of the trade.
        stop_price: float
            The price at which the position would be closed to limit losses.

        Returns
        -------
        float
            Number of shares to purchase.  A value of 0 indicates that
            the stop distance is zero or the account value is zero.
        """
        if account_value <= 0:
            return 0.0
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0.0
        risk_amount = account_value * self.risk_per_trade
        # At times the calculated position may be fractional; rounding
        # decisions are delegated to the caller based on whether fractional
        # shares are permitted.
        qty = risk_amount / stop_distance
        return max(qty, 0.0)


def compute_stop_price(entry_price: float, stop_loss_pct: float) -> float:
    """Compute a stop price below the entry price based on a percentage.

    A stop‑loss helps cap the downside of a position.  This helper
    returns a stop price by multiplying the entry price by (1 –
    stop_loss_pct).  For example a 3% stop corresponds to a multiplier
    of 0.97.  Values should be positive and less than 1.

    Parameters
    ----------
    entry_price: float
        Entry price of the trade.
    stop_loss_pct: float
        Fractional stop loss (e.g. 0.03 for 3%).

    Returns
    -------
    float
        The price at which to exit the position for a loss.  If
        stop_loss_pct >= 1 no stop is applied and the entry price is
        returned.
    """
    if stop_loss_pct <= 0 or stop_loss_pct >= 1:
        return entry_price
    return entry_price * (1.0 - stop_loss_pct)