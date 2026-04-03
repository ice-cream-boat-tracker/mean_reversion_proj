"""
Trading strategy implementations.

This module defines data structures and classes for implementing trading
strategies.  Strategies are designed to be stateful so that they can
maintain internal history (e.g. rolling windows) while processing
streaming market data.  Each strategy exposes a single method,
`generate_signal`, which consumes a sequence of prices and returns a
`Signal` indicating what action to take at the current bar.

Currently included is a simple mean‑reversion strategy based on the
z‑score of the most recent price relative to a moving average.  The
strategy goes long when the price is sufficiently below its mean and
exits when the price mean reverts.  See the accompanying README for a
discussion of the logic and assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """A trading signal returned by a strategy.

    Attributes
    ----------
    action:
        One of "BUY", "SELL" or "HOLD".  These correspond to entering a
        long position, exiting an existing long position, or doing
        nothing, respectively.
    confidence:
        A float measuring how strong the signal is.  For example, in
        mean reversion a larger absolute z‑score may correspond to a
        higher confidence.  The backtest engine may use this value to
        scale position sizes.  Confidence values should be signed: a
        positive confidence indicates long bias while a negative
        confidence could indicate a short bias in an extended strategy.
    z_score:
        The raw z‑score computed by the strategy.  Exposed for
        diagnostic purposes.
    """

    action: str
    confidence: float
    z_score: float


class MeanReversionStrategy:
    """Simple mean reversion strategy using z‑scores.

    The strategy computes a rolling mean and standard deviation over a
    lookback window.  When the latest price is sufficiently below the
    mean (z < entry_z) the strategy signals a long entry.  When the
    price has reverted to the mean (z >= exit_z) the strategy signals
    an exit.  Additional fields such as the computed z‑score and
    confidence are returned for downstream consumers.

    Parameters
    ----------
    lookback: int
        Length of the rolling window in bars.  Must be >= 2.
    entry_z: float
        Threshold below which to enter a long position.  Typical values
        are -1 to -2.  More negative values indicate the price is
        further below its mean.
    exit_z: float
        Threshold above which to exit the long position.  This value is
        typically greater than or equal to zero, indicating the price
        has reverted back to its mean.
    """

    def __init__(self, lookback: int, entry_z: float, exit_z: float) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signal(self, history: Sequence[float], current_price: float) -> Signal:
        """Compute the trading signal based on historical prices and the latest price.

        Parameters
        ----------
        history: Sequence[float]
            Sequence of past prices used to compute the rolling mean and
            standard deviation.  The length of `history` must equal
            `self.lookback`.  It must not include the current price to
            avoid lookahead bias.
        current_price: float
            The price at which a trade would be executed if the signal
            triggers.  This value is not included when computing the
            mean and standard deviation of the history.

        Returns
        -------
        Signal
            A signal indicating whether to buy, sell or hold, along
            with a confidence measure and the computed z‑score.
        """
        if len(history) != self.lookback:
            raise ValueError(f"history length {len(history)} != lookback {self.lookback}")

        window = np.asarray(history, dtype=float)

        mean = window.mean()
        std = window.std(ddof=0)
        if std == 0:
            return Signal(action="HOLD", confidence=0.0, z_score=0.0)

        z = (float(current_price) - mean) / std
        confidence = abs(z)

        if z < self.entry_z:
            return Signal(action="BUY", confidence=confidence, z_score=z)
        elif z >= self.exit_z:
            return Signal(action="SELL", confidence=confidence, z_score=z)
        else:
            return Signal(action="HOLD", confidence=confidence, z_score=z)