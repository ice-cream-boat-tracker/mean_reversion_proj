"""
Data access utilities for historical and real‑time price series.

This module provides thin wrappers around external data providers such as
Yahoo Finance (via `yfinance`) and Alpaca.  The goal is to abstract
data retrieval away from the rest of the strategy so that the backtest
engine and live trading code do not need to worry about provider‑specific
details.  Where appropriate the functions will raise exceptions if
data cannot be retrieved so that calling code can handle failures or
retry appropriately.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional, Any, Union

import pandas as pd
try:
    import yfinance as yf  # type: ignore
except Exception:
    # yfinance may not be installed in all environments.  We lazily import
    # it inside the fetch_yfinance_data function to allow the rest of the
    # package (e.g. backtesting) to be used without this optional
    # dependency.
    yf = None  # type: ignore

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
except Exception:
    # Alpaca may not be installed when only backtesting; importing lazily
    StockHistoricalDataClient = None  # type: ignore
    StockBarsRequest = None  # type: ignore


def fetch_yfinance_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Retrieve historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    symbol: str
        Ticker symbol, e.g. "SPY".
    start: str
        ISO date string (YYYY‑MM‑DD) indicating the start of the data.
    end: str
        ISO date string (YYYY‑MM‑DD) indicating the end of the data (non‑inclusive).
    interval: str, optional
        Bar interval understood by yfinance (e.g. "1d", "1h", "5m").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamps with columns "Open", "High",
        "Low", "Close", "Adj Close", "Volume".  Raises a
        ValueError if no data is returned.
    """
    # Import yfinance lazily to allow this module to be imported even when
    # yfinance is not installed.  If yfinance is missing, raise an
    # informative error so callers can handle it gracefully.
    global yf  # use module‑level variable
    if yf is None:
        raise ImportError("yfinance is not installed. Install it to fetch data from Yahoo Finance.")
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")
    return df


def fetch_alpaca_bars(
    api_key: str,
    api_secret: str,
    symbol: str,
    timeframe: Any,
    limit: int,
    feed: str = "iex",
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch recent bars using Alpaca's historical data API.

    This function wraps the Alpaca `StockHistoricalDataClient` and returns a
    pandas DataFrame indexed by timestamp.  It deliberately keeps
    provider‑specific details (like timezones and multi‑index handling)
    encapsulated.

    Parameters
    ----------
    api_key: str
        Alpaca API key.
    api_secret: str
        Alpaca secret key.
    symbol: str
        Ticker symbol to fetch.
    timeframe: Any
        An instance of `alpaca.data.timeframe.TimeFrame` indicating the
        bar duration (e.g. TimeFrame.Minute).
    limit: int
        Number of bars to return.
    feed: str
        Data feed to use ("iex" for free data; "sip" for full SIP).
    base_url: str, optional
        Alpaca base URL. If provided this will override the default
        environment.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by timestamp containing OHLCV columns.  If
        no data is returned an exception is raised.

    Notes
    -----
    This function does not implement any retry logic.  Callers are
    expected to catch exceptions and retry if desired.  See
    `quant_trading.execution` for a safe wrapper that includes retries.
    """
    if StockHistoricalDataClient is None or StockBarsRequest is None:
        raise RuntimeError("alpaca-py is not installed. Install alpaca-py to use Alpaca data functions.")

    client = StockHistoricalDataClient(api_key, api_secret, base_url=base_url)  # type: ignore
    end_ts = pd.Timestamp.utcnow()
    start_ts = end_ts - pd.Timedelta(minutes=limit * 2)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol], timeframe=timeframe, start=start_ts, end=end_ts, limit=limit, feed=feed
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        raise ValueError(f"No bar data returned for {symbol}")
    # Flatten the multi‑index and keep only this symbol
    df = df.xs(symbol)
    df = df.tail(limit)
    return df