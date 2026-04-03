"""
Configuration dataclasses for the quant_trading package.

These dataclasses centralize all configurable parameters related to the
strategy, backtesting environment, risk management and live trading.  By
encapsulating parameters in well‑typed objects we make it easy to
document defaults and pass around settings in a single object rather than
relying on global variables or ad hoc dictionaries.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class StrategyConfig:
    """Parameters specific to the mean reversion strategy.

    Attributes
    ----------
    symbol:
        The ticker symbol to trade. Defaults to "SPY".
    lookback:
        Number of bars to use when computing the rolling mean and
        standard deviation for the z‑score. Must be >= 2.
    entry_z:
        Threshold below which a long position is entered. For a short
        version of the strategy the sign can be flipped.
    exit_z:
        Threshold above which an existing long position is exited.
    use_fractional_shares:
        If True, permits fractional share sizes when sizing positions.
    """

    symbol: str = "SPY"
    lookback: int = 20
    entry_z: float = -1.5
    exit_z: float = -0.25
    use_fractional_shares: bool = True


@dataclass
class RiskConfig:
    """Parameters for risk management and transaction cost assumptions.

    Attributes
    ----------
    risk_per_trade:
        Fraction of total capital to risk on any single trade (e.g. 0.01
        corresponds to 1%).  Professional trading literature often
        advocates risking 1–2% per trade to preserve capital【624816229230078†L38-L45】.
    max_drawdown:
        Maximum allowable drawdown on the total portfolio during the
        backtest. If the running drawdown exceeds this value the
        backtest will liquidate open positions and stop. Expressed as a
        decimal fraction (0.2 = 20%).
    transaction_cost:
        Proportional transaction cost applied on both buys and sells
        (expressed in decimals, e.g. 0.0005 = 5 bps).  This models
        brokerage commissions and exchange fees.
    slippage:
        Slippage as a fraction of the trade price. Real fills rarely
        occur exactly at the quote; this parameter inflates buy prices
        and deflates sell prices to approximate execution slippage.
    stop_loss_pct:
        Hard stop‑loss as a fraction below the entry price. If the
        instrument’s price falls by more than this amount relative to
        the entry price, the position is exited.  Keeping losses
        manageable helps prevent large drawdowns【624816229230078†L38-L45】.
    """

    risk_per_trade: float = 0.01
    max_drawdown: float = 0.2
    transaction_cost: float = 0.0005
    slippage: float = 0.0001
    stop_loss_pct: float = 0.03


@dataclass
class BacktestConfig:
    """Parameters governing the backtesting environment.

    Attributes
    ----------
    initial_cash:
        Starting cash balance for the backtest.
    start_date:
        ISO format (YYYY‑MM‑DD) string for the beginning of the backtest.
    end_date:
        ISO format (YYYY‑MM‑DD) string for the end of the backtest (non‑inclusive).
    risk_free_rate:
        Annualized risk‑free rate used when computing Sharpe ratios.
        In most backtests this can be set to 0 when the holding period
        is short and the risk free rate is negligible, but it is
        configurable for completeness. Note: When computing daily
        returns the risk free rate will be divided by 252 trading days.
    """

    initial_cash: float = 10_000.0
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    risk_free_rate: float = 0.0


@dataclass
class LiveConfig:
    """Configuration parameters for live trading via the Alpaca API.

    Attributes
    ----------
    api_key:
        Your Alpaca API key. Must be provided via environment variables
        or other secure means.
    api_secret:
        Your Alpaca secret key.
    base_url:
        Base URL for the Alpaca API. Use the paper trading URL for
        testing and the live URL for production.
    symbol:
        The ticker symbol to trade. Defaults to match the strategy.
    timeframe:
        Timeframe for historical bar data and the streaming loop. A
        `alpaca.data.timeframe.TimeFrame` object may be passed at
        runtime to avoid importing Alpaca at module import time.
    dollar_position:
        Dollar notional allocated to each trade. Note that the risk
        manager still controls the size relative to account equity.
    sleep_seconds:
        Number of seconds to sleep between polling the market in the
        live loop. Must be greater than or equal to the bar duration.
    """

    api_key: Optional[str] = field(default=None)
    api_secret: Optional[str] = field(default=None)
    base_url: str = "https://paper-api.alpaca.markets"
    symbol: str = "SPY"
    timeframe: Any = None
    dollar_position: float = 300.0
    sleep_seconds: int = 60