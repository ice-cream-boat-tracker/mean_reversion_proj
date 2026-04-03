"""
quant_trading package
This package contains a modular implementation of a mean‑reversion trading
strategy together with a reusable backtesting engine, risk management
helpers, metrics utilities, and an execution layer for live trading via
Alpaca's API.  The goal of this package is to provide clean and
maintainable building blocks for quantitative trading projects.

All modules are designed with type hints, clear docstrings and minimal
external state to encourage testability and readability.  See the
README.md at the root of this repository for details on how to use
these components in backtests and live trading.
"""

# Expose a simple API at the package level
from .config import StrategyConfig, BacktestConfig, RiskConfig, LiveConfig  # noqa: F401
from .strategy import MeanReversionStrategy, Signal  # noqa: F401
from .backtest import BacktestEngine, Trade  # noqa: F401
from .risk import FixedFractionalSizer  # noqa: F401
from .metrics import sharpe_ratio, max_drawdown, volatility, win_rate  # noqa: F401