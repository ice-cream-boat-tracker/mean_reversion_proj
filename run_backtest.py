"""
Example script for running a backtest of the mean reversion strategy.

This script uses the quant_trading package to download historical data
from Yahoo Finance, configure the strategy and risk parameters, run
the backtest and print a summary of the results.  It can be used as
a starting point for exploring different symbols, parameter values or
additional strategies.

Usage
-----
python run_backtest.py

Note: ensure that the dependencies listed in requirements.txt are
installed prior to running this script.
"""

import pandas as pd

from quant_trading import (
    StrategyConfig,
    RiskConfig,
    BacktestConfig,
    MeanReversionStrategy,
    FixedFractionalSizer,
    BacktestEngine,
)
from quant_trading.data import fetch_yfinance_data


def main() -> None:
    # Configure parameters
    strategy_cfg = StrategyConfig(symbol="SPY", lookback=20, entry_z=-1.5, exit_z=-0.25)
    risk_cfg = RiskConfig(risk_per_trade=0.01, max_drawdown=0.2, transaction_cost=0.0005, slippage=0.0001, stop_loss_pct=0.03)
    backtest_cfg = BacktestConfig(initial_cash=10_000.0, start_date="2022-01-01", end_date="2024-01-01", risk_free_rate=0.0)

    # Download historical prices from Yahoo Finance
    # Attempt to fetch real historical data.  If this fails (e.g. missing
    # yfinance dependency or network), fall back to generating synthetic
    # prices so that the backtest can still run.
    try:
        data = fetch_yfinance_data(
            strategy_cfg.symbol, start=backtest_cfg.start_date, end=backtest_cfg.end_date, interval="1d"
        )
        prices = data["Adj Close"].dropna().squeeze()
        if prices.empty:
            raise ValueError("Empty price series returned from data provider")
    except Exception as e:
        print(f"Warning: could not fetch real data ({type(e).__name__}: {e}). Using synthetic data.")
        import numpy as np
        # Generate a random walk with small drift and volatility
        num_days = 252 * 2  # approx two years of trading days
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, num_days)
        prices_array = 100 * np.cumprod(1 + returns)
        index = pd.date_range(start=backtest_cfg.start_date, periods=num_days, freq="B")
        prices = pd.Series(prices_array, index=index)

    # Instantiate strategy and sizer
    strategy = MeanReversionStrategy(lookback=strategy_cfg.lookback, entry_z=strategy_cfg.entry_z, exit_z=strategy_cfg.exit_z)
    sizer = FixedFractionalSizer(risk_per_trade=risk_cfg.risk_per_trade)

    # Run backtest
    engine = BacktestEngine(strategy=strategy, prices=prices, config=backtest_cfg, risk_config=risk_cfg, sizer=sizer)
    result = engine.run()

    # Print summary
    print("Backtest Results:\n-----------------")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Total portfolio value: ${result.portfolio_values[-1]:.2f}")
    print(f"Number of trades: {len(result.trades)}")

    # Optionally show trades
    # for trade in result.trades:
    #     print(trade)


if __name__ == "__main__":
    main()