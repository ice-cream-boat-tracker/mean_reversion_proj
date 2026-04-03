# Mean Reversion Trading Strategy (Z-Score Based)

## Project Overview
This project implements a mean reversion trading strategy using Z-scores on SPY price data. The goal is to identify when price deviates from its short-term average and trade on the expectation that it will revert.

## Strategy Intuition
We compute a Z-score:

z = (price - moving average) / standard deviation

- If z < -1 → price is below average → BUY
- If z ≥ 0 → price has reverted → SELL

## Parameters
- Symbol: SPY  
- Lookback window: 20 periods  
- Entry threshold: z < -1  
- Exit threshold: z ≥ 0  
- Position size: $300  

## Implementation
- Uses Alpaca API for live market data  
- Computes rolling mean and standard deviation  
- Generates trading signals based on Z-score  
- Executes trades using market orders  
- Includes retry logic and error handling  

## How to Run
1. Set API keys:
   export APCA_API_KEY_ID=your_key  
   export APCA_API_SECRET_KEY=your_secret  

2. Install dependencies:
   pip install alpaca-py pandas  

3. Run:
   python main.py  

## Results (Preliminary)
This strategy is designed to capture short-term mean reversion signals in liquid markets like SPY.

Future work will include:
- Backtesting performance
- Evaluating Sharpe ratio and drawdown
- Parameter tuning and optimization

## Author
Christopher Munroe