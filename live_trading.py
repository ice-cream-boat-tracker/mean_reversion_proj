import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# === CONFIG ===
SYMBOL = "SPY"
LOOKBACK = 30
ENTRY_Z = -2.0
EXIT_Z = -0.5
POSITION_SIZE = 1000  # dollars
SLEEP_SECONDS = 60


# === API KEYS ===
API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")


# === CLIENTS ===
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)


def market_is_open() -> bool:
    """Return whether the US market is currently open."""
    clock = trading_client.get_clock()
    return bool(clock.is_open)


def get_prices() -> np.ndarray:
    """Fetch recent minute bars and return close prices."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=LOOKBACK + 20,
        feed="iex",
    )

    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        raise ValueError("No bar data returned from Alpaca.")

    # Handle MultiIndex like (symbol, timestamp)
    if hasattr(bars.index, "nlevels") and bars.index.nlevels > 1:
        bars = bars.xs(SYMBOL)

    bars = bars.sort_index().dropna()

    if "close" not in bars.columns:
        raise ValueError(f"'close' column missing. Columns returned: {list(bars.columns)}")

    prices = bars["close"].to_numpy()

    if len(prices) < LOOKBACK + 1:
        raise ValueError(f"Not enough price data returned: got {len(prices)} rows")

    return prices[-(LOOKBACK + 1):]


def compute_z(prices: np.ndarray) -> float:
    """Compute z-score using the previous LOOKBACK prices vs latest price."""
    window = prices[:-1]
    current = prices[-1]

    mean = np.mean(window)
    std = np.std(window)

    if std == 0:
        return 0.0

    return float((current - mean) / std)


def get_position() -> float:
    """Return current position quantity for SYMBOL, or 0 if flat."""
    try:
        pos = trading_client.get_open_position(SYMBOL)
        return float(pos.qty)
    except Exception:
        return 0.0


def submit_order(side: OrderSide, qty: float) -> None:
    """Submit a market order."""
    qty = round(float(qty), 4)

    if qty <= 0:
        print("Skipping order: quantity <= 0")
        return

    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    trading_client.submit_order(order)


def main() -> None:
    print("Running in PAPER trading mode")

    while True:
        try:
            if not market_is_open():
                print("Market is closed. Sleeping 60 seconds.")
                time.sleep(SLEEP_SECONDS)
                continue

            prices = get_prices()
            z = compute_z(prices)
            position = get_position()

            print(f"Z-score: {z:.2f} | Position: {position}")

            if z < ENTRY_Z and position == 0:
                qty = POSITION_SIZE / prices[-1]
                submit_order(OrderSide.BUY, qty)
                print("BUY")

            elif z >= EXIT_Z and position > 0:
                submit_order(OrderSide.SELL, position)
                print("SELL")

        except Exception as e:
            print("Error:", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()