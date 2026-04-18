"""
Strategy 3: MACD Bullish Divergence
Detects bullish divergence: price makes lower low while MACD histogram makes higher low.
Entry on MACD line crossing above signal line after divergence.
Stop-loss: -5%, Take-profit: +10%.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_swing_lows(series: pd.Series, window: int = 5) -> pd.Series:
    """Returns boolean mask where local minima occur within ±window bars."""
    is_low = pd.Series(False, index=series.index)
    for i in range(window, len(series) - window):
        segment = series.iloc[i - window: i + window + 1]
        if series.iloc[i] == segment.min():
            is_low.iloc[i] = True
    return is_low


class MACDDivergenceStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 lookback: int = 20):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: close, volume.
        Bullish divergence: lower price low + higher MACD histogram low,
        confirmed by MACD line crossing signal line upward.
        """
        close = df["close"]
        volume = df["volume"]

        macd_line, signal_line, histogram = compute_macd(
            close, self.fast, self.slow, self.signal_period
        )

        price_swing_low = find_swing_lows(close, window=3)
        hist_swing_low = find_swing_lows(histogram, window=3)

        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        vol_increase = volume > volume.rolling(20).mean()

        signals = pd.Series(0, index=df.index)

        for i in range(self.lookback, len(df)):
            if not macd_cross_up.iloc[i]:
                continue

            window = slice(max(0, i - self.lookback), i)
            prev_price_lows = close[window][price_swing_low[window]]
            prev_hist_lows = histogram[window][hist_swing_low[window]]

            if len(prev_price_lows) < 1 or len(prev_hist_lows) < 1:
                continue

            # Bullish divergence: current price area below previous low,
            # but histogram area above previous hist low
            current_price_area = close.iloc[max(0, i - 3): i + 1].min()
            current_hist_area = histogram.iloc[max(0, i - 3): i + 1].min()

            price_lower_low = current_price_area < prev_price_lows.iloc[-1]
            hist_higher_low = current_hist_area > prev_hist_lows.iloc[-1]

            if price_lower_low and hist_higher_low and vol_increase.iloc[i]:
                signals.iloc[i] = 1

        # Sell: MACD histogram turns negative
        sell = (histogram < 0) & (histogram.shift(1) >= 0)
        signals[sell & (signals != 1)] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.10,
            take_profit2=0.20,
            position_size=0.02 / 0.05,
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    current_histogram: float) -> int:
        pct = (current_price - entry_price) / entry_price
        if pct <= -0.05:
            return -1
        if pct >= 0.10:
            return -1
        if current_histogram < 0:
            return -1
        return 0
