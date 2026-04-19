"""
Strategy 6: MACD Zero-Cross Trend
Buy when MACD line crosses above zero while price is above the 200-day MA.
Sell when MACD line crosses below zero. Stop-loss and take-profit managed by engine.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class MACDZeroCrossStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, ma_period: int = 200):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.ma_period = ma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow

        trend_ma = close.rolling(self.ma_period).mean()
        above_trend = close > trend_ma

        zero_cross_up = (macd_line > 0) & (macd_line.shift(1) <= 0)
        zero_cross_down = (macd_line < 0) & (macd_line.shift(1) >= 0)

        buy = zero_cross_up & above_trend
        sell = zero_cross_down

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.25,
            position_size=0.02 / 0.05,
        )
