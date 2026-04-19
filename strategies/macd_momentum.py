"""
Strategy 11: MACD Momentum
Entry: MACD line crosses above signal line while MACD histogram is increasing.
Exit: MACD line crosses below signal line.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class MACDMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        golden = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        hist_rising = histogram > histogram.shift(1)
        entry = golden & hist_rising

        dead = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[dead] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.25,
            position_size=0.02 / 0.05,
        )
