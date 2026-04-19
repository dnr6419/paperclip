"""
Strategy 11: MACD Momentum
Buy when MACD line crosses above signal line with expanding histogram and
price above 200-SMA (trend filter). Sell on MACD cross below signal.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class MACDMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        sma_trend: int = 200,
        vol_multiplier: float = 1.1,
    ):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.sma_trend = sma_trend
        self.vol_multiplier = vol_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal_line

        sma200 = close.rolling(self.sma_trend).mean()
        vol_avg = volume.rolling(20).mean()

        # MACD bullish crossover: MACD crosses above signal
        macd_cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        # Histogram expanding (momentum increasing)
        hist_expanding = histogram > histogram.shift(1)
        # Price above 200-SMA (uptrend regime)
        above_trend = close > sma200
        # Volume confirmation
        vol_confirm = volume > (vol_avg * self.vol_multiplier)

        # MACD bearish crossover: exit signal
        macd_cross_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

        buy = macd_cross_up & hist_expanding & above_trend & vol_confirm
        sell = macd_cross_down

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
