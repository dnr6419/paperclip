"""
Strategy 5: Candle Pattern + RSI Combined
Detects bullish reversal candle patterns (Hammer, Morning Star, Bullish Engulfing)
combined with RSI(14) < 40 confirmation.
Stop-loss: pattern low - 1%. Take-profit: +6% (partial) / +10%.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


def is_hammer(open_: pd.Series, high: pd.Series, low: pd.Series,
              close: pd.Series) -> pd.Series:
    body = (close - open_).abs()
    lower_shadow = open_.combine(close, min) - low
    upper_shadow = high - open_.combine(close, max)
    return (lower_shadow > body * 2) & (upper_shadow < body * 0.5) & (close > open_)


def is_bullish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    prev_bearish = close.shift(1) < open_.shift(1)
    current_bullish = close > open_
    engulf = (open_ < close.shift(1)) & (close > open_.shift(1))
    return prev_bearish & current_bullish & engulf


def is_morning_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    day1_bearish = close.shift(2) < open_.shift(2)
    day2_small_body = (close.shift(1) - open_.shift(1)).abs() < (
        (high.shift(1) - low.shift(1)) * 0.3
    )
    day3_bullish = close > open_
    gap_down = open_.shift(1) < close.shift(2)
    day3_closes_above_midpoint = close > (open_.shift(2) + close.shift(2)) / 2
    return day1_bearish & day2_small_body & day3_bullish & gap_down & day3_closes_above_midpoint


class CandleRSIStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, rsi_threshold: float = 40,
                 vol_multiplier: float = 1.2):
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.vol_multiplier = vol_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: open, high, low, close, volume.
        """
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        rsi = compute_rsi(close, self.rsi_period)
        vol_avg = volume.rolling(20).mean()
        vol_confirm = volume > vol_avg * self.vol_multiplier

        rsi_weak = rsi < self.rsi_threshold

        hammer = is_hammer(open_, high, low, close)
        engulfing = is_bullish_engulfing(open_, close)
        morning_star = is_morning_star(open_, high, low, close)

        any_pattern = hammer | engulfing | morning_star

        buy = any_pattern & rsi_weak & vol_confirm

        # Sell: next candle closes below pattern low
        pattern_low = low.shift(1)
        sell = (close < pattern_low * 0.99)

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.03,   # pattern low - 1% (approx 3% from entry)
            take_profit=0.06,
            take_profit2=0.10,
            position_size=0.015 / 0.03,
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    pattern_low: float) -> int:
        if current_price < pattern_low * 0.99:
            return -1
        pct = (current_price - entry_price) / entry_price
        if pct >= 0.06:
            return -1
        return 0
