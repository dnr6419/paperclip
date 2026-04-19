"""
Strategy 10: Donchian Channel Breakout (DCB)
Entry: close breaks above highest high of last `entry_period` bars (Turtle breakout).
Exit: close drops below lowest low of last `exit_period` bars.
Filter: ADX > adx_threshold confirms trending regime to avoid whipsaws in flat markets.
Parameters: entry_period=20, exit_period=10, adx_period=14, adx_threshold=20.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus = high.diff()
    dm_minus = -low.diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(0)


class DCBStrategy(BaseStrategy):
    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        # Channel bounds — shift(1) so today's bar doesn't feed its own breakout
        channel_high = close.rolling(self.entry_period).max().shift(1)
        channel_low = close.rolling(self.exit_period).min().shift(1)

        adx = _compute_adx(high, low, close, self.adx_period)
        trending = adx > self.adx_threshold

        breakout = (close > channel_high) & trending
        # First bar of breakout only
        prev_breakout = breakout.shift(1).fillna(False)
        entry = breakout & ~prev_breakout

        breakdown = close < channel_low

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[breakdown] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.25,
            position_size=0.02 / 0.05,
        )
