"""
Strategy: TTM Squeeze Momentum Breakout
Entry: Bollinger Band compresses inside Keltner Channel (squeeze ON),
       then expands (squeeze OFF) with positive momentum value.
Exit: Momentum turns negative, or stop-loss hit.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal


class TTMSqueezeStrategy(BaseStrategy):
    def __init__(
        self,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_period: int = 5,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
    ):
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.mom_period = mom_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def _linreg_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Rolling linear regression value (last point) over `period` bars."""
        def last_linreg(vals):
            if len(vals) < period:
                return np.nan
            x = np.arange(period)
            slope, intercept = np.polyfit(x, vals, 1)
            return slope * (period - 1) + intercept

        return series.rolling(period).apply(last_linreg, raw=True)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # Keltner Channel (ATR-based)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.kc_period).mean()
        kc_mid = close.ewm(span=self.kc_period, adjust=False).mean()
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        # Squeeze: BB inside KC
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Momentum: close minus midpoint of (highest high, lowest low) over mom_period
        highest = high.rolling(self.mom_period).max()
        lowest = low.rolling(self.mom_period).min()
        delta = close - (highest + lowest) / 2
        momentum = self._linreg_slope(delta, self.mom_period)

        # Entry: squeeze just released (prev ON, curr OFF) with positive momentum
        squeeze_release = squeeze.shift(1) & ~squeeze
        entry = squeeze_release & (momentum > 0)

        # Exit: momentum turns negative
        exit_sig = momentum < 0

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_sig] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.02 / self.stop_loss,
        )
