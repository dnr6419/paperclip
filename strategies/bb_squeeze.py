"""
Strategy 4: Bollinger Band Squeeze Breakout
Detects volatility compression (squeeze) and trades breakout direction.
Entry: close breaks above upper band after squeeze, with volume spike and ADX>20.
Stop-loss: close below middle band (SMA20). Take-profit: +12%~+20%.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


class BBSqueezeStrategy(BaseStrategy):
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 squeeze_threshold: float = 0.5, adx_threshold: float = 25):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_threshold = squeeze_threshold
        self.adx_threshold = adx_threshold

    def _compute_bb(self, close: pd.Series):
        middle = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        width = (upper - lower) / middle * 100
        return upper, middle, lower, width

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: high, low, close, volume.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        upper, middle, lower, width = self._compute_bb(close)
        width_avg = width.rolling(self.bb_period).mean()

        squeeze = width < (width_avg * self.squeeze_threshold)
        adx = compute_adx(high, low, close)
        vol_spike = volume > volume.rolling(20).mean() * 2

        # Squeeze must have occurred in recent N bars
        recent_squeeze = squeeze.rolling(5).max().astype(bool)

        buy = (
            recent_squeeze
            & (close > upper)
            & vol_spike
            & (adx > self.adx_threshold)
        )

        sell = close < middle

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=None,  # dynamic: below middle band
            take_profit=0.12,
            take_profit2=0.20,
            position_size=0.05,
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    middle_band: float) -> int:
        pct = (current_price - entry_price) / entry_price
        if current_price < middle_band:
            return -1
        if pct >= 0.12:
            return -1
        return 0
