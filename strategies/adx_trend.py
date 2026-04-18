"""
Strategy 6: ADX Trend Following Momentum
Enter long when ADX > 25, +DI crosses above -DI, and price > SMA(50).
Stop-loss: -5% or below SMA(50). Take-profit: +15%.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_adx_full(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14):
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
    return adx, plus_di, minus_di


class ADXTrendStrategy(BaseStrategy):
    def __init__(self, adx_period: int = 14, adx_threshold: float = 25,
                 sma_period: int = 50, vol_threshold: float = 1.5):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sma_period = sma_period
        self.vol_threshold = vol_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: high, low, close, volume.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        adx, plus_di, minus_di = compute_adx_full(high, low, close, self.adx_period)
        sma50 = close.rolling(self.sma_period).mean()
        vol_avg = volume.rolling(20).mean()

        strong_trend = adx > self.adx_threshold
        di_cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        above_sma = close > sma50
        vol_confirm = volume > vol_avg * self.vol_threshold

        buy = strong_trend & di_cross_up & above_sma & vol_confirm

        # Sell: trend weakens or DI reverses
        trend_weak = adx < 20
        di_cross_down = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
        sell = trend_weak | di_cross_down

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell & (signals != 1)] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.15,
            position_size=0.025 / 0.05,
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    peak_price: float, sma50: float) -> int:
        pct = (current_price - entry_price) / entry_price
        if pct <= -0.05:
            return -1
        if current_price < sma50:
            return -1
        # Trailing stop: -7% from peak
        if peak_price > 0 and (current_price - peak_price) / peak_price <= -0.07:
            return -1
        if pct >= 0.15:
            return -1
        return 0
