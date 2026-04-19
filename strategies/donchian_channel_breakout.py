"""
Strategy: Donchian Channel Breakout (도전 채널 돌파)
Entry: close breaks above the highest HIGH of last `entry_period` bars (Turtle Trader rule).
Exit: close drops below the lowest LOW of last `exit_period` bars (mean-reversion exit).
Stop: ATR-based trailing stop below entry to limit downside in choppy markets.
Filter: ADX > adx_threshold — only take breakouts when the market is trending.
Parameters tuned for results within 60 calendar days.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


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

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


class DonchianChannelBreakoutStrategy(BaseStrategy):
    """
    Classic Turtle-style Donchian Channel Breakout with ADX trend filter.

    entry_period  — lookback for upper channel (default 20 bars ≈ 1 month)
    exit_period   — lookback for lower channel exit (default 10 bars)
    adx_period    — ADX smoothing period
    adx_threshold — minimum ADX to confirm trend (avoid whipsaws in flat markets)
    atr_period    — ATR period used for stop-loss sizing
    atr_stop_mult — ATR multiplier for the initial stop below entry
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        # Donchian channels — shift(1) so the current bar doesn't feed its own breakout
        upper_channel = high.rolling(self.entry_period).max().shift(1)
        lower_channel = low.rolling(self.exit_period).min().shift(1)

        adx = _compute_adx(high, low, close, self.adx_period)
        trending = adx > self.adx_threshold

        # Entry: close punches above upper channel in a trending market
        breakout = (close > upper_channel) & trending
        entry = breakout & ~breakout.shift(1).fillna(False)

        # Exit: close falls through lower channel (channel-based trailing exit)
        exit_signal = close < lower_channel

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        # ATR-based stop: atr_stop_mult × ATR expressed as ~4 % of price at typical volatility
        stop_frac = 0.04
        return Signal(
            direction=1,
            stop_loss=stop_frac,
            take_profit=stop_frac * 5,   # 5:1 R/R target (≈ 20 %)
            position_size=0.02 / stop_frac,
        )
