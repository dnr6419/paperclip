"""
Strategy 9: Multi-Timeframe Momentum (MTM)
Entry: weekly (above MA + RSI > 50) AND daily (above MA + RSI > 50).
Exit: daily bearish (below MA OR RSI < 45) OR weekly reversal.
Lookahead bias prevention: weekly signal shifted by 1 bar before reindex.
Parameters: daily_ma=20, weekly_ma=10, rsi_threshold=50.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


class MTMStrategy(BaseStrategy):
    def __init__(
        self,
        daily_ma: int = 20,
        weekly_ma: int = 10,
        rsi_period: int = 14,
        rsi_threshold: float = 50.0,
        rsi_exit: float = 45.0,
    ):
        self.daily_ma = daily_ma
        self.weekly_ma = weekly_ma
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.rsi_exit = rsi_exit

    def _weekly_bull_signal(self, close: pd.Series) -> pd.Series:
        """Return a daily-indexed boolean series for weekly bullish condition.

        Shift weekly indicators by 1 bar before reindexing to daily to prevent
        lookahead bias: the weekly signal is only observable after the week closes.
        """
        weekly_close = close.resample("W-FRI").last()
        w_ma = weekly_close.rolling(self.weekly_ma).mean()
        w_rsi = compute_rsi(weekly_close, self.rsi_period)

        # shift(1) means we only see last week's completed signal
        w_above_ma = (weekly_close > w_ma).shift(1)
        w_rsi_bull = (w_rsi > self.rsi_threshold).shift(1)
        w_bull = (w_above_ma & w_rsi_bull).fillna(False)

        return w_bull.reindex(close.index, method="ffill").fillna(False)

    def _weekly_bear_signal(self, close: pd.Series) -> pd.Series:
        """Return a daily-indexed boolean series for weekly bearish reversal."""
        weekly_close = close.resample("W-FRI").last()
        w_ma = weekly_close.rolling(self.weekly_ma).mean()
        w_rsi = compute_rsi(weekly_close, self.rsi_period)

        w_bear = ((weekly_close < w_ma) | (w_rsi < self.rsi_threshold)).shift(1).fillna(False)

        return w_bear.reindex(close.index, method="ffill").fillna(False)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        d_ma = close.rolling(self.daily_ma).mean()
        d_rsi = compute_rsi(close, self.rsi_period)

        d_above_ma = close > d_ma
        d_rsi_bull = d_rsi > self.rsi_threshold

        weekly_bull = self._weekly_bull_signal(close)
        weekly_bear = self._weekly_bear_signal(close)

        buy = d_above_ma & d_rsi_bull & weekly_bull

        # Don't re-enter on consecutive days of the same condition — edge on first cross
        prev_buy = buy.shift(1).fillna(False)
        entry = buy & ~prev_buy

        # Exit: daily bearish OR weekly reversal
        d_below_ma = close < d_ma
        d_rsi_weak = d_rsi < self.rsi_exit
        sell = d_below_ma | d_rsi_weak | weekly_bear

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.20,
            position_size=0.02 / 0.05,
        )
