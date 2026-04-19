"""
Strategy 7: ATR Volatility Breakout
Buy when price closes above the 20-day high plus a fraction of ATR(14),
confirming volatility-adjusted expansion. Uses ATR-scaled TP/SL.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class ATRBreakoutStrategy(BaseStrategy):
    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        breakout_mult: float = 0.7,
        take_profit: float = 0.20,
        stop_loss: float = 0.04,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.breakout_mult = breakout_mult
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        atr = self._compute_atr(df)

        rolling_high = close.shift(1).rolling(self.lookback).max()
        breakout_level = rolling_high + self.breakout_mult * atr

        buy = close > breakout_level

        prev_buy = buy.shift(1).fillna(False)
        entry_day = buy & ~prev_buy

        signals = pd.Series(0, index=df.index)
        signals[entry_day] = 1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.02 / self.stop_loss,
        )
