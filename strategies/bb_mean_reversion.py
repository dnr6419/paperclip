"""
Strategy 8: Bollinger Band Bounce
Buy when price crosses back above the lower BB(20, 2σ) after being below it
(confirmed mean-reversion bounce), with RSI(14) < 50 (not overbought) and
price above SMA(50) (medium-term uptrend filter).
This avoids catching falling knives — requires confirmed recovery first.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


class BBMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 1.5,
        rsi_period: int = 14,
        rsi_cap: float = 55.0,
        sma_period: int = 50,
        take_profit: float = 0.25,
        stop_loss: float = 0.05,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_cap = rsi_cap
        self.sma_period = sma_period
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        lower_bb = sma - self.bb_std * std

        sma50 = close.rolling(self.sma_period).mean()
        rsi = compute_rsi(close, self.rsi_period)

        # Bounce: price was below lower BB yesterday, now crosses back above
        was_below = (close.shift(1) < lower_bb.shift(1))
        now_above = close >= lower_bb
        bb_bounce = was_below & now_above

        # Medium-term uptrend and not overbought
        uptrend = close > sma50
        not_overbought = rsi < self.rsi_cap

        entry = bb_bounce & uptrend & not_overbought

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.015 / self.stop_loss,
        )
