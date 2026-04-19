"""
Strategy 7: Bollinger Band Mean Reversion (BB Mean Reversion)
Entry: close drops below lower BB and RSI < rsi_cap (oversold bounce).
Exit: close crosses above SMA (mean reversion target).
Filter: ATR ratio above atr_threshold confirms sufficient volatility.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class BBMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 1.5,
        rsi_period: int = 14,
        rsi_cap: float = 60.0,
        sma_period: int = 50,
        atr_period: int = 14,
        atr_threshold: float = 0.8,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_cap = rsi_cap
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        lower_bb = sma - self.bb_std * std

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.fillna(50)

        sma_long = close.rolling(self.sma_period).mean()

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        atr_ratio = atr / close
        vol_ok = atr_ratio > (self.atr_threshold / 100)

        touch_lower = close < lower_bb
        rsi_ok = rsi < self.rsi_cap
        entry = touch_lower & rsi_ok & vol_ok

        prev_entry = entry.shift(1).fillna(False).astype(bool)
        entry = entry & ~prev_entry

        exit_signal = (close > sma_long) & (close.shift(1) <= sma_long.shift(1))

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.06,
            take_profit=0.18,
            position_size=0.02 / 0.06,
        )
