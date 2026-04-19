"""
Strategy: Keltner Channel Mean Reversion (KC MR)
Entry: close touches lower KC band + RSI oversold + volume surge.
Exit: close reaches EMA center or upper KC band.
Keltner Channel uses EMA ± ATR_mult * ATR (no Bollinger std).
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class KeltnerChannelMRStrategy(BaseStrategy):
    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        volume_mult: float = 1.2,
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.volume_mult = volume_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # Keltner Channel
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        lower_kc = ema - self.atr_mult * atr
        upper_kc = ema + self.atr_mult * atr

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.fillna(50)

        # Volume confirmation: above rolling average
        vol_avg = volume.rolling(20).mean()
        vol_ok = volume > self.volume_mult * vol_avg

        # Entry: close below lower band + RSI oversold + volume surge
        touch_lower = close < lower_kc
        rsi_ok = rsi < self.rsi_oversold
        entry = touch_lower & rsi_ok & vol_ok

        # De-duplicate consecutive entry bars
        prev_entry = entry.shift(1).fillna(False).astype(bool)
        entry = entry & ~prev_entry

        # Exit: close crosses above EMA center or upper band
        cross_ema = (close >= ema) & (close.shift(1) < ema.shift(1))
        cross_upper = (close >= upper_kc) & (close.shift(1) < upper_kc.shift(1))
        exit_signal = cross_ema | cross_upper

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.15,
            position_size=0.02 / 0.05,
        )
