"""
Strategy 8: Volume-Weighted Breakout (VWB)
Entry: price closes above (vol_period-day rolling high + breakout_mult * ATR(atr_period))
       AND volume > vol_multiplier * vol_period-day average volume
Exit:  trailing stop (highest price since entry - atr_mult * ATR) OR hard stop (-stop_pct%)
atr_mult=2.0 governs trailing-stop width; breakout_mult=0.5 governs entry noise filter
(consistent with the ATR Breakout base strategy).
"""
import pandas as pd
from .base import BaseStrategy, Signal


class VWBStrategy(BaseStrategy):
    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        breakout_mult: float = 0.5,
        vol_period: int = 20,
        vol_multiplier: float = 1.5,
        stop_pct: float = 0.05,
    ):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.breakout_mult = breakout_mult
        self.vol_period = vol_period
        self.vol_multiplier = vol_multiplier
        self.stop_pct = stop_pct
        self.last_atr: pd.Series = None

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
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        atr = self._compute_atr(df)
        self.last_atr = atr

        rolling_high = close.shift(1).rolling(self.vol_period).max()
        breakout_level = rolling_high + self.breakout_mult * atr

        vol_ma = volume.shift(1).rolling(self.vol_period).mean()
        vol_condition = volume > self.vol_multiplier * vol_ma

        buy = (close > breakout_level) & vol_condition
        prev_buy = buy.shift(1).fillna(False)
        entry_day = buy & ~prev_buy

        signals = pd.Series(0, index=df.index)
        signals[entry_day] = 1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_pct,
            take_profit=10.0,  # no fixed TP; trailing stop manages exit
            position_size=0.02 / self.stop_pct,
        )
