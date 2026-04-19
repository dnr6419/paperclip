"""
Strategy: CCI Mean Reversion (Commodity Channel Index)
Entry: CCI crosses above -100 (oversold exit) while price > 200-day SMA.
Exit: CCI reaches +100 or stop-loss triggered.
Reference: Lambert (1980), Chong & Ng (2008), Bhatt & Bhatt (2022).
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class CCIMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        cci_period: int = 20,
        oversold: float = -100.0,
        overbought: float = 100.0,
        sma_filter: int = 200,
        stop_loss: float = 0.05,
    ):
        self.cci_period = cci_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_filter = sma_filter
        self.stop_loss = stop_loss

    def _calc_cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(self.cci_period).mean()
        mean_dev = tp.rolling(self.cci_period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
        return cci.fillna(0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        cci = self._calc_cci(high, low, close)

        # Trend filter: price above 200-day SMA
        sma = close.rolling(self.sma_filter).mean()
        trend_up = close > sma

        # Entry: CCI crosses above oversold threshold (-100)
        cross_up = (cci > self.oversold) & (cci.shift(1) <= self.oversold)
        entry = cross_up & trend_up

        # De-duplicate consecutive entry bars
        prev_entry = entry.shift(1).fillna(False).astype(bool)
        entry = entry & ~prev_entry

        # Exit: CCI reaches overbought threshold (+100)
        exit_signal = (cci >= self.overbought) & (cci.shift(1) < self.overbought)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=0.15,
            position_size=0.02 / self.stop_loss,
        )
