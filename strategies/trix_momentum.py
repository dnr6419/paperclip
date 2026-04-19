"""
Strategy: TRIX Momentum
TRIX = 1-period % change of triple-smoothed EMA.
Filters out short-cycle noise, captures medium-term momentum.

Entry: TRIX crosses above its signal line (9-period SMA) while TRIX > 0
Exit: TRIX crosses below its signal line
Reference: Hutson (1983), Technical Analysis of Stocks & Commodities
"""
import pandas as pd
from .base import BaseStrategy, Signal


class TRIXMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        trix_period: int = 15,
        signal_period: int = 9,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
    ):
        self.trix_period = trix_period
        self.signal_period = signal_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def _trix(self, close: pd.Series) -> pd.Series:
        ema1 = close.ewm(span=self.trix_period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.trix_period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.trix_period, adjust=False).mean()
        return (ema3 - ema3.shift(1)) / ema3.shift(1) * 100

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trix = self._trix(close)
        signal = trix.rolling(self.signal_period).mean()

        prev_trix = trix.shift(1)
        prev_signal = signal.shift(1)

        # Bullish crossover: TRIX crosses above signal from below, and TRIX > 0
        entry = (prev_trix <= prev_signal) & (trix > signal) & (trix > 0)
        # Bearish crossover: TRIX crosses below signal
        exit_sig = (prev_trix >= prev_signal) & (trix < signal)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_sig] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.02 / self.stop_loss,
        )
