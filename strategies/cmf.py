"""
Strategy: Chaikin Money Flow (CMF)
Volume-weighted accumulation/distribution indicator measuring buying/selling pressure.

Steps:
  1. Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
  2. Money Flow Volume = MFM × volume
  3. CMF = sum(MFV, period) / sum(volume, period)

Entry : CMF crosses above +0.05 (sustained buying pressure)
Exit  : CMF crosses below -0.05 (sustained selling pressure)

Suitable for: confirming breakouts and trend strength on daily equity data.
Reference: Marc Chaikin
"""
import pandas as pd
from .base import BaseStrategy, Signal


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Return Chaikin Money Flow series."""
    hl_range = high - low
    hl_range = hl_range.replace(0, float("nan"))

    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume

    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf


class CMFStrategy(BaseStrategy):
    def __init__(
        self,
        period: int = 20,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
        stop_loss: float = 0.04,
        take_profit: float = 0.10,
    ):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        cmf = compute_cmf(
            df["high"], df["low"], df["close"], df["volume"], self.period
        )
        prev = cmf.shift(1)

        entry = (prev <= self.buy_threshold) & (cmf > self.buy_threshold)
        exit_sig = (prev >= self.sell_threshold) & (cmf < self.sell_threshold)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_sig] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=min(0.02 / self.stop_loss, 1.0),
        )
