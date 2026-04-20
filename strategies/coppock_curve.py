"""
Strategy: Coppock Curve
Long-term momentum reversal indicator originally designed for monthly equity indices.

Steps:
  1. ROC_14 = Rate of Change over 14 periods
  2. ROC_11 = Rate of Change over 11 periods
  3. Coppock = WMA(ROC_14 + ROC_11, period=10)
     where WMA uses linearly declining weights (10, 9, 8, ..., 1)

Entry : Coppock Curve crosses from negative to positive (0-line crossover upward)
Exit  : Coppock Curve crosses from positive to negative (0-line crossover downward)

Suitable for: long-term trend entries on daily/weekly equity data.
Reference: Edwin "Sedge" Coppock (1962), Barron's
"""
import pandas as pd
from .base import BaseStrategy, Signal


def compute_coppock(
    close: pd.Series,
    roc_long: int = 14,
    roc_short: int = 11,
    wma_period: int = 10,
) -> pd.Series:
    """Return Coppock Curve series."""
    roc14 = close.pct_change(roc_long) * 100
    roc11 = close.pct_change(roc_short) * 100
    combined = roc14 + roc11

    weights = list(range(wma_period, 0, -1))
    total_weight = sum(weights)

    def _wma(x):
        return (x * weights).sum() / total_weight

    coppock = combined.rolling(wma_period).apply(_wma, raw=True)
    return coppock


class CoppockCurveStrategy(BaseStrategy):
    def __init__(
        self,
        roc_long: int = 14,
        roc_short: int = 11,
        wma_period: int = 10,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
    ):
        self.roc_long = roc_long
        self.roc_short = roc_short
        self.wma_period = wma_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        coppock = compute_coppock(
            df["close"], self.roc_long, self.roc_short, self.wma_period
        )
        prev = coppock.shift(1)

        entry = (prev <= 0) & (coppock > 0)
        exit_sig = (prev >= 0) & (coppock < 0)

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
