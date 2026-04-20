"""
Strategy: Aroon Oscillator
Time-based trend detection measuring how recently price made its highest high
and lowest low within a lookback window.

Steps:
  1. Aroon Up   = 100 × (period − periods_since_highest_high) / period
  2. Aroon Down = 100 × (period − periods_since_lowest_low)  / period
  3. Aroon Osc  = Aroon Up − Aroon Down  (range: −100 to +100)

Entry : Aroon Oscillator crosses from negative to positive
        AND Aroon Up > 70 (strong uptrend confirmation)
Exit  : Aroon Oscillator crosses from positive to negative
        AND Aroon Down > 70 (strong downtrend confirmation)

Suitable for: trending equity markets on daily timeframes.
Reference: Tushar Chande (1995)
"""
import pandas as pd
from .base import BaseStrategy, Signal


def compute_aroon(
    high: pd.Series,
    low: pd.Series,
    period: int = 25,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (aroon_up, aroon_down, aroon_oscillator)."""
    def periods_since_high(x):
        return period - x.argmax()

    def periods_since_low(x):
        return period - x.argmin()

    window = period + 1
    since_high = high.rolling(window).apply(periods_since_high, raw=True)
    since_low = low.rolling(window).apply(periods_since_low, raw=True)

    aroon_up = 100 * (period - since_high) / period
    aroon_down = 100 * (period - since_low) / period
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc


class AroonOscillatorStrategy(BaseStrategy):
    def __init__(
        self,
        period: int = 25,
        threshold: float = 70.0,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
    ):
        self.period = period
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        aroon_up, aroon_down, aroon_osc = compute_aroon(
            df["high"], df["low"], self.period
        )
        prev_osc = aroon_osc.shift(1)

        entry = (prev_osc <= 0) & (aroon_osc > 0) & (aroon_up > self.threshold)
        exit_sig = (prev_osc >= 0) & (aroon_osc < 0) & (aroon_down > self.threshold)

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
