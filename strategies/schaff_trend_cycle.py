"""
Strategy: Schaff Trend Cycle (STC)
Combines MACD with Stochastic smoothing to produce a cycle oscillator.

Steps:
  1. Compute MACD line (fast EMA - slow EMA)
  2. Apply Stochastic %K and %D to the MACD values over `cycle_period`
  3. Apply a second Stochastic pass (Stoch of Stoch) for STC output

Entry : STC crosses up through buy_threshold (default 25)
Exit  : STC crosses down through sell_threshold (default 75)

Reference: Doug Schaff (1999), "A New Technical Indicator: The STC"
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _stoch_k(series: pd.Series, period: int) -> pd.Series:
    low_min = series.rolling(period).min()
    high_max = series.rolling(period).max()
    denom = high_max - low_min
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(denom == 0, 0.5, (series - low_min) / denom) * 100
    return pd.Series(k, index=series.index)


def compute_stc(
    close: pd.Series,
    fast_period: int = 23,
    slow_period: int = 50,
    cycle_period: int = 10,
    stoch_smoothing: int = 3,
) -> pd.Series:
    macd = _ema(close, fast_period) - _ema(close, slow_period)

    # First stochastic pass on MACD
    stoch1_k = _stoch_k(macd, cycle_period)
    stoch1_d = _ema(stoch1_k, stoch_smoothing)

    # Second stochastic pass (STC)
    stoch2_k = _stoch_k(stoch1_d, cycle_period)
    stc = _ema(stoch2_k, stoch_smoothing)
    return stc


class SchaffTrendCycleStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        stoch_smoothing: int = 3,
        buy_threshold: float = 25.0,
        sell_threshold: float = 75.0,
        stop_loss: float = 0.04,
        take_profit: float = 0.08,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.cycle_period = cycle_period
        self.stoch_smoothing = stoch_smoothing
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        stc = compute_stc(
            close,
            self.fast_period,
            self.slow_period,
            self.cycle_period,
            self.stoch_smoothing,
        )
        prev_stc = stc.shift(1)

        # Buy: STC crosses up through buy_threshold
        entry = (prev_stc <= self.buy_threshold) & (stc > self.buy_threshold)
        # Sell: STC crosses down through sell_threshold
        exit_sig = (prev_stc >= self.sell_threshold) & (stc < self.sell_threshold)

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
