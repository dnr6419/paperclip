"""
Strategy: True Strength Index (TSI) Momentum
Double-smoothed momentum oscillator that filters noise while preserving trend direction.

Steps:
  1. Compute raw momentum: m = close - close.shift(1)
  2. Double-smooth momentum: DS_m = EMA(EMA(m, fast), slow)
  3. Double-smooth absolute momentum: DS_abs = EMA(EMA(|m|, fast), slow)
  4. TSI = 100 * DS_m / DS_abs  (range: -100 to +100)
  5. Signal line = EMA(TSI, signal_period)

Entry : TSI crosses up through signal line (bullish momentum confirmation)
Exit  : TSI crosses down through signal line (bearish momentum confirmation)

Suitable for: trending equity markets; works well on daily/weekly timeframes.
Reference: William Blau (1991), "Momentum, Direction, and Divergence"
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def compute_tsi(
    close: pd.Series,
    fast_period: int = 13,
    slow_period: int = 25,
    signal_period: int = 13,
) -> tuple[pd.Series, pd.Series]:
    momentum = close.diff(1)
    abs_momentum = momentum.abs()

    ds_momentum = _ema(_ema(momentum, fast_period), slow_period)
    ds_abs_momentum = _ema(_ema(abs_momentum, fast_period), slow_period)

    with np.errstate(invalid="ignore", divide="ignore"):
        tsi = np.where(ds_abs_momentum == 0, 0.0, 100 * ds_momentum / ds_abs_momentum)
    tsi = pd.Series(tsi, index=close.index)

    signal = _ema(tsi, signal_period)
    return tsi, signal


class TSIMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 13,
        slow_period: int = 25,
        signal_period: int = 13,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        tsi, signal = compute_tsi(
            close,
            self.fast_period,
            self.slow_period,
            self.signal_period,
        )
        prev_tsi = tsi.shift(1)
        prev_signal = signal.shift(1)

        # Buy: TSI crosses above signal line
        entry = (prev_tsi <= prev_signal) & (tsi > signal)
        # Sell: TSI crosses below signal line
        exit_sig = (prev_tsi >= prev_signal) & (tsi < signal)

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
