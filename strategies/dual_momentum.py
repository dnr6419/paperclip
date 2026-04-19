"""
Strategy 12: Dual Momentum (Antonacci)
Combines absolute and relative momentum to enter only when an asset
has both beaten cash (absolute) and outperformed the market proxy
(relative) over the lookback window.

Entry conditions (all must hold on the first qualifying bar):
  1. Absolute momentum: close / close[lookback bars ago] - 1 > abs_threshold
  2. Relative momentum: asset's lookback return > market's lookback return
     (falls back to absolute-only if no market_close is provided)
  3. MA filter: close > SMA(sma_period) — avoids structural downtrends

Exit: absolute momentum turns negative OR close < SMA(sma_period).

Parameters: lookback=252, abs_threshold=0.0, sma_period=200.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class DualMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        lookback: int = 252,
        abs_threshold: float = 0.0,
        sma_period: int = 200,
    ):
        self.lookback = lookback
        self.abs_threshold = abs_threshold
        self.sma_period = sma_period

    def generate_signals(
        self, df: pd.DataFrame, market_close: pd.Series = None
    ) -> pd.Series:
        close = df["close"]

        # Absolute momentum: 12-month return vs. cash proxy (threshold)
        abs_return = close / close.shift(self.lookback) - 1
        abs_bull = abs_return > self.abs_threshold

        # Relative momentum: outperform market benchmark over same lookback
        if market_close is not None:
            mkt_aligned = market_close.reindex(close.index, method="ffill")
            mkt_return = mkt_aligned / mkt_aligned.shift(self.lookback) - 1
            rel_bull = abs_return > mkt_return
        else:
            rel_bull = abs_bull

        # MA filter: structural trend confirmation
        sma = close.rolling(self.sma_period).mean()
        above_sma = close > sma

        buy = abs_bull & rel_bull & above_sma
        # Enter only on the first bar the combined signal fires
        prev_buy = buy.shift(1).fillna(False)
        entry = buy & ~prev_buy

        # Exit when absolute momentum turns negative or price falls below SMA
        exit_signal = (abs_return < 0) | (close < sma)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal & ~entry] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.30,
            position_size=0.02 / 0.05,
        )
