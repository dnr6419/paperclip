"""
Strategy: Williams %R Mean Reversion
Entry: Williams %R crosses above -80 (exits oversold) while price > 50-day SMA.
Exit: Williams %R crosses above -20 (overbought) or stop-loss triggered.
Reference: Williams (1973) "How I Made One Million Dollars Last Year Trading Commodities".
"""
import pandas as pd
from .base import BaseStrategy, Signal


class WilliamsRReversionStrategy(BaseStrategy):
    def __init__(
        self,
        wr_period: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
        sma_filter: int = 50,
        stop_loss: float = 0.04,
    ):
        self.wr_period = wr_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_filter = sma_filter
        self.stop_loss = stop_loss

    def _calc_williams_r(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        highest_high = high.rolling(self.wr_period).max()
        lowest_low = low.rolling(self.wr_period).min()
        denom = (highest_high - lowest_low).replace(0, float("nan"))
        wr = -100 * (highest_high - close) / denom
        return wr.fillna(-50)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        wr = self._calc_williams_r(high, low, close)

        # Trend filter: price above 50-day SMA
        sma = close.rolling(self.sma_filter).mean()
        trend_up = close > sma

        # Entry: %R crosses above oversold (-80) in uptrend
        cross_up = (wr > self.oversold) & (wr.shift(1) <= self.oversold)
        entry = cross_up & trend_up

        # De-duplicate consecutive entry bars
        prev_entry = entry.shift(1).fillna(False).astype(bool)
        entry = entry & ~prev_entry

        # Exit: %R crosses above overbought (-20)
        exit_signal = (wr > self.overbought) & (wr.shift(1) <= self.overbought)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=0.12,
            position_size=0.02 / self.stop_loss,
        )
