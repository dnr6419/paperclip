"""
MFI (Money Flow Index) Mean Reversion Strategy
Volume-weighted RSI that detects overbought/oversold conditions using both price and volume.
Entry: MFI crosses above oversold (20) while price > SMA(50).
Exit: MFI crosses above overbought (80) or stop-loss triggered.
Reference: Quong & Soudack (1989), Technical Analysis of Stocks & Commodities.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    tp_change = typical_price.diff()
    positive_mf = money_flow.where(tp_change > 0, 0.0)
    negative_mf = money_flow.where(tp_change < 0, 0.0)

    pos_sum = positive_mf.rolling(period).sum()
    neg_sum = negative_mf.rolling(period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.where(neg_sum > 0, 100.0)
    return mfi


class MFIMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        mfi_period: int = 14,
        oversold: float = 20.0,
        overbought: float = 80.0,
        sma_filter: int = 50,
        stop_loss: float = 0.04,
    ):
        self.mfi_period = mfi_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_filter = sma_filter
        self.stop_loss = stop_loss

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: close, high, low, volume.
        Falls back to close for high/low and ones for volume when columns missing.
        Returns Series: 1=buy, -1=sell, 0=hold.
        """
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        mfi = compute_mfi(high, low, close, volume, self.mfi_period)

        sma = close.rolling(self.sma_filter).mean()
        trend_up = close > sma

        cross_up = (mfi > self.oversold) & (mfi.shift(1) <= self.oversold)
        entry = cross_up & trend_up

        prev_entry = entry.shift(1).fillna(False).astype(bool)
        entry = entry & ~prev_entry

        exit_signal = (mfi > self.overbought) & (mfi.shift(1) <= self.overbought)

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
