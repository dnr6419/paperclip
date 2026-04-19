"""
Strategy 2: RSI Oversold/Overbought Mean Reversion
Buy when RSI(14) crosses back above 30 (oversold exit) with price > SMA(200).
Sell when RSI reaches 65 or -5% stop-loss.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss == 0 and avg_gain > 0, RSI is 100 (fully overbought)
    rsi = rsi.where(avg_loss > 0, 100.0)
    return rsi


class RSIReversalStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, oversold: float = 40, overbought: float = 70,
                 sell_rsi: float = 70, sma_period: int = 200):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.sell_rsi = sell_rsi
        self.sma_period = sma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: close.
        Returns Series: 1=buy, -1=sell, 0=hold.
        """
        close = df["close"]
        rsi = compute_rsi(close, self.rsi_period)
        sma200 = close.rolling(self.sma_period).mean()

        # Buy: RSI crosses upward through oversold, price above long-term MA
        rsi_cross_up = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        above_sma = close > sma200

        buy = rsi_cross_up & above_sma

        # Sell: RSI reaches overbought zone
        sell = rsi >= self.sell_rsi

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.05,
            take_profit=0.07,
            position_size=0.015 / 0.05,
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    current_rsi: float) -> int:
        pct = (current_price - entry_price) / entry_price
        if pct <= -0.05:
            return -1
        if pct >= 0.07:
            return -1
        if current_rsi < 20:
            return -1
        return 0
