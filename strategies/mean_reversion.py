"""
Strategy 7: Mean Reversion (Z-Score)
Buy when Z-Score <= -2.0 (price 2 std below 20-day mean) with RSI < 35 confirmation.
Partial exit at Z-Score = -1.0, full exit at Z-Score = 0 (mean reversion complete).
Stop-loss: Z-Score <= -3.0 or -6% price drop.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


def compute_zscore(close: pd.Series, period: int = 20) -> pd.Series:
    rolling_mean = close.rolling(period).mean()
    rolling_std = close.rolling(period).std()
    return (close - rolling_mean) / rolling_std.replace(0, np.nan)


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, period: int = 20, entry_z: float = -2.0,
                 partial_exit_z: float = -1.0, full_exit_z: float = 0.0,
                 stop_z: float = -3.0, rsi_threshold: float = 35):
        self.period = period
        self.entry_z = entry_z
        self.partial_exit_z = partial_exit_z
        self.full_exit_z = full_exit_z
        self.stop_z = stop_z
        self.rsi_threshold = rsi_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: close.
        Returns 1 (buy), -1 (sell), 0 (hold).
        """
        close = df["close"]
        zscore = compute_zscore(close, self.period)
        rsi = compute_rsi(close, 14)

        # Entry: Z-Score crosses below -2.0 and RSI confirms oversold
        entry = (zscore <= self.entry_z) & (rsi < self.rsi_threshold)

        # Exit: Z-Score recovers to 0 (mean reversion complete) or emergency stop
        full_exit = (zscore >= self.full_exit_z) | (zscore <= self.stop_z)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[full_exit] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.06,
            take_profit=None,  # dynamic: z-score = 0
            position_size=0.03,  # split into 2 tranches of 3% each
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    current_zscore: float, holding_days: int) -> int:
        pct = (current_price - entry_price) / entry_price
        if pct <= -0.06:
            return -1
        if current_zscore <= self.stop_z:
            return -1
        if current_zscore >= self.full_exit_z:
            return -1
        if holding_days >= 10 and pct <= 0:
            return -1
        return 0

    def get_zscore(self, df: pd.DataFrame) -> pd.Series:
        return compute_zscore(df["close"], self.period)
