"""
Strategy 1: EMA Crossover Momentum
Buy when EMA(12) crosses above EMA(26) with volume confirmation.
Sell on dead cross, -4% stop-loss, or +10%/+15% take-profit.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, vol_multiplier: float = 1.5):
        self.fast = fast
        self.slow = slow
        self.vol_multiplier = vol_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df must contain: close, volume.
        Returns Series: 1=buy, -1=sell, 0=hold.
        """
        close = df["close"]
        volume = df["volume"]

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        vol_avg = volume.rolling(20).mean()

        golden_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        dead_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        vol_confirm = volume > (vol_avg * self.vol_multiplier)
        prev_close_above_slow = close.shift(1) > ema_slow.shift(1)

        buy = golden_cross & vol_confirm & prev_close_above_slow
        sell = dead_cross

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.04,
            take_profit=0.10,
            take_profit2=0.15,
            position_size=0.02 / 0.04,  # capital*2% / stop4% = 50% of capital unit
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float) -> int:
        """Returns -1 (exit) or 0 (hold) based on price movement from entry."""
        pct = (current_price - entry_price) / entry_price
        if pct <= -0.04:
            return -1
        if pct >= 0.10:
            return -1
        return 0
