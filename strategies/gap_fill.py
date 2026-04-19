"""
Strategy 11: Gap Fill
Entry: buy when today's open gaps down > gap_threshold from yesterday's close
and RSI(14) < rsi_oversold, betting on mean-reversion gap fill.
Exit: close returns to previous close (gap filled) or stop/TP hit.
Parameters: gap_threshold=0.02, rsi_period=14, rsi_oversold=35.
"""
import pandas as pd
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


class GapFillStrategy(BaseStrategy):
    def __init__(
        self,
        gap_threshold: float = 0.02,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
    ):
        self.gap_threshold = gap_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        open_ = df["open"] if "open" in df.columns else close

        prev_close = close.shift(1)
        gap_down = (prev_close - open_) / prev_close

        rsi = compute_rsi(close, self.rsi_period)

        entry = (gap_down > self.gap_threshold) & (rsi < self.rsi_oversold)

        gap_filled = close >= prev_close
        exit_signal = gap_filled

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal & ~entry] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.03,
            take_profit=0.05,
            position_size=0.02 / 0.03,
        )
