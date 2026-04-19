"""
Strategy: Stochastic Oscillator Mean Reversion
Entry: %K crosses above %D from oversold zone (<= oversold), with optional
       SMA trend filter (price above SMA) and volume confirmation.
Exit:  %K reaches overbought zone (>= overbought) OR max holding period reached.

%K = (Close - Lowest Low[k_period]) / (Highest High[k_period] - Lowest Low[k_period]) * 100
%D = d_period-bar SMA of %K
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def _stochastic(close: pd.Series, high: pd.Series, low: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = (close - lowest_low) / denom * 100
    d = k.rolling(d_period).mean()
    return k, d


class StochasticMRStrategy(BaseStrategy):
    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        sma_period: int = 200,
        use_sma_filter: bool = True,
        volume_mult: float = 1.0,
        max_hold_days: int = 20,
        stop_loss: float = 0.05,
        take_profit: float = 0.12,
    ):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_period = sma_period
        self.use_sma_filter = use_sma_filter
        self.volume_mult = volume_mult
        self.max_hold_days = max_hold_days
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        k, d = _stochastic(close, high, low, self.k_period, self.d_period)

        # Bullish crossover: %K crosses above %D from oversold territory
        k_prev = k.shift(1)
        d_prev = d.shift(1)
        cross_up = (k > d) & (k_prev <= d_prev)
        in_oversold = k_prev <= self.oversold

        buy_raw = cross_up & in_oversold

        # Optional SMA trend filter: only buy when price is above long-term average
        if self.use_sma_filter and len(close) >= self.sma_period:
            sma = close.rolling(self.sma_period).mean()
            buy_raw = buy_raw & (close > sma)

        # Optional volume confirmation: volume above rolling average
        if self.volume_mult > 1.0:
            vol_avg = volume.rolling(20).mean()
            buy_raw = buy_raw & (volume >= self.volume_mult * vol_avg)

        # Exit: %K crosses into overbought zone
        exit_raw = k >= self.overbought

        signals = pd.Series(0, index=df.index)
        in_position = False
        hold_count = 0

        for i in range(len(df)):
            if not in_position:
                if buy_raw.iloc[i]:
                    signals.iloc[i] = 1
                    in_position = True
                    hold_count = 0
            else:
                hold_count += 1
                if exit_raw.iloc[i] or hold_count >= self.max_hold_days:
                    signals.iloc[i] = -1
                    in_position = False
                    hold_count = 0

        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.02 / self.stop_loss,
        )
