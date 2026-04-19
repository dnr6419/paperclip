"""
Strategy: Supertrend Trend-Following
Entry: Price crosses above Supertrend line (trend flips bullish) with volume confirmation.
Exit: Price crosses below Supertrend line (trend flips bearish) or stop-loss hit.
Reference: Olivier Seban (2009), popularized in Indian/Asian retail trading communities.
KRX applicability: Yes — daily bars, handles price-limit days (ATR naturally absorbs gaps).
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class SupertrendStrategy(BaseStrategy):
    def __init__(
        self,
        atr_period: int = 10,
        multiplier: float = 3.0,
        volume_filter: bool = True,
        volume_period: int = 20,
        stop_loss: float = 0.05,
    ):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.volume_filter = volume_filter
        self.volume_period = volume_period
        self.stop_loss = stop_loss

    def _calc_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def _calc_supertrend(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        atr = self._calc_atr(high, low, close)
        hl2 = (high + low) / 2

        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        # Build final bands iteratively so they only tighten, never widen mid-trend
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        for i in range(1, len(close)):
            prev_upper = final_upper.iloc[i - 1]
            prev_lower = final_lower.iloc[i - 1]
            cur_close = close.iloc[i - 1]

            # Upper band: tightens only when prior close was below prior upper band
            final_upper.iloc[i] = (
                basic_upper.iloc[i]
                if basic_upper.iloc[i] < prev_upper or cur_close > prev_upper
                else prev_upper
            )
            # Lower band: tightens only when prior close was above prior lower band
            final_lower.iloc[i] = (
                basic_lower.iloc[i]
                if basic_lower.iloc[i] > prev_lower or cur_close < prev_lower
                else prev_lower
            )

        # Supertrend line: follows final_lower in uptrend, final_upper in downtrend
        supertrend = pd.Series(np.nan, index=close.index)
        trend = pd.Series(1, index=close.index)  # 1=bullish, -1=bearish

        for i in range(1, len(close)):
            prev_trend = trend.iloc[i - 1]
            prev_st = supertrend.iloc[i - 1]
            cur_close = close.iloc[i]
            upper = final_upper.iloc[i]
            lower = final_lower.iloc[i]

            if prev_trend == 1:
                supertrend.iloc[i] = lower
                if cur_close < lower:
                    trend.iloc[i] = -1
                    supertrend.iloc[i] = upper
                else:
                    trend.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper
                if cur_close > upper:
                    trend.iloc[i] = 1
                    supertrend.iloc[i] = lower
                else:
                    trend.iloc[i] = -1

        return trend

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        trend = self._calc_supertrend(high, low, close)

        # Entry: trend flips from bearish to bullish
        entry = (trend == 1) & (trend.shift(1) == -1)

        # Volume confirmation: entry day volume above rolling average
        if self.volume_filter and "volume" in df.columns:
            vol_ma = df["volume"].rolling(self.volume_period).mean()
            entry = entry & (df["volume"] > vol_ma)

        # Exit: trend flips from bullish to bearish
        exit_signal = (trend == -1) & (trend.shift(1) == 1)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_signal] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=0.15,
            position_size=0.02 / self.stop_loss,
        )
