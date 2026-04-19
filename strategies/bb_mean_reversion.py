"""
Strategy 8: Bollinger Band Bounce
Buy when price crosses back above the lower BB(20, std) after being below it
(confirmed mean-reversion bounce), with RSI(14) < rsi_cap (not overbought) and
price above SMA(sma_period) (medium-term uptrend filter).

Volatility filter: require ATR(atr_period) > ATR SMA(atr_ma_period) * atr_threshold
to only trade during expanding volatility where mean-reversion is more reliable.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal
from .rsi_reversal import compute_rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class BBMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_cap: float = 55.0,
        sma_period: int = 50,
        take_profit: float = 0.25,
        stop_loss: float = 0.05,
        atr_period: int = 14,
        atr_ma_period: int = 50,
        atr_threshold: float = 1.0,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_cap = rsi_cap
        self.sma_period = sma_period
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.atr_period = atr_period
        self.atr_ma_period = atr_ma_period
        self.atr_threshold = atr_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        lower_bb = sma - self.bb_std * std

        sma_trend = close.rolling(self.sma_period).mean()
        rsi = compute_rsi(close, self.rsi_period)

        # ATR volatility filter: only trade when ATR is above its moving average
        atr = compute_atr(df, self.atr_period)
        atr_ma = atr.rolling(self.atr_ma_period).mean()
        vol_expanding = atr >= atr_ma * self.atr_threshold

        # Bounce: price was below lower BB yesterday, now crosses back above
        was_below = close.shift(1) < lower_bb.shift(1)
        now_above = close >= lower_bb
        bb_bounce = was_below & now_above

        # Medium-term uptrend and not overbought
        uptrend = close > sma_trend
        not_overbought = rsi < self.rsi_cap

        entry = bb_bounce & uptrend & not_overbought & vol_expanding

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.015 / self.stop_loss,
        )
