"""
Leveraged ETF Short-Term Strategies — targeting 80%+ win rate.

Key design principle: asymmetric risk/reward.
Tight profit targets (1-3%) with wider stops (8-15%).
On 3x leveraged ETFs with ~5% daily vol, tight TP is hit quickly.
Combined with directional filters to tilt probability further.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss > 0, 100.0)
    return rsi


class LETFTightTPStrategy(BaseStrategy):
    """
    Variant A — Tight TP with uptrend filter.
    Entry: close > EMA(20) (uptrend) AND RSI(2) < 30 (short-term dip)
    Exit: TP +2% or SL -10% or RSI(2) > 70
    High win rate from asymmetric risk + trend following.
    """
    def __init__(self):
        self.ema_period = 20
        self.rsi_entry = 30
        self.rsi_exit = 70

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi2 = compute_rsi(close, 2)
        ema = close.ewm(span=self.ema_period).mean()

        buy = (close > ema) & (rsi2 < self.rsi_entry)
        sell = rsi2 > self.rsi_exit

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(direction=1, stop_loss=0.10, take_profit=0.02, position_size=0.10)


class LETFMicroScalpStrategy(BaseStrategy):
    """
    Variant B — Micro-scalp: 1.5% target, 12% stop.
    Entry: close > EMA(10) AND 1-day return < -1% (minor pullback in uptrend)
    Exit: TP +1.5% or SL -12% or 3-day max hold
    """
    def __init__(self):
        self.ema_period = 10
        self.pullback = -0.01
        self.max_hold = 3

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema = close.ewm(span=self.ema_period).mean()
        ret1d = close.pct_change()

        buy = (close > ema) & (ret1d < self.pullback)

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1

        in_trade = False
        hold_count = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1 and not in_trade:
                in_trade = True
                hold_count = 0
            elif in_trade:
                hold_count += 1
                if hold_count >= self.max_hold:
                    signals.iloc[i] = -1
                    in_trade = False
                    hold_count = 0

        return signals

    def get_signal_params(self) -> Signal:
        return Signal(direction=1, stop_loss=0.12, take_profit=0.015, position_size=0.10)


class LETFTrendDipStrategy(BaseStrategy):
    """
    Variant C — Strong trend + dip buy.
    Entry: close > EMA(50) AND RSI(5) < 40 AND close > close.shift(5)
    (medium-term uptrend with short-term weakness, but 5-day net positive)
    Exit: TP +2.5% or SL -8% or RSI(5) > 65
    """
    def __init__(self):
        self.ema_period = 50
        self.rsi_period = 5
        self.rsi_entry = 40
        self.rsi_exit = 65
        self.lookback = 5

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi5 = compute_rsi(close, self.rsi_period)
        ema50 = close.ewm(span=self.ema_period).mean()

        buy = (close > ema50) & (rsi5 < self.rsi_entry) & (close > close.shift(self.lookback))
        sell = rsi5 > self.rsi_exit

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(direction=1, stop_loss=0.08, take_profit=0.025, position_size=0.10)


class LETFMomentumBurstStrategy(BaseStrategy):
    """
    Variant D — Momentum burst: buy after 2 consecutive up days with RSI rising.
    Entry: 2 consecutive up closes AND RSI(3) > 50 AND RSI(3) rising AND close > EMA(20)
    Exit: TP +1.5% or SL -10% or first down close
    Exploits short-term momentum continuation in leveraged ETFs.
    """
    def __init__(self):
        self.ema_period = 20
        self.rsi_period = 3

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi3 = compute_rsi(close, self.rsi_period)
        ema20 = close.ewm(span=self.ema_period).mean()

        up1 = close > close.shift(1)
        up2 = close.shift(1) > close.shift(2)
        rsi_rising = rsi3 > rsi3.shift(1)

        buy = up1 & up2 & (rsi3 > 50) & rsi_rising & (close > ema20)

        down_close = close < close.shift(1)
        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[down_close] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(direction=1, stop_loss=0.10, take_profit=0.015, position_size=0.10)
