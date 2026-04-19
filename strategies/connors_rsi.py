"""
Strategy: Connors RSI (CRSI) Mean Reversion
CRSI = average of three components:
  1. RSI(3)  — short-term price RSI
  2. RSI(UpDown Streak) — RSI of consecutive up/down day count
  3. Percent Rank of today's 1-day return vs last `rank_period` days

Entry : CRSI < oversold (default 20) AND close > SMA(200) (trend filter)
Exit  : CRSI > overbought (default 70) OR stop-loss / take-profit hit

Reference: Larry Connors & Cesar Alvarez, "Short-Term Trading Strategies That Work" (2008)
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy, Signal


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    # When avg_loss == 0 and avg_gain > 0, RSI = 100; both zero → 50
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.where(avg_gain > 0, np.inf, 1.0), avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = pd.Series(rsi, index=series.index)
    rsi[avg_gain.isna()] = np.nan
    return rsi


def _streak(close: pd.Series) -> pd.Series:
    """Return series counting consecutive up (+) or down (-) days."""
    direction = np.sign(close.diff())
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        d = direction.iloc[i]
        if d == 0:
            streak.iloc[i] = 0
        elif d == streak.iloc[i - 1] / abs(streak.iloc[i - 1]) if streak.iloc[i - 1] != 0 else False:
            streak.iloc[i] = streak.iloc[i - 1] + d
        else:
            streak.iloc[i] = d
    return streak


def _percent_rank(series: pd.Series, period: int) -> pd.Series:
    """Percent rank: fraction of past `period` values less than today's value."""
    def _prank(window):
        today = window[-1]
        return np.sum(window[:-1] < today) / (len(window) - 1) * 100

    return series.rolling(period).apply(_prank, raw=True)


def compute_crsi(
    close: pd.Series,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> pd.Series:
    rsi3 = _compute_rsi(close, rsi_period)
    streak_vals = _streak(close)
    rsi_streak = _compute_rsi(streak_vals, streak_period)
    roc1 = close.pct_change() * 100
    prank = _percent_rank(roc1, rank_period)
    return (rsi3 + rsi_streak + prank) / 3


class ConnorsRSIStrategy(BaseStrategy):
    def __init__(
        self,
        rsi_period: int = 3,
        streak_period: int = 2,
        rank_period: int = 100,
        oversold: float = 20.0,
        overbought: float = 70.0,
        sma_period: int = 200,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
    ):
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.rank_period = rank_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_period = sma_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        crsi = compute_crsi(close, self.rsi_period, self.streak_period, self.rank_period)
        sma = close.rolling(self.sma_period).mean()

        above_sma = close > sma
        prev_crsi = crsi.shift(1)

        # Entry: CRSI crosses up through oversold threshold while above long-term MA
        entry = (prev_crsi <= self.oversold) & (crsi > self.oversold) & above_sma

        # Exit: CRSI crosses above overbought threshold
        exit_sig = (prev_crsi <= self.overbought) & (crsi > self.overbought)

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_sig] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.02 / self.stop_loss,
        )
