"""
Strategy 8: 52-Week High Breakout
Buy when daily close breaks above 52-week (252 trading days) high
with strong volume (2x average) and market in uptrend (above 200-day MA).
Stop-loss: close below breakout-day low. Take-profit: +15% (partial) / +25%.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class High52WBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 200, vol_multiplier: float = 1.5,
                 market_sma: int = 200, max_daily_gain: float = 0.08,
                 min_daily_gain: float = 0.02):
        self.lookback = lookback
        self.vol_multiplier = vol_multiplier
        self.market_sma = market_sma
        self.max_daily_gain = max_daily_gain
        self.min_daily_gain = min_daily_gain

    def generate_signals(self, df: pd.DataFrame,
                         market_close: pd.Series = None) -> pd.Series:
        """
        df must contain: close, volume.
        market_close: optional Series for S&P 500 market filter.
        Returns 1=buy, -1=sell, 0=hold.
        """
        close = df["close"]
        volume = df["volume"]

        rolling_high = close.rolling(self.lookback).max()
        vol_avg = volume.rolling(20).mean()

        # Breakout: today's close exceeds previous 252-day high
        breakout = close >= rolling_high.shift(1)
        vol_spike = volume > vol_avg * self.vol_multiplier

        # Stable breakout: gain between min and max (avoid gap-up chasing)
        daily_return = close.pct_change()
        stable_breakout = (
            (daily_return >= self.min_daily_gain) &
            (daily_return <= self.max_daily_gain)
        )

        buy = breakout & vol_spike & stable_breakout

        # Market filter: S&P 500 above 200-day MA
        if market_close is not None:
            market_above_sma = market_close > market_close.rolling(self.market_sma).mean()
            buy = buy & market_above_sma

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.06,   # trailing stop -6% from peak
            take_profit=0.20,
            take_profit2=0.25,
            position_size=0.06,  # 5-8% of capital
        )

    def apply_stop_loss_take_profit(self, entry_price: float, current_price: float,
                                    breakout_day_low: float, peak_price: float,
                                    holding_days: int) -> int:
        if current_price < breakout_day_low * 0.99:
            return -1
        if peak_price > 0 and (current_price - peak_price) / peak_price <= -0.06:
            return -1
        pct = (current_price - entry_price) / entry_price
        if pct >= 0.20:
            return -1
        if holding_days >= 10 and pct < 0.05:
            return -1
        return 0
