"""
Strategy: Ichimoku Cloud Breakout (이치모쿠 구름 돌파)
Entry: 4-way confirmation — price above Kumo, TK golden cross,
       bullish future Kumo, Chikou Span above past price.
Short: reverse 4-way — price below Kumo, TK dead cross,
       bearish future Kumo, Chikou below past price.
"""
import pandas as pd
from .base import BaseStrategy, Signal


class IchimokuCloudBreakoutStrategy(BaseStrategy):
    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    def _midpoint(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        return (high.rolling(period).max() + low.rolling(period).min()) / 2

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        tenkan = self._midpoint(high, low, self.tenkan_period)
        kijun = self._midpoint(high, low, self.kijun_period)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = self._midpoint(high, low, self.senkou_b_period)

        # Current Kumo: Senkou values projected displacement bars ago are now at current bar
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1).shift(self.displacement)
        cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1).shift(self.displacement)

        # Future Kumo direction (already plotted on chart, no lookahead bias)
        future_bullish = senkou_a > senkou_b
        future_bearish = senkou_a < senkou_b

        # Chikou Span: current close vs price displacement bars ago
        chikou_above = close > close.shift(self.displacement)
        chikou_below = close < close.shift(self.displacement)

        # TK Cross
        tk_golden = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_dead = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

        # 4-way buy confirmation
        buy = (close > cloud_top) & tk_golden & future_bullish & chikou_above
        # 4-way sell confirmation
        sell = (close < cloud_bot) & tk_dead & future_bearish & chikou_below

        # De-duplicate consecutive triggers
        buy = buy & ~buy.shift(1, fill_value=False)
        sell = sell & ~sell.shift(1, fill_value=False)

        signals = pd.Series(0, index=df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.04,
            take_profit=0.12,
            position_size=0.02 / 0.04,
        )
