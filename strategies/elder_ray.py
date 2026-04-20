"""
Strategy: Elder Ray Index
Decomposes price action into trend (EMA) and power (Bulls/Bears Power).

Indicators:
  Bull Power = High - EMA(Close, period)
  Bear Power = Low  - EMA(Close, period)

Buy  : EMA is rising (uptrend confirmed) AND Bear Power is negative but
       rising (buying pressure emerging from oversold dip).
Sell : EMA is falling (downtrend confirmed) AND Bull Power is positive but
       falling (selling pressure re-asserting from overbought bounce).

Reference: Dr. Alexander Elder, "Trading for a Living" (1993), Chapter 38.
Suitable for: trending equity markets; daily timeframe; KRX stocks.
"""
import pandas as pd
from .base import BaseStrategy, Signal


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def compute_elder_ray(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 13,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema = _ema(close, ema_period)
    bull_power = high - ema
    bear_power = low - ema
    return ema, bull_power, bear_power


class ElderRayStrategy(BaseStrategy):
    def __init__(
        self,
        ema_period: int = 13,
        stop_loss: float = 0.05,
        take_profit: float = 0.12,
    ):
        self.ema_period = ema_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema, bull_power, bear_power = compute_elder_ray(
            df["high"], df["low"], df["close"], self.ema_period
        )

        ema_rising = ema > ema.shift(1)
        ema_falling = ema < ema.shift(1)

        # Bear Power negative but turning up → buying dip within uptrend
        bear_rising = bear_power > bear_power.shift(1)
        # Bull Power positive but turning down → selling bounce within downtrend
        bull_falling = bull_power < bull_power.shift(1)

        entry = ema_rising & (bear_power < 0) & bear_rising
        exit_sig = ema_falling & (bull_power > 0) & bull_falling

        signals = pd.Series(0, index=df.index)
        signals[entry] = 1
        signals[exit_sig] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=min(0.02 / self.stop_loss, 1.0),
        )
