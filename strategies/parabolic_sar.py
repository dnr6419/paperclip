"""
Strategy: Parabolic SAR Trend-Following
Entry: Price crosses above SAR (SAR flips from above to below price) — bullish flip.
Exit: Price crosses below SAR (SAR flips from below to above price) — bearish flip,
      or stop-loss hit.
Reference: J. Welles Wilder Jr. (1978), "New Concepts in Technical Trading Systems".
KRX applicability: Yes — daily bars, responsive to Korean market's trending sessions.
"""
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


class ParabolicSARStrategy(BaseStrategy):
    def __init__(
        self,
        af_init: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
        adx_filter: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        stop_loss: float = 0.05,
    ):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max
        self.adx_filter = adx_filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.stop_loss = stop_loss

    def _calc_psar(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """Returns trend series: 1=bullish, -1=bearish."""
        n = len(close)
        trend = np.zeros(n, dtype=int)
        psar = np.zeros(n)
        ep = np.zeros(n)  # extreme point
        af = np.zeros(n)

        # Initialise with first bar
        psar[0] = low.iloc[0]
        trend[0] = 1
        ep[0] = high.iloc[0]
        af[0] = self.af_init

        for i in range(1, n):
            prev_trend = trend[i - 1]
            prev_psar = psar[i - 1]
            prev_ep = ep[i - 1]
            prev_af = af[i - 1]
            cur_high = high.iloc[i]
            cur_low = low.iloc[i]

            if prev_trend == 1:
                # Uptrend: SAR is below price
                new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
                # SAR must not be above the two previous lows
                new_psar = min(new_psar, low.iloc[i - 1])
                if i >= 2:
                    new_psar = min(new_psar, low.iloc[i - 2])

                if cur_low < new_psar:
                    # Trend reversal to bearish
                    trend[i] = -1
                    psar[i] = prev_ep
                    ep[i] = cur_low
                    af[i] = self.af_init
                else:
                    trend[i] = 1
                    psar[i] = new_psar
                    if cur_high > prev_ep:
                        ep[i] = cur_high
                        af[i] = min(prev_af + self.af_step, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:
                # Downtrend: SAR is above price
                new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
                # SAR must not be below the two previous highs
                new_psar = max(new_psar, high.iloc[i - 1])
                if i >= 2:
                    new_psar = max(new_psar, high.iloc[i - 2])

                if cur_high > new_psar:
                    # Trend reversal to bullish
                    trend[i] = 1
                    psar[i] = prev_ep
                    ep[i] = cur_high
                    af[i] = self.af_init
                else:
                    trend[i] = -1
                    psar[i] = new_psar
                    if cur_low < prev_ep:
                        ep[i] = cur_low
                        af[i] = min(prev_af + self.af_step, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        return pd.Series(trend, index=close.index)

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        dm_plus = (high - prev_high).clip(lower=0)
        dm_minus = (prev_low - low).clip(lower=0)
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        p = self.adx_period
        atr = tr.ewm(span=p, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(span=p, adjust=False).mean() / atr
        di_minus = 100 * dm_minus.ewm(span=p, adjust=False).mean() / atr

        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9))
        return dx.ewm(span=p, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        trend = self._calc_psar(high, low, close)

        # Entry: SAR flips bullish (bearish→bullish)
        entry = (trend == 1) & (trend.shift(1) == -1)

        # ADX trend strength filter
        if self.adx_filter and "high" in df.columns:
            adx = self._calc_adx(high, low, close)
            entry = entry & (adx >= self.adx_threshold)

        # Exit: SAR flips bearish (bullish→bearish)
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
