"""
Strategy 13: VIX Spike Reversion (변동성 급등 역추세 전략)

Logic:
  - Detects a volatility spike using either an external VIX/VKOSPI column or
    a rolling Historical Volatility (HV) proxy computed from the close series.
  - Entry (Buy): volatility spikes above a Z-score threshold AND
    the underlying price is oversold (1-day return < -entry_drop_pct).
    This captures fear-driven selloffs that tend to mean-revert.
  - Exit: volatility Z-score reverts below exit_zscore OR
    price recovers by take_profit OR stop-loss is breached OR
    max holding period (max_hold_days) is reached.

Hedge rationale:
  Momentum / trend strategies (DCB, MACD, MTM, ATR Breakout) tend to lose
  during sharp VIX spikes.  This strategy fires precisely in those regimes,
  creating a natural negative correlation to the momentum cluster and
  providing portfolio-level downside cushion.

Parameters (defaults tuned for KOSPI 200 / Korean equity universe):
  hv_period         : rolling window for HV computation (default 20 days)
  zscore_period     : lookback for Z-score normalisation (default 60 days)
  spike_zscore      : minimum Z-score to qualify as a spike (default 2.0)
  entry_drop_pct    : minimum 1-day price drop to confirm fear (default 0.02)
  exit_zscore       : Z-score level at which volatility is "normalised" (default 0.5)
  max_hold_days     : hard cap on holding period in bars (default 15)
  stop_loss         : fractional stop below entry (default 0.04 = 4 %)
  take_profit       : fractional target above entry (default 0.08 = 8 %)

VKOSPI support:
  If the input DataFrame contains a column named 'vkospi' (or 'vix'), that
  series is used directly as the volatility measure instead of HV.
  Pass daily VKOSPI closing values aligned with the price index.
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal


def _compute_hv(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualised historical volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(period).std() * np.sqrt(252)
    return hv


def _zscore(series: pd.Series, period: int) -> pd.Series:
    mu = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return (series - mu) / sigma.replace(0, np.nan)


class VIXSpikeReversionStrategy(BaseStrategy):
    def __init__(
        self,
        hv_period: int = 20,
        zscore_period: int = 60,
        spike_zscore: float = 2.0,
        entry_drop_pct: float = 0.02,
        exit_zscore: float = 0.5,
        max_hold_days: int = 15,
        stop_loss: float = 0.04,
        take_profit: float = 0.08,
    ):
        self.hv_period = hv_period
        self.zscore_period = zscore_period
        self.spike_zscore = spike_zscore
        self.entry_drop_pct = entry_drop_pct
        self.exit_zscore = exit_zscore
        self.max_hold_days = max_hold_days
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def _get_vol_series(self, df: pd.DataFrame) -> pd.Series:
        for col in ("vkospi", "vix", "VKOSPI", "VIX"):
            if col in df.columns:
                return df[col].astype(float)
        return _compute_hv(df["close"], self.hv_period)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        vol = self._get_vol_series(df)
        vol_z = _zscore(vol, self.zscore_period)

        daily_ret = close.pct_change()

        # Entry: vol spike + price fear drop
        spike = vol_z >= self.spike_zscore
        fear_drop = daily_ret <= -self.entry_drop_pct
        buy = spike & fear_drop

        # Exit: vol normalises
        vol_normalised = vol_z <= self.exit_zscore

        signals = pd.Series(0, index=df.index)

        in_position = False
        entry_idx = None
        hold_count = 0

        for i in range(len(df)):
            if not in_position:
                if buy.iloc[i]:
                    signals.iloc[i] = 1
                    in_position = True
                    entry_idx = i
                    hold_count = 0
            else:
                hold_count += 1
                should_exit = (
                    vol_normalised.iloc[i]
                    or hold_count >= self.max_hold_days
                )
                if should_exit:
                    signals.iloc[i] = -1
                    in_position = False
                    entry_idx = None
                    hold_count = 0

        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size=0.08,
        )
