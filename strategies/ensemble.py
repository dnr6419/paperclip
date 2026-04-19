"""
Strategy 12: Cluster Ensemble with Regime Filter

Combines three strategy clusters with cluster-level weighting:
  - Trend-Following   35%: EMA Crossover, ADX Trend, 52W Breakout, ATR Breakout
  - Momentum          30%: MACD Momentum, MTM, DCB
  - Mean-Reversion    35%: RSI Reversal, Candle+RSI, VWB

Regime filter (200-day MA + volatility ratio + ADX) scales each cluster's
effective weight dynamically:
  - Bull / Low-Vol: boosts trend + momentum, holds mean-reversion
  - Bear / High-Vol: dampens trend + momentum, boosts mean-reversion
  - Choppy / Ranging: neutral across clusters, slightly reduces all signals

Signal output: combined score in [-1, 1]. Threshold of 0.20 triggers buy entry.
"""
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal
from .ema_crossover   import EMACrossoverStrategy
from .adx_trend       import ADXTrendStrategy
from .high52w_breakout import High52WBreakoutStrategy
from .atr_breakout    import ATRBreakoutStrategy
from .macd_momentum   import MACDMomentumStrategy
from .mtm             import MTMStrategy
from .dcb             import DCBStrategy
from .rsi_reversal    import RSIReversalStrategy, compute_rsi
from .candle_rsi      import CandleRSIStrategy
from .vwb             import VWBStrategy


# ── Regime detection helpers ──────────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0
    mask2 = dm_minus < dm_plus
    dm_minus[mask2] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.rolling(period).mean() / atr.replace(0, np.nan)
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    adx = dx.rolling(period).mean()
    return adx.fillna(0)


def detect_regime(
    df: pd.DataFrame,
    ma_period: int = 200,
    vol_lookback: int = 20,
    vol_long_period: int = 252,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    Returns a regime Series with values:
      1  = Bull / trending (price > 200MA, low vol, high ADX)
      0  = Neutral / choppy
     -1  = Bear / high-vol

    Composite score uses 3 factors:
      (a) price vs 200-day MA  → +1 / -1
      (b) recent vol / long vol ratio < 1.2 → low-vol (+1), else high-vol (-1)
      (c) ADX > threshold → trending (+1), else ranging (-0.5)

    Score range [-3, 3]; thresholded to {-1, 0, 1}.
    """
    close = df["close"]
    ma200 = close.rolling(ma_period, min_periods=ma_period // 2).mean()

    trend_score = pd.Series(np.where(close > ma200, 1.0, -1.0), index=df.index)

    recent_vol = close.pct_change().rolling(vol_lookback).std() * np.sqrt(252)
    long_vol   = close.pct_change().rolling(vol_long_period, min_periods=100).std() * np.sqrt(252)
    vol_ratio  = (recent_vol / long_vol.replace(0, np.nan)).fillna(1.0)
    vol_score  = pd.Series(np.where(vol_ratio < 1.2, 1.0, -1.0), index=df.index)

    has_ohlc = "high" in df.columns and "low" in df.columns
    if has_ohlc:
        adx = _compute_adx(df, adx_period)
    else:
        # Approximate ADX from close only: use rolling range / rolling mean as proxy
        rng  = close.rolling(adx_period).max() - close.rolling(adx_period).min()
        adx  = (rng / close.rolling(adx_period).mean() * 100).fillna(0)

    adx_score = pd.Series(np.where(adx >= adx_threshold, 1.0, -0.5), index=df.index)

    composite = trend_score + vol_score + adx_score  # range ~[-3, 3]
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[composite >=  1.5] =  1   # bull / trending
    regime[composite <= -1.5] = -1   # bear / high-vol
    return regime


# ── Cluster definitions ────────────────────────────────────────────────────────

TREND_FOLLOWING_STRATEGIES = [
    EMACrossoverStrategy(),
    ADXTrendStrategy(),
    High52WBreakoutStrategy(),
    ATRBreakoutStrategy(),
]

MOMENTUM_STRATEGIES = [
    MACDMomentumStrategy(),
    MTMStrategy(),
    DCBStrategy(),
]

MEAN_REVERSION_STRATEGIES = [
    RSIReversalStrategy(),
    CandleRSIStrategy(),
    VWBStrategy(),
]

# Base cluster weights (must sum to 1.0)
BASE_WEIGHTS = {
    "trend":      0.35,
    "momentum":   0.30,
    "mean_rev":   0.35,
}

# Per-regime multiplier scaling (applied before re-normalizing)
REGIME_MULTIPLIERS = {
    #           trend  momentum  mean_rev
     1: {"trend": 1.4, "momentum": 1.2, "mean_rev": 0.6},   # bull
     0: {"trend": 1.0, "momentum": 1.0, "mean_rev": 1.0},   # neutral
    -1: {"trend": 0.5, "momentum": 0.7, "mean_rev": 1.5},   # bear
}


def _get_cluster_weight(cluster: str, regime: int) -> float:
    mult = REGIME_MULTIPLIERS[regime][cluster]
    return BASE_WEIGHTS[cluster] * mult


def _normalized_cluster_weights(regime: int) -> dict[str, float]:
    raw = {c: _get_cluster_weight(c, regime) for c in BASE_WEIGHTS}
    total = sum(raw.values())
    return {c: v / total for c, v in raw.items()}


# ── Ensemble strategy ─────────────────────────────────────────────────────────

class EnsembleRegimeStrategy(BaseStrategy):
    """
    Cluster ensemble with dynamic regime-based weighting.

    signal_threshold: minimum combined score to trigger a buy (default 0.20).
    Equal weighting within each cluster; cluster weights adjusted by regime.
    """

    def __init__(
        self,
        signal_threshold: float = 0.20,
        ma_period: int = 200,
        adx_threshold: float = 25.0,
        vol_ratio_threshold: float = 1.2,
    ):
        self.signal_threshold  = signal_threshold
        self.ma_period         = ma_period
        self.adx_threshold     = adx_threshold
        self.vol_ratio_threshold = vol_ratio_threshold

        self._clusters = {
            "trend":    TREND_FOLLOWING_STRATEGIES,
            "momentum": MOMENTUM_STRATEGIES,
            "mean_rev": MEAN_REVERSION_STRATEGIES,
        }

    def _cluster_signal(
        self, df: pd.DataFrame, strategies: list[BaseStrategy], market_close: pd.Series | None = None
    ) -> pd.Series:
        """Average normalized signal across all strategies in a cluster."""
        sigs = []
        import inspect
        for strat in strategies:
            try:
                sig_params = inspect.signature(strat.generate_signals).parameters
                if "market_close" in sig_params and market_close is not None:
                    s = strat.generate_signals(df, market_close=market_close)
                else:
                    s = strat.generate_signals(df)
                sigs.append(s.reindex(df.index, fill_value=0).astype(float))
            except Exception:
                pass
        if not sigs:
            return pd.Series(0.0, index=df.index)
        return pd.concat(sigs, axis=1).mean(axis=1)

    def generate_signals(
        self, df: pd.DataFrame, market_close: pd.Series | None = None
    ) -> pd.Series:
        regime = detect_regime(
            df,
            ma_period=self.ma_period,
            adx_threshold=self.adx_threshold,
        )

        cluster_signals = {
            cluster: self._cluster_signal(df, strats, market_close)
            for cluster, strats in self._clusters.items()
        }

        combined = pd.Series(0.0, index=df.index)
        for i, date in enumerate(df.index):
            r = int(regime.iloc[i])
            weights = _normalized_cluster_weights(r)
            score = sum(
                weights[c] * cluster_signals[c].iloc[i]
                for c in self._clusters
            )
            combined.iloc[i] = score

        signals = pd.Series(0, index=df.index)
        signals[combined >= self.signal_threshold]  =  1
        signals[combined <= -self.signal_threshold] = -1
        return signals

    def get_signal_params(self) -> Signal:
        return Signal(
            direction=1,
            stop_loss=0.04,
            take_profit=0.20,
            position_size=0.10,
        )

    def describe_regime_weights(self, regime: int) -> str:
        w = _normalized_cluster_weights(regime)
        labels = {1: "Bull/Trending", 0: "Neutral/Choppy", -1: "Bear/High-Vol"}
        return (
            f"Regime: {labels[regime]} | "
            f"Trend={w['trend']*100:.0f}% "
            f"Momentum={w['momentum']*100:.0f}% "
            f"MeanRev={w['mean_rev']*100:.0f}%"
        )
