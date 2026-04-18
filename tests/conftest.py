import pandas as pd
import numpy as np
import pytest


def make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0003) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 50_000 * np.exp(np.cumsum(rng.normal(trend, 0.015, n)))
    spread = close * 0.005
    high = close + rng.uniform(0, spread)
    low = close - rng.uniform(0, spread)
    open_ = close + rng.uniform(-spread / 2, spread / 2)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def make_trending_ohlcv(n: int = 300, seed: int = 1) -> pd.DataFrame:
    return make_ohlcv(n=n, seed=seed, trend=0.002)


def make_falling_ohlcv(n: int = 300, seed: int = 2) -> pd.DataFrame:
    return make_ohlcv(n=n, seed=seed, trend=-0.002)
