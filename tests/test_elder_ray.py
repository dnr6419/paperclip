import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from strategies.elder_ray import ElderRayStrategy, compute_elder_ray


class TestComputeElderRay:
    def test_bull_power_positive_in_strong_uptrend(self):
        n = 100
        prices = pd.Series(range(100, 100 + n), dtype=float)
        highs = prices * 1.005
        lows = prices * 0.995
        _, bull_power, _ = compute_elder_ray(highs, lows, prices)
        # In a strong sustained uptrend price >> EMA so bull power should be positive
        assert bull_power.dropna().iloc[-1] > 0

    def test_bear_power_negative_in_strong_downtrend(self):
        n = 100
        prices = pd.Series(range(200, 200 - n, -1), dtype=float)
        highs = prices * 1.005
        lows = prices * 0.995
        _, _, bear_power = compute_elder_ray(highs, lows, prices)
        assert bear_power.dropna().iloc[-1] < 0

    def test_ema_length_matches_input(self):
        df = make_ohlcv(n=200)
        ema, bull_power, bear_power = compute_elder_ray(df["high"], df["low"], df["close"])
        assert len(ema) == len(df)
        assert len(bull_power) == len(df)
        assert len(bear_power) == len(df)


@pytest.fixture
def strategy():
    return ElderRayStrategy(ema_period=13, stop_loss=0.05, take_profit=0.12)


class TestElderRaySignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_signal_in_uptrend_with_dip(self, strategy):
        """EMA rising + bear power negative but increasing should fire a buy."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 50_000 * np.exp(np.cumsum(rng.normal(0.002, 0.008, n)))
        high = close * (1 + rng.uniform(0, 0.005, n))
        low = close * (1 - rng.uniform(0, 0.005, n))
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": [500_000] * n},
            index=dates,
        )
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected at least one buy signal in uptrend"

    def test_sell_signal_in_downtrend_with_bounce(self, strategy):
        """EMA falling + bull power positive but declining should fire a sell."""
        rng = np.random.default_rng(99)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 50_000 * np.exp(np.cumsum(rng.normal(-0.002, 0.008, n)))
        high = close * (1 + rng.uniform(0, 0.005, n))
        low = close * (1 - rng.uniform(0, 0.005, n))
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": [500_000] * n},
            index=dates,
        )
        signals = strategy.generate_signals(df)
        assert (signals == -1).any(), "Expected at least one sell signal in downtrend"


class TestElderRayParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.12
        assert 0 < params.position_size <= 1.0

    def test_position_size_scales_with_stop_loss(self):
        s1 = ElderRayStrategy(stop_loss=0.02)
        s2 = ElderRayStrategy(stop_loss=0.10)
        assert s1.get_signal_params().position_size > s2.get_signal_params().position_size
