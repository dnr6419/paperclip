import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from strategies.mean_reversion import MeanReversionStrategy, compute_zscore


class TestComputeZScore:
    def test_zscore_near_zero_for_mean(self):
        # Slightly noisy around constant to avoid zero std
        rng = np.random.default_rng(0)
        prices = pd.Series(100.0 + rng.normal(0, 0.01, 30))
        z = compute_zscore(prices, period=20)
        assert abs(z.dropna().iloc[-1]) < 1.0

    def test_zscore_negative_when_below_mean(self):
        prices = pd.Series([100.0] * 25 + [80.0] * 5)
        z = compute_zscore(prices, period=20)
        assert z.iloc[-1] < 0

    def test_zscore_positive_when_above_mean(self):
        prices = pd.Series([100.0] * 25 + [120.0] * 5)
        z = compute_zscore(prices, period=20)
        assert z.iloc[-1] > 0

    def test_zscore_extreme_below(self):
        # 24 prices at 100 then 2 at 50: rolling 20 window has 18x100 + 2x50
        # mean=95, std~15.4, z=(50-95)/15.4 ~ -2.9
        prices = pd.Series([100.0] * 24 + [50.0] * 2)
        z = compute_zscore(prices, period=20)
        assert z.iloc[-1] <= -2.0


@pytest.fixture
def strategy():
    return MeanReversionStrategy()


class TestMeanReversionSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_when_zscore_extreme(self):
        """Manually create extreme dip to trigger buy."""
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Stable then massive drop
        prices = [100.0] * 30 + [70.0] * 30
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.002 for p in prices],
            "low": [p * 0.998 for p in prices], "close": prices,
            "volume": [500_000] * n
        }, index=dates)
        strategy = MeanReversionStrategy(period=20, entry_z=-1.5, rsi_threshold=50)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected buy on extreme negative z-score"

    def test_sell_when_mean_recovered(self):
        """After dip, recovery to mean should produce sell."""
        n = 80
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [100.0] * 30 + [70.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.002 for p in prices],
            "low": [p * 0.998 for p in prices], "close": prices,
            "volume": [500_000] * n
        }, index=dates)
        strategy = MeanReversionStrategy(period=20, entry_z=-1.5, full_exit_z=0.1,
                                         rsi_threshold=60)
        signals = strategy.generate_signals(df)
        assert (signals == -1).any()


class TestMeanReversionRiskParams:
    def test_price_stop_loss(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=93.5, current_zscore=-2.5, holding_days=3
        )
        assert result == -1

    def test_zscore_emergency_stop(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=98, current_zscore=-3.1, holding_days=2
        )
        assert result == -1

    def test_mean_reversion_exit(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=108, current_zscore=0.1, holding_days=5
        )
        assert result == -1

    def test_time_stop(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=99.5, current_zscore=-1.0, holding_days=10
        )
        assert result == -1

    def test_hold_within_bounds(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=103, current_zscore=-1.5, holding_days=4
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.06
        assert params.position_size == 0.03
