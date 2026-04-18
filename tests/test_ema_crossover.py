import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.ema_crossover import EMACrossoverStrategy


@pytest.fixture
def strategy():
    return EMACrossoverStrategy(fast=12, slow=26, vol_multiplier=1.5)


class TestEMACrossoverSignals:
    def test_signals_have_correct_values(self, strategy):
        df = make_ohlcv()
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index_as_input(self, strategy):
        df = make_ohlcv()
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_trending_market_generates_buy_signals(self, strategy):
        df = make_trending_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected at least one buy signal in uptrend"

    def test_falling_market_generates_sell_signals(self, strategy):
        df = make_falling_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert (signals == -1).any(), "Expected at least one sell signal in downtrend"

    def test_no_signal_without_volume_spike(self):
        df = make_ohlcv()
        # Suppress volume
        df["volume"] = 1
        strategy = EMACrossoverStrategy(vol_multiplier=1.5)
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0, "No buy without volume confirmation"

    def test_golden_cross_triggers_buy(self):
        """Manually craft a golden cross scenario using a longer series."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Long decline then clear uptrend to force a golden cross
        prices = [100 - i * 0.3 for i in range(50)] + [85 + i * 0.8 for i in range(50)]
        volumes = [300_000] * 50 + [900_000] * 50
        df = pd.DataFrame({"open": prices, "high": [p * 1.01 for p in prices],
                           "low": [p * 0.99 for p in prices],
                           "close": prices, "volume": volumes}, index=dates)
        # Disable volume multiplier and SMA close filter for simplicity
        strategy = EMACrossoverStrategy(fast=5, slow=12, vol_multiplier=1.0)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any()


class TestEMACrossoverRiskParams:
    def test_stop_loss_triggers_below_threshold(self, strategy):
        result = strategy.apply_stop_loss_take_profit(entry_price=100, current_price=95.9)
        assert result == -1

    def test_hold_within_range(self, strategy):
        result = strategy.apply_stop_loss_take_profit(entry_price=100, current_price=105)
        assert result == 0

    def test_take_profit_triggers_at_threshold(self, strategy):
        result = strategy.apply_stop_loss_take_profit(entry_price=100, current_price=110)
        assert result == -1

    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.04
        assert params.take_profit == 0.10
        assert params.take_profit2 == 0.15
        assert params.position_size > 0

    def test_risk_reward_ratio(self, strategy):
        params = strategy.get_signal_params()
        ratio = params.take_profit / params.stop_loss
        assert ratio >= 2.5, f"Risk/reward {ratio:.1f} < minimum 2.5"
