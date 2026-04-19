import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv
from strategies.atr_breakout import ATRBreakoutStrategy


@pytest.fixture
def strategy():
    return ATRBreakoutStrategy(lookback=20, atr_period=14, breakout_mult=0.5)


class TestATRBreakoutSignals:
    def test_signals_have_correct_values(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index_as_input(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_trending_market_generates_buy_signals(self, strategy):
        df = make_trending_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected buy signals in uptrend"

    def test_buy_signal_only_on_breakout_day(self, strategy):
        """Signal fires only on the first breakout day, not on continuation."""
        df = make_trending_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        buys = signals[signals == 1]
        if len(buys) >= 2:
            # Consecutive buy days should not both be 1 (entry only fires once)
            idx = signals.index.get_indexer(buys.index)
            consecutive = any(idx[i + 1] == idx[i] + 1 for i in range(len(idx) - 1))
            assert not consecutive, "Buy signals should not fire on consecutive days"

    def test_no_signals_in_flat_market(self):
        """Flat price should never exceed rolling high + ATR buffer."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [100.0] * n
        df = pd.DataFrame({
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [500_000] * n,
        }, index=dates)
        strat = ATRBreakoutStrategy(lookback=20, atr_period=14, breakout_mult=0.5)
        signals = strat.generate_signals(df)
        assert (signals == 1).sum() == 0, "No breakout signals in flat market"

    def test_breakout_triggered_by_sharp_move(self):
        """A sharp upward price jump should trigger a buy signal."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = [50.0] * 60 + [80.0] * 40  # sharp jump on day 60
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [500_000] * n,
        }, index=dates)
        strat = ATRBreakoutStrategy(lookback=20, atr_period=14, breakout_mult=0.5)
        signals = strat.generate_signals(df)
        assert (signals == 1).any(), "Expected buy signal after sharp price jump"


class TestATRBreakoutParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.20
        assert params.position_size > 0

    def test_risk_reward_ratio(self, strategy):
        params = strategy.get_signal_params()
        ratio = params.take_profit / params.stop_loss
        assert ratio >= 2.0, f"Risk/reward {ratio:.1f} < minimum 2.0"

    def test_position_size_proportional_to_stop(self):
        strat = ATRBreakoutStrategy(stop_loss=0.10)
        params = strat.get_signal_params()
        assert abs(params.position_size - 0.02 / 0.10) < 1e-9
