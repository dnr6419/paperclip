import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.macd_zero_cross import MACDZeroCrossStrategy


@pytest.fixture
def strategy():
    return MACDZeroCrossStrategy(fast=12, slow=26, signal=9, ma_period=200)


class TestMACDZeroCrossSignals:
    def test_signals_have_correct_values(self, strategy):
        df = make_ohlcv(n=400)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index_as_input(self, strategy):
        df = make_ohlcv(n=400)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_trending_market_generates_buy_signals(self, strategy):
        df = make_trending_ohlcv(n=500)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected at least one buy signal in uptrend"

    def test_falling_market_generates_sell_signals(self, strategy):
        df = make_falling_ohlcv(n=500)
        signals = strategy.generate_signals(df)
        assert (signals == -1).any(), "Expected at least one sell signal in downtrend"

    def test_no_buy_when_price_below_ma(self):
        """No buy signals when price consistently below 200-day MA."""
        n = 400
        dates = pd.date_range("2019-01-01", periods=n, freq="B")
        # Strongly declining: price stays well below its 200-day MA
        prices = [100 - i * 0.1 for i in range(n)]
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices, "volume": [500_000] * n,
        }, index=dates)
        strategy = MACDZeroCrossStrategy(fast=12, slow=26, ma_period=200)
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0, "No buy when price below trend MA"

    def test_macd_zero_cross_up_triggers_buy(self):
        """Craft series where MACD crosses zero upward while above MA."""
        n = 400
        dates = pd.date_range("2019-01-01", periods=n, freq="B")
        # Strong downtrend for 200 bars then sharp reversal to trigger MACD zero-cross
        prices = [200 - i * 0.2 for i in range(200)] + [160 + i * 0.8 for i in range(200)]
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices, "volume": [500_000] * n,
        }, index=dates)
        strategy = MACDZeroCrossStrategy(fast=12, slow=26, ma_period=50)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected buy after MACD crosses zero upward"


class TestMACDZeroCrossParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.25
        assert params.position_size > 0

    def test_risk_reward_ratio(self, strategy):
        params = strategy.get_signal_params()
        ratio = params.take_profit / params.stop_loss
        assert ratio >= 2.5, f"Risk/reward {ratio:.1f} < minimum 2.5"
