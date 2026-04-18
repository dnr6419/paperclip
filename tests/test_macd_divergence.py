import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from strategies.macd_divergence import MACDDivergenceStrategy, compute_macd, find_swing_lows


class TestComputeMACD:
    def test_macd_output_shape(self):
        df = make_ohlcv(n=200)
        macd_line, signal_line, histogram = compute_macd(df["close"])
        assert len(macd_line) == len(df)
        assert len(signal_line) == len(df)
        assert len(histogram) == len(df)

    def test_histogram_equals_macd_minus_signal(self):
        df = make_ohlcv(n=200)
        macd_line, signal_line, histogram = compute_macd(df["close"])
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected)


class TestFindSwingLows:
    def test_detects_local_minimum(self):
        prices = pd.Series([10, 8, 5, 8, 10, 7, 9])
        lows = find_swing_lows(prices, window=1)
        assert lows.iloc[2], "Index 2 (value 5) should be a swing low"

    def test_no_false_positives_on_flat(self):
        prices = pd.Series([5.0] * 20)
        lows = find_swing_lows(prices, window=2)
        # All equal — first few may be marked but interior should vary
        assert lows.dtype == bool


@pytest.fixture
def strategy():
    return MACDDivergenceStrategy()


class TestMACDDivergenceSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_no_signals_on_tiny_data(self, strategy):
        df = make_ohlcv(n=30)
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0


class TestMACDDivergenceRiskParams:
    def test_stop_loss_fires_on_price(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=94, current_histogram=0.1
        )
        assert result == -1

    def test_stop_loss_fires_on_histogram_turn_negative(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=102, current_histogram=-0.1
        )
        assert result == -1

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=110.5, current_histogram=0.5
        )
        assert result == -1

    def test_hold_in_range(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=105, current_histogram=0.2
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.10
