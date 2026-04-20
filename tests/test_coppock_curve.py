import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.coppock_curve import CoppockCurveStrategy, compute_coppock


class TestComputeCoppock:
    def test_output_length_matches_input(self):
        df = make_ohlcv(n=200)
        coppock = compute_coppock(df["close"])
        assert len(coppock) == len(df)

    def test_nan_at_start(self):
        df = make_ohlcv(n=200)
        coppock = compute_coppock(df["close"], roc_long=14, roc_short=11, wma_period=10)
        # Need roc_long + wma_period - 1 = 23 bars before first valid value
        assert coppock.iloc[:23].isna().all()

    def test_positive_in_sustained_uptrend(self):
        prices = pd.Series(range(100, 250), dtype=float)
        coppock = compute_coppock(prices)
        assert coppock.dropna().iloc[-1] > 0

    def test_negative_in_sustained_downtrend(self):
        prices = pd.Series(range(250, 100, -1), dtype=float)
        coppock = compute_coppock(prices)
        assert coppock.dropna().iloc[-1] < 0

    def test_custom_periods(self):
        df = make_ohlcv(n=200)
        coppock = compute_coppock(df["close"], roc_long=20, roc_short=14, wma_period=14)
        assert coppock.dropna().shape[0] > 0


@pytest.fixture
def strategy():
    return CoppockCurveStrategy(stop_loss=0.05, take_profit=0.15)


class TestCoppockCurveSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_signal_in_uptrend(self, strategy):
        df = make_trending_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected at least one buy signal in uptrend"

    def test_sell_signal_in_downtrend(self, strategy):
        df = make_falling_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert (signals == -1).any(), "Expected at least one sell signal in downtrend"

    def test_no_signal_in_warmup_rows(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        warmup = strategy.roc_long + strategy.wma_period
        assert (signals.iloc[:warmup] == 0).all()


class TestCoppockCurveParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.15
        assert 0 < params.position_size <= 1.0

    def test_position_size_scales_with_stop_loss(self):
        s1 = CoppockCurveStrategy(stop_loss=0.02)
        s2 = CoppockCurveStrategy(stop_loss=0.10)
        assert s1.get_signal_params().position_size > s2.get_signal_params().position_size

    def test_default_direction_is_buy(self, strategy):
        assert strategy.get_signal_params().direction == 1
