import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.aroon_oscillator import AroonOscillatorStrategy, compute_aroon


class TestComputeAroon:
    def test_aroon_up_high_in_sustained_uptrend(self):
        n = 100
        prices = pd.Series(range(100, 100 + n), dtype=float)
        aroon_up, _, _ = compute_aroon(prices, prices, period=25)
        assert aroon_up.dropna().iloc[-1] == 100.0

    def test_aroon_down_high_in_sustained_downtrend(self):
        n = 100
        prices = pd.Series(range(200, 200 - n, -1), dtype=float)
        _, aroon_down, _ = compute_aroon(prices, prices, period=25)
        assert aroon_down.dropna().iloc[-1] == 100.0

    def test_oscillator_range(self):
        df = make_ohlcv(n=300)
        _, _, aroon_osc = compute_aroon(df["high"], df["low"], period=25)
        valid = aroon_osc.dropna()
        assert (valid >= -100).all() and (valid <= 100).all()

    def test_output_length_matches_input(self):
        df = make_ohlcv(n=200)
        aroon_up, aroon_down, aroon_osc = compute_aroon(df["high"], df["low"])
        assert len(aroon_up) == len(df)
        assert len(aroon_down) == len(df)
        assert len(aroon_osc) == len(df)

    def test_oscillator_positive_in_uptrend(self):
        n = 100
        prices = pd.Series(range(100, 100 + n), dtype=float)
        _, _, aroon_osc = compute_aroon(prices, prices, period=25)
        assert aroon_osc.dropna().iloc[-1] > 0

    def test_oscillator_negative_in_downtrend(self):
        n = 100
        prices = pd.Series(range(200, 200 - n, -1), dtype=float)
        _, _, aroon_osc = compute_aroon(prices, prices, period=25)
        assert aroon_osc.dropna().iloc[-1] < 0


@pytest.fixture
def strategy():
    return AroonOscillatorStrategy(period=25, threshold=70.0, stop_loss=0.05, take_profit=0.10)


class TestAroonOscillatorSignals:
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

    def test_no_signal_in_first_rows(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        # Not enough data for Aroon in first `period` rows
        assert (signals.iloc[: strategy.period] == 0).all()


class TestAroonOscillatorParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.10
        assert 0 < params.position_size <= 1.0

    def test_position_size_scales_with_stop_loss(self):
        s1 = AroonOscillatorStrategy(stop_loss=0.02)
        s2 = AroonOscillatorStrategy(stop_loss=0.10)
        assert s1.get_signal_params().position_size > s2.get_signal_params().position_size

    def test_default_direction_is_buy(self, strategy):
        assert strategy.get_signal_params().direction == 1
