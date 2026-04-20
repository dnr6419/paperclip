import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.cmf import CMFStrategy, compute_cmf


class TestComputeCMF:
    def test_output_length_matches_input(self):
        df = make_ohlcv(n=200)
        cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"])
        assert len(cmf) == len(df)

    def test_range_bounded(self):
        df = make_ohlcv(n=300)
        cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"])
        valid = cmf.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_nan_at_start(self):
        df = make_ohlcv(n=200)
        cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"], period=20)
        assert cmf.iloc[:19].isna().all()

    def test_positive_when_close_near_high(self):
        n = 100
        high = pd.Series([110.0] * n)
        low = pd.Series([100.0] * n)
        close = pd.Series([109.0] * n)  # near high → MFM close to +1
        volume = pd.Series([1_000_000.0] * n)
        cmf = compute_cmf(high, low, close, volume, period=20)
        assert cmf.dropna().iloc[-1] > 0

    def test_negative_when_close_near_low(self):
        n = 100
        high = pd.Series([110.0] * n)
        low = pd.Series([100.0] * n)
        close = pd.Series([101.0] * n)  # near low → MFM close to -1
        volume = pd.Series([1_000_000.0] * n)
        cmf = compute_cmf(high, low, close, volume, period=20)
        assert cmf.dropna().iloc[-1] < 0

    def test_custom_period(self):
        df = make_ohlcv(n=200)
        cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"], period=14)
        assert cmf.dropna().shape[0] > 0


@pytest.fixture
def strategy():
    return CMFStrategy(period=20, buy_threshold=0.05, sell_threshold=-0.05,
                       stop_loss=0.04, take_profit=0.10)


class TestCMFSignals:
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
        assert (signals.iloc[: strategy.period - 1] == 0).all()


class TestCMFParams:
    def test_signal_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.04
        assert params.take_profit == 0.10
        assert 0 < params.position_size <= 1.0

    def test_position_size_scales_with_stop_loss(self):
        s1 = CMFStrategy(stop_loss=0.02)
        s2 = CMFStrategy(stop_loss=0.10)
        assert s1.get_signal_params().position_size > s2.get_signal_params().position_size

    def test_default_direction_is_buy(self, strategy):
        assert strategy.get_signal_params().direction == 1
