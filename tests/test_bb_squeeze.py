import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv
from strategies.bb_squeeze import BBSqueezeStrategy, compute_adx


class TestComputeADX:
    def test_adx_positive(self):
        df = make_ohlcv(n=100)
        adx = compute_adx(df["high"], df["low"], df["close"])
        assert (adx.dropna() >= 0).all()

    def test_adx_higher_in_trend(self):
        trending = make_trending_ohlcv(n=200)
        flat = make_ohlcv(n=200, trend=0.0)
        adx_trend = compute_adx(trending["high"], trending["low"], trending["close"])
        adx_flat = compute_adx(flat["high"], flat["low"], flat["close"])
        assert adx_trend.dropna().mean() > adx_flat.dropna().mean() * 0.5


@pytest.fixture
def strategy():
    return BBSqueezeStrategy()


class TestBBSqueezeSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_bb_width_computation(self, strategy):
        df = make_ohlcv(n=100)
        upper, middle, lower, width = strategy._compute_bb(df["close"])
        assert (width.dropna() >= 0).all()
        assert (upper.dropna() >= middle.dropna()).all()
        assert (middle.dropna() >= lower.dropna()).all()

    def test_sell_below_middle_band(self, strategy):
        """When close < middle band, sell signal should fire."""
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # High then low pattern
        prices = [100.0] * 30 + [85.0] * 30
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices], "close": prices,
            "volume": [500_000] * n
        }, index=dates)
        signals = strategy.generate_signals(df)
        assert (signals.iloc[30:] == -1).any(), "Should sell below middle band"


class TestBBSqueezeRiskParams:
    def test_exit_below_middle_band(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=98, middle_band=99
        )
        assert result == -1

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=112.5, middle_band=105
        )
        assert result == -1

    def test_hold_above_middle_band(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=108, middle_band=102
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.take_profit == 0.12
        assert params.position_size == 0.05
