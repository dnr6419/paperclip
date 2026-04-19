import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.adx_trend import ADXTrendStrategy, compute_adx_full


class TestComputeADXFull:
    def test_output_lengths(self):
        df = make_ohlcv(n=100)
        adx, plus_di, minus_di = compute_adx_full(df["high"], df["low"], df["close"])
        assert len(adx) == len(df)
        assert len(plus_di) == len(df)
        assert len(minus_di) == len(df)

    def test_adx_non_negative(self):
        df = make_ohlcv(n=150)
        adx, _, _ = compute_adx_full(df["high"], df["low"], df["close"])
        assert (adx.dropna() >= 0).all()

    def test_plus_di_higher_in_uptrend(self):
        df = make_trending_ohlcv(n=200)
        _, plus_di, minus_di = compute_adx_full(df["high"], df["low"], df["close"])
        # In strong uptrend, +DI should dominate on average
        tail = slice(-50, None)
        assert plus_di.iloc[tail].mean() > minus_di.iloc[tail].mean()

    def test_minus_di_higher_in_downtrend(self):
        df = make_falling_ohlcv(n=200)
        _, plus_di, minus_di = compute_adx_full(df["high"], df["low"], df["close"])
        tail = slice(-50, None)
        assert minus_di.iloc[tail].mean() > plus_di.iloc[tail].mean()


@pytest.fixture
def strategy():
    return ADXTrendStrategy()


class TestADXTrendSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_in_strong_uptrend(self):
        df = make_trending_ohlcv(n=300)
        # Use lenient thresholds and add volume spike
        df["volume"] = df["volume"] * 2.0  # ensure vol_confirm fires
        strategy = ADXTrendStrategy(adx_threshold=10, sma_period=20, vol_threshold=1.0)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Should generate buy in strong uptrend"

    def test_no_sell_signals(self):
        df = make_falling_ohlcv(n=200)
        strategy = ADXTrendStrategy()
        signals = strategy.generate_signals(df)
        # Strategy uses TP/SL-only exits; no signal-based sells
        assert (signals == -1).sum() == 0


class TestADXTrendRiskParams:
    def test_stop_loss_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=94.5
        )
        assert result == -1

    def test_hold_below_take_profit(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=110
        )
        assert result == 0

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=125.5
        )
        assert result == -1

    def test_hold_in_normal_trend(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=108
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.04
        assert params.take_profit == 0.25
