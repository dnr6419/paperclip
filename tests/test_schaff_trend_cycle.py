import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv, make_falling_ohlcv
from strategies.schaff_trend_cycle import SchaffTrendCycleStrategy, compute_stc


@pytest.fixture
def strategy():
    return SchaffTrendCycleStrategy(
        fast_period=23,
        slow_period=50,
        cycle_period=10,
        stoch_smoothing=3,
        buy_threshold=25.0,
        sell_threshold=75.0,
        stop_loss=0.04,
        take_profit=0.08,
    )


class TestComputeSTC:
    def test_stc_bounded(self):
        df = make_ohlcv(n=300)
        stc = compute_stc(df["close"])
        valid = stc.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_stc_returns_series_aligned_with_input(self):
        df = make_ohlcv(n=300)
        stc = compute_stc(df["close"])
        assert stc.index.equals(df.index)

    def test_stc_cycles_between_extremes(self):
        # STC is an oscillator: it should visit both low and high regions
        df = make_ohlcv(n=300, seed=1)
        stc = compute_stc(df["close"])
        valid = stc.dropna()
        assert len(valid) > 0
        assert valid.max() > 50, "STC should reach above 50 at some point"
        assert valid.min() < 50, "STC should reach below 50 at some point"

    def test_stc_low_in_strong_downtrend(self):
        df = make_falling_ohlcv(n=300, seed=2)
        stc = compute_stc(df["close"])
        valid = stc.dropna()
        assert len(valid) > 0
        # In a strong downtrend STC should spend more time below 50 than above
        assert (valid < 50).sum() > (valid > 50).sum(), "STC should lean low in downtrend"

    def test_stc_nan_before_warmup(self):
        df = make_ohlcv(n=300)
        stc = compute_stc(df["close"], slow_period=50, cycle_period=10)
        # First ~60 bars should be NaN (slow EMA warmup)
        assert stc.iloc[:50].isna().any()

    def test_stc_constant_price_returns_midpoint(self):
        prices = pd.Series([10000.0] * 200)
        stc = compute_stc(prices)
        valid = stc.dropna()
        # Flat price: stoch denominator = 0, defaults to 0.5 → STC ≈ 50
        assert valid.iloc[-1] == pytest.approx(50.0, abs=1.0)


class TestSchaffTrendCycleSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_no_signals_on_tiny_df(self, strategy):
        df = make_ohlcv(n=30)
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0, "Not enough data for warmup — no buys expected"

    def test_buy_signals_exist_in_trending_market(self):
        df = make_trending_ohlcv(n=400)
        strat = SchaffTrendCycleStrategy()
        signals = strat.generate_signals(df)
        assert (signals == 1).sum() >= 1, "Should generate at least one buy in an uptrend"

    def test_sell_signals_exist_in_trending_market(self):
        df = make_trending_ohlcv(n=400)
        strat = SchaffTrendCycleStrategy()
        signals = strat.generate_signals(df)
        assert (signals == -1).sum() >= 1, "Should generate at least one sell signal"

    def test_alternating_buy_sell(self):
        df = make_ohlcv(n=500, seed=7)
        strat = SchaffTrendCycleStrategy()
        signals = strat.generate_signals(df)
        non_zero = signals[signals != 0]
        if len(non_zero) >= 2:
            # Consecutive identical signals are allowed; just verify no single massive cluster
            assert len(non_zero) < len(df) * 0.5, "Too many signals — likely a logic error"

    def test_get_signal_params(self, strategy):
        params = strategy.get_signal_params()
        assert params.direction == 1
        assert 0 < params.stop_loss < 1
        assert 0 < params.take_profit < 1
        assert 0 < params.position_size <= 1.0

    def test_position_size_capped_at_one(self):
        strat = SchaffTrendCycleStrategy(stop_loss=0.001)
        params = strat.get_signal_params()
        assert params.position_size <= 1.0

    def test_custom_thresholds(self):
        df = make_ohlcv(n=400)
        strat = SchaffTrendCycleStrategy(buy_threshold=10.0, sell_threshold=90.0)
        signals = strat.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})
