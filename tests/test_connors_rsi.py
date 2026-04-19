import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv
from strategies.connors_rsi import ConnorsRSIStrategy, compute_crsi


@pytest.fixture
def strategy():
    return ConnorsRSIStrategy(
        rsi_period=3,
        streak_period=2,
        rank_period=100,
        oversold=20,
        overbought=70,
        sma_period=200,
        stop_loss=0.05,
        take_profit=0.15,
    )


class TestComputeCRSI:
    def test_crsi_bounded(self):
        df = make_ohlcv(n=400)
        crsi = compute_crsi(df["close"])
        valid = crsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_crsi_low_in_sustained_downtrend(self):
        n = 400
        prices = pd.Series(np.linspace(10000, 5000, n), dtype=float)
        crsi = compute_crsi(prices)
        valid = crsi.dropna()
        assert len(valid) > 0, "Should have valid CRSI values"
        assert valid.iloc[-1] < 40, "CRSI should be low in sustained downtrend"

    def test_crsi_high_in_sustained_uptrend(self):
        n = 400
        prices = pd.Series(np.linspace(5000, 10000, n), dtype=float)
        crsi = compute_crsi(prices)
        valid = crsi.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 60, "CRSI should be high in sustained uptrend"

    def test_crsi_returns_series_aligned_with_input(self):
        df = make_ohlcv(n=400)
        crsi = compute_crsi(df["close"])
        assert crsi.index.equals(df.index)


class TestConnorsRSISignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=500)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=500)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_no_signals_in_downtrend(self):
        """In a sustained downtrend, price stays below SMA200 — no buy signals."""
        df = make_ohlcv(n=500, seed=2, trend=-0.003)
        strat = ConnorsRSIStrategy(sma_period=200)
        signals = strat.generate_signals(df)
        assert (signals == 1).sum() == 0, "Should not buy when price is below SMA200"

    def test_get_signal_params(self, strategy):
        params = strategy.get_signal_params()
        assert params.direction == 1
        assert 0 < params.stop_loss < 1
        assert 0 < params.take_profit < 1
        assert params.position_size > 0

    def test_entry_requires_above_sma(self):
        """Force CRSI below oversold but with price below SMA — no entry."""
        n = 500
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Strongly declining price so it stays below SMA200
        close = 10000 * np.exp(np.cumsum(rng.normal(-0.005, 0.01, n)))
        spread = close * 0.003
        df = pd.DataFrame({
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.ones(n) * 500_000,
        }, index=dates)
        strat = ConnorsRSIStrategy(oversold=20, sma_period=200)
        signals = strat.generate_signals(df)
        assert (signals == 1).sum() == 0
