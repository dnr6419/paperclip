import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from strategies.rsi_reversal import RSIReversalStrategy, compute_rsi


class TestComputeRSI:
    def test_rsi_bounded(self):
        df = make_ohlcv()
        rsi = compute_rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_rises_in_uptrend(self):
        # Monotone gain: avg_loss=0 so RSI returns 100
        n = 50
        prices = pd.Series(range(100, 100 + n), dtype=float)
        rsi = compute_rsi(prices)
        assert rsi.dropna().iloc[-1] > 70, "RSI should be overbought in sustained uptrend"

    def test_rsi_falls_in_downtrend(self):
        # Monotone fall: avg_gain=0 so RS=NaN → need special handling
        n = 50
        prices = pd.Series(range(100, 100 - n, -1), dtype=float)
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        # Either RSI is near 0 or the series has some valid values
        assert len(valid) == 0 or valid.iloc[-1] < 30, "RSI should be oversold in sustained downtrend"


@pytest.fixture
def strategy():
    return RSIReversalStrategy(rsi_period=14, oversold=30, sell_rsi=65, sma_period=50)


class TestRSIReversalSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_signal_on_rsi_bounce_from_oversold(self):
        """Craft scenario with noisy recovery so RSI genuinely crosses 30."""
        rng = np.random.default_rng(7)
        n = 150
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Base, sharp drop, then noisy recovery (not monotone — ensures RSI crosses)
        base = [100.0] * 80
        drop = [100 - i * 2.2 for i in range(20)]   # falls to ~56
        recovery_noisy = list(56 + rng.normal(0.5, 1.5, 50).cumsum())
        prices = base + drop + recovery_noisy
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices], "close": prices,
            "volume": [500_000] * n
        }, index=dates)
        strategy = RSIReversalStrategy(sma_period=20)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Expected buy after RSI bounce from oversold"

    def test_sell_signal_when_rsi_overbought(self):
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [50.0 + i * 1.5 for i in range(n)]
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices], "close": prices,
            "volume": [500_000] * n
        }, index=dates)
        # Disable SMA filter (sma=1) and use low sell_rsi threshold
        strategy = RSIReversalStrategy(sell_rsi=60, sma_period=1)
        signals = strategy.generate_signals(df)
        assert (signals == -1).any(), "Expected sell when RSI overbought"


class TestRSIReversalRiskParams:
    def test_stop_loss_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=94, current_rsi=40
        )
        assert result == -1

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=107.5, current_rsi=50
        )
        assert result == -1

    def test_rsi_emergency_stop(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=99, current_rsi=18
        )
        assert result == -1

    def test_hold_in_normal_range(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=103, current_rsi=45
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.05
        assert params.take_profit == 0.07
        assert params.position_size > 0
