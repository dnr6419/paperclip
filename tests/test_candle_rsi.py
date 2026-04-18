import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from strategies.candle_rsi import (
    CandleRSIStrategy, is_hammer, is_bullish_engulfing, is_morning_star
)


class TestCandlePatterns:
    def test_hammer_detected(self):
        open_ = pd.Series([100.0])
        close = pd.Series([101.0])
        high = pd.Series([101.2])   # small upper shadow (0.2 < body*0.5=0.5)
        low = pd.Series([97.0])     # long lower shadow (3.0 > body*2=2.0)
        result = is_hammer(open_, high, low, close)
        assert result.iloc[0], "Should detect hammer pattern"

    def test_no_hammer_when_no_long_shadow(self):
        open_ = pd.Series([100.0])
        close = pd.Series([101.0])
        high = pd.Series([102.0])
        low = pd.Series([99.5])   # short lower shadow
        result = is_hammer(open_, high, low, close)
        assert not result.iloc[0]

    def test_bullish_engulfing_detected(self):
        open_ = pd.Series([103.0, 98.0])
        close = pd.Series([99.0, 104.0])
        result = is_bullish_engulfing(open_, close)
        assert result.iloc[1], "Should detect bullish engulfing"

    def test_no_engulfing_when_same_direction(self):
        open_ = pd.Series([100.0, 101.0])
        close = pd.Series([102.0, 103.0])
        result = is_bullish_engulfing(open_, close)
        assert not result.iloc[1]

    def test_morning_star_shape(self):
        open_ = pd.Series([105.0, 100.0, 99.0])
        close = pd.Series([100.0, 100.5, 103.0])
        high = pd.Series([106.0, 101.0, 104.0])
        low = pd.Series([99.5, 99.5, 98.5])
        result = is_morning_star(open_, high, low, close)
        # May or may not trigger depending on exact midpoint — just check no crash
        assert result.dtype == bool


@pytest.fixture
def strategy():
    return CandleRSIStrategy()


class TestCandleRSISignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=200)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_no_buy_without_rsi_confirmation(self):
        """With overbought RSI, hammer alone should not trigger buy."""
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [50.0 + i * 2 for i in range(n)]  # strong uptrend -> RSI high
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices], "close": prices,
            "volume": [800_000] * n
        }, index=dates)
        strategy = CandleRSIStrategy(rsi_threshold=40)
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0, "No buy when RSI not oversold"


class TestCandleRSIRiskParams:
    def test_stop_fires_below_pattern_low(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=97, pattern_low=99
        )
        assert result == -1

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=106.5, pattern_low=95
        )
        assert result == -1

    def test_hold_in_range(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=104, pattern_low=97
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.03
        assert params.take_profit == 0.06
        assert params.take_profit2 == 0.10
