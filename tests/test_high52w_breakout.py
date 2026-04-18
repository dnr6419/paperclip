import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv, make_trending_ohlcv
from strategies.high52w_breakout import High52WBreakoutStrategy


@pytest.fixture
def strategy():
    return High52WBreakoutStrategy(lookback=252, vol_multiplier=2.0)


class TestHigh52WSignals:
    def test_signals_valid_values(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_same_index(self, strategy):
        df = make_ohlcv(n=300)
        signals = strategy.generate_signals(df)
        assert signals.index.equals(df.index)

    def test_buy_on_52w_high_breakout(self):
        """Craft a clear 52-week high breakout."""
        n = 270
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # Flat for 252 days, then breakout
        prices = [100.0] * 252 + [102.5 + i * 0.5 for i in range(n - 252)]
        volumes = [300_000] * 252 + [800_000] * (n - 252)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": volumes
        }, index=dates)
        strategy = High52WBreakoutStrategy(lookback=252, vol_multiplier=1.5,
                                           min_daily_gain=0.0, max_daily_gain=0.5)
        signals = strategy.generate_signals(df)
        assert (signals == 1).any(), "Should detect 52-week high breakout"

    def test_no_buy_without_volume_spike(self, strategy):
        df = make_trending_ohlcv(n=300)
        # Suppress volume to below threshold
        df["volume"] = 100
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 0

    def test_market_filter_blocks_buy_in_downmarket(self):
        df = make_trending_ohlcv(n=300)
        market = pd.Series(
            [5000.0 - i * 5 for i in range(len(df))], index=df.index
        )
        strategy = High52WBreakoutStrategy(lookback=50, vol_multiplier=0.5,
                                           min_daily_gain=0.0, max_daily_gain=0.5)
        signals = strategy.generate_signals(df, market_close=market)
        assert (signals == 1).sum() == 0, "Market filter should block buys in downtrend"

    def test_market_filter_allows_buy_in_upmarket(self):
        df = make_trending_ohlcv(n=300)
        market = pd.Series(
            [3000.0 + i * 5 for i in range(len(df))], index=df.index
        )
        strategy = High52WBreakoutStrategy(lookback=50, vol_multiplier=0.5,
                                           min_daily_gain=0.0, max_daily_gain=0.5)
        signals = strategy.generate_signals(df, market_close=market)
        # In upmarket with relaxed thresholds, should find some buys
        assert (signals == 1).any()


class TestHigh52WRiskParams:
    def test_stop_below_breakout_day_low(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=98.5, breakout_day_low=100,
            peak_price=102, holding_days=2
        )
        assert result == -1

    def test_trailing_stop_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=110, breakout_day_low=98,
            peak_price=120, holding_days=5
        )
        assert result == -1  # 110 is -8.3% from peak 120

    def test_take_profit_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=115.5, breakout_day_low=98,
            peak_price=115.5, holding_days=5
        )
        assert result == -1

    def test_time_stop_fires(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=104, breakout_day_low=98,
            peak_price=104, holding_days=10
        )
        assert result == -1  # 10 days, only +4% < 5%

    def test_hold_in_normal_breakout(self, strategy):
        result = strategy.apply_stop_loss_take_profit(
            entry_price=100, current_price=112, breakout_day_low=98,
            peak_price=115, holding_days=5
        )
        assert result == 0

    def test_params_valid(self, strategy):
        params = strategy.get_signal_params()
        assert params.stop_loss == 0.08
        assert params.take_profit == 0.15
        assert params.take_profit2 == 0.25
        assert params.position_size == 0.06
