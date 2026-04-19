"""
Backtesting engine: event-driven single-stock backtester.
Supports stop-loss, take-profit, and position sizing.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'end'


@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    period: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None

    # Computed metrics
    cagr: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_days: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    stop_loss: float,
    take_profit: float,
    position_size: float = 0.1,
    initial_capital: float = 10_000_000,
    trailing_atr_series: Optional[pd.Series] = None,
    trailing_atr_mult: float = 2.0,
) -> BacktestResult:
    """
    df: OHLCV with lowercase columns, DatetimeIndex
    signals: Series aligned with df.index (1=buy, -1=sell, 0=hold)
    stop_loss: fraction below entry (e.g. 0.04)
    take_profit: fraction above entry (e.g. 0.10)
    position_size: fraction of capital per trade
    """
    capital = float(initial_capital)
    equity = []
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    shares = 0
    high_water = 0.0

    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes
    opens = df["open"].values if "open" in df.columns else closes
    dates = df.index

    for i in range(len(df)):
        price = closes[i]
        high = highs[i]
        low = lows[i]

        if in_position:
            # Track high water mark for trailing stop
            high_water = max(high_water, high)

            # Hard stop floor
            sl_price = entry_price * (1 - stop_loss)

            # Trailing ATR stop overrides hard stop when it's higher
            if trailing_atr_series is not None:
                atr_val = trailing_atr_series.iloc[i]
                trailing_sl = high_water - trailing_atr_mult * atr_val
                sl_price = max(sl_price, trailing_sl)

            tp_price = entry_price * (1 + take_profit)
            exit_price = None
            exit_reason = None

            if low <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
            elif high >= tp_price:
                exit_price = tp_price
                exit_reason = "take_profit"
            elif signals.iloc[i] == -1:
                exit_price = price
                exit_reason = "signal"

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price
                capital += shares * exit_price
                trade = Trade(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                in_position = False
                high_water = 0.0

        elif signals.iloc[i] == 1 and not in_position:
            # Enter at next bar open (approximated as current close for daily)
            entry_price = opens[i] if i + 1 < len(df) else price
            trade_capital = capital * position_size
            shares = trade_capital / entry_price
            capital -= shares * entry_price
            entry_date = dates[i]
            in_position = True
            high_water = entry_price

        equity.append(capital + (shares * price if in_position else 0))

    # Close open position at end
    if in_position:
        exit_price = closes[-1]
        pnl_pct = (exit_price - entry_price) / entry_price
        capital += shares * exit_price
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=dates[-1],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            exit_reason="end",
        ))
        equity[-1] = capital

    equity_series = pd.Series(equity, index=dates)
    result = BacktestResult(strategy_name="", ticker="", period="", trades=trades, equity_curve=equity_series)
    result = _compute_metrics(result, equity_series, initial_capital)
    return result


def _compute_metrics(result: BacktestResult, equity: pd.Series, initial_capital: float) -> BacktestResult:
    if len(equity) == 0:
        return result

    years = len(equity) / 252
    final_val = equity.iloc[-1]
    result.cagr = ((final_val / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    # MDD
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    result.mdd = abs(drawdown.min()) * 100

    # Sharpe (daily returns, annualized)
    daily_ret = equity.pct_change().dropna()
    if daily_ret.std() > 0:
        result.sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        result.sharpe = 0.0

    # Win rate and profit factor
    if result.trades:
        result.total_trades = len(result.trades)
        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]
        result.win_rate = (len(wins) / result.total_trades) * 100
        gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        holding_days = []
        for t in result.trades:
            if t.exit_date and t.entry_date:
                holding_days.append((t.exit_date - t.entry_date).days)
        result.avg_holding_days = np.mean(holding_days) if holding_days else 0

    return result
