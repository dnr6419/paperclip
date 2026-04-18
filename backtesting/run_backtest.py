"""
Main backtest runner: runs all 8 strategies on synthetic Korean market data
for In-Sample (2019-2022) and Out-of-Sample (2023-2024) periods.
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    EMACrossoverStrategy, RSIReversalStrategy, MACDDivergenceStrategy,
    BBSqueezeStrategy, CandleRSIStrategy, ADXTrendStrategy,
    MeanReversionStrategy, High52WBreakoutStrategy,
)
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics
from backtesting.generate_data import generate_all_data

STRATEGIES = {
    "EMA Crossover":    (EMACrossoverStrategy(),    {"stop_loss": 0.04, "take_profit": 0.10, "position_size": 0.10}),
    "RSI Reversal":     (RSIReversalStrategy(),     {"stop_loss": 0.05, "take_profit": 0.07, "position_size": 0.10}),
    "MACD Divergence":  (MACDDivergenceStrategy(),  {"stop_loss": 0.05, "take_profit": 0.10, "position_size": 0.10}),
    "BB Squeeze":       (BBSqueezeStrategy(),       {"stop_loss": 0.05, "take_profit": 0.10, "position_size": 0.10}),
    "Candle+RSI":       (CandleRSIStrategy(),       {"stop_loss": 0.04, "take_profit": 0.08, "position_size": 0.10}),
    "ADX Trend":        (ADXTrendStrategy(),        {"stop_loss": 0.05, "take_profit": 0.12, "position_size": 0.10}),
    "Mean Reversion":   (MeanReversionStrategy(),   {"stop_loss": 0.04, "take_profit": 0.06, "position_size": 0.10}),
    "52W Breakout":     (High52WBreakoutStrategy(), {"stop_loss": 0.08, "take_profit": 0.15, "position_size": 0.10}),
}

PERIODS = {
    "in_sample":     ("2019-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
}

INITIAL_CAPITAL = 10_000_000


def aggregate_portfolio_results(per_stock_results):
    """Equal-weight portfolio aggregation from per-stock results."""
    if not per_stock_results:
        return None

    all_equity = []
    all_trades = []
    for r in per_stock_results:
        if r.equity_curve is not None and len(r.equity_curve) > 0:
            norm = r.equity_curve / r.equity_curve.iloc[0]
            all_equity.append(norm)
        all_trades.extend(r.trades)

    if not all_equity:
        return None

    combined = pd.concat(all_equity, axis=1).mean(axis=1)
    equity = combined * INITIAL_CAPITAL

    agg = BacktestResult(strategy_name="", ticker="portfolio", period="")
    agg.trades = all_trades
    agg.equity_curve = equity
    agg = _compute_metrics(agg, equity, INITIAL_CAPITAL)
    return agg


def run_strategy_on_universe(strategy, params, stock_data, kospi_data):
    results = []
    for ticker, df in stock_data.items():
        if ticker == "KOSPI":
            continue
        try:
            import inspect
            sig = inspect.signature(strategy.generate_signals)
            if "market_close" in sig.parameters:
                market_close = kospi_data["close"].reindex(df.index, method="ffill")
                signals = strategy.generate_signals(df, market_close=market_close)
            else:
                signals = strategy.generate_signals(df)

            if signals.abs().sum() == 0:
                continue

            r = run_backtest(df, signals, **params)
            r.strategy_name = params.get("_name", "")
            r.ticker = ticker
            if r.total_trades >= 2:
                results.append(r)
        except Exception as e:
            pass
    return results


def main():
    print("=" * 60)
    print("BACKTESTING RUNNER — 8 Strategies, Synthetic KOSPI Universe")
    print("=" * 60)

    print("\nGenerating synthetic market data (2019-2024)...", flush=True)
    all_data = generate_all_data("2019-01-01", "2024-12-31")
    print(f"  {len(all_data)-1} stocks + KOSPI index generated")

    all_results = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n[{period_name.upper().replace('_', '-')}] {start} ~ {end}")
        stock_data = {k: v.loc[start:end] for k, v in all_data.items() if len(v.loc[start:end]) >= 60}
        kospi_data = stock_data.pop("KOSPI", None)

        period_metrics = {}
        for strat_name, (strategy, params) in STRATEGIES.items():
            print(f"  Running {strat_name}...", flush=True, end="")
            run_params = {k: v for k, v in params.items()}
            results = run_strategy_on_universe(strategy, run_params, stock_data, kospi_data)
            agg = aggregate_portfolio_results(results)
            if agg:
                agg.strategy_name = strat_name
                period_metrics[strat_name] = agg
                verdict = "PASS" if (agg.sharpe > 1.0 and agg.mdd < 20 and agg.cagr > 10) else "fail"
                print(f" CAGR={agg.cagr:+.1f}% MDD={agg.mdd:.1f}% Sharpe={agg.sharpe:.2f} "
                      f"Win={agg.win_rate:.1f}% PF={agg.profit_factor:.2f} Trades={agg.total_trades} [{verdict}]")
            else:
                print(" no trades")

        all_results[period_name] = period_metrics

    return all_results


if __name__ == "__main__":
    results = main()

    output = {}
    for period, metrics in results.items():
        output[period] = {}
        for strat, r in metrics.items():
            output[period][strat] = {
                "cagr": round(r.cagr, 2),
                "mdd": round(r.mdd, 2),
                "sharpe": round(r.sharpe, 3),
                "win_rate": round(r.win_rate, 1),
                "profit_factor": round(r.profit_factor, 2),
                "total_trades": r.total_trades,
                "avg_holding_days": round(r.avg_holding_days, 1),
            }

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
