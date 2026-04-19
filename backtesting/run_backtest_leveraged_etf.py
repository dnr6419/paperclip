"""
Backtest runner for leveraged ETF short-term strategies (DOF-151).
Tests 3 strategy variants on SOXL/TQQQ synthetic data.
Target: 80%+ win rate with conservative entry filters.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.leveraged_etf_oversold import (
    LETFTightTPStrategy,
    LETFMicroScalpStrategy,
    LETFTrendDipStrategy,
    LETFMomentumBurstStrategy,
)
from backtesting.engine import run_backtest
from backtesting.generate_data_leveraged_etf import generate_leveraged_etf_data

STRATEGIES = {
    "A: Tight TP Uptrend Dip": (
        LETFTightTPStrategy(),
        {"stop_loss": 0.10, "take_profit": 0.02},
    ),
    "B: Micro Scalp": (
        LETFMicroScalpStrategy(),
        {"stop_loss": 0.12, "take_profit": 0.015},
    ),
    "C: Trend Dip RSI(5)": (
        LETFTrendDipStrategy(),
        {"stop_loss": 0.08, "take_profit": 0.025},
    ),
    "D: Momentum Burst": (
        LETFMomentumBurstStrategy(),
        {"stop_loss": 0.10, "take_profit": 0.015},
    ),
}

PERIODS = {
    "in_sample":     ("2019-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
}

IC = 100_000


def run_single(strategy, params, df, ticker, period_name):
    signals = strategy.generate_signals(df)
    buy_count = (signals == 1).sum()
    if buy_count == 0:
        return None
    r = run_backtest(df, signals, position_size=1.0, initial_capital=IC, **params)
    r.strategy_name = ""
    r.ticker = ticker
    r.period = period_name
    return r


def main():
    print("=" * 70)
    print("LEVERAGED ETF SHORT-TERM STRATEGY BACKTEST (DOF-151)")
    print("Target: Win Rate 80%+")
    print("=" * 70)

    print("\nGenerating synthetic leveraged ETF data (2019-2024)...")
    etf_data = generate_leveraged_etf_data("2019-01-01", "2024-12-31")
    for name, df in etf_data.items():
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        vol = df["close"].pct_change().std() * np.sqrt(252) * 100
        print(f"  {name}: ${s:.2f} -> ${e:.2f}, AnnVol={vol:.1f}%")

    output = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*70}")
        print(f"[{period_name.upper().replace('_', '-')}] {start} ~ {end}")
        print(f"{'='*70}")

        period_data = {k: v.loc[start:end] for k, v in etf_data.items()}
        period_results = {}

        for strat_name, (strategy, params) in STRATEGIES.items():
            print(f"\n  Strategy: {strat_name}")
            strat_results = {}

            for ticker, df in period_data.items():
                r = run_single(strategy, params, df, ticker, period_name)
                if r is None:
                    print(f"    {ticker}: no signals")
                    strat_results[ticker] = None
                    continue

                win_flag = "✓" if r.win_rate >= 80 else "✗"
                print(
                    f"    {ticker}: Win={r.win_rate:.1f}% {win_flag} "
                    f"CAGR={r.cagr:+.1f}% MDD={r.mdd:.1f}% "
                    f"Sharpe={r.sharpe:.2f} PF={r.profit_factor:.2f} "
                    f"Trades={r.total_trades} AvgHold={r.avg_holding_days:.1f}d"
                )

                strat_results[ticker] = {
                    "cagr": round(r.cagr, 2),
                    "mdd": round(r.mdd, 2),
                    "sharpe": round(r.sharpe, 3),
                    "win_rate": round(r.win_rate, 1),
                    "profit_factor": round(min(r.profit_factor, 99.0), 2),
                    "total_trades": r.total_trades,
                    "avg_holding_days": round(r.avg_holding_days, 1),
                }

                if r.trades:
                    wins = [t for t in r.trades if t.pnl_pct > 0]
                    losses = [t for t in r.trades if t.pnl_pct <= 0]
                    if wins:
                        avg_win = np.mean([t.pnl_pct for t in wins]) * 100
                        strat_results[ticker]["avg_win_pct"] = round(avg_win, 2)
                    if losses:
                        avg_loss = np.mean([t.pnl_pct for t in losses]) * 100
                        strat_results[ticker]["avg_loss_pct"] = round(avg_loss, 2)

                    by_reason = {}
                    for t in r.trades:
                        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
                    strat_results[ticker]["exit_reasons"] = by_reason

            period_results[strat_name] = strat_results

        output[period_name] = period_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — 80%+ Win Rate Strategies")
    print("=" * 70)
    passing = []
    for period_name in PERIODS:
        for strat_name in STRATEGIES:
            for ticker in etf_data:
                r = output.get(period_name, {}).get(strat_name, {}).get(ticker)
                if r and r["win_rate"] >= 80:
                    passing.append({
                        "period": period_name,
                        "strategy": strat_name,
                        "ticker": ticker,
                        **r,
                    })

    if passing:
        for p in passing:
            print(
                f"  [{p['period']}] {p['strategy']} / {p['ticker']}: "
                f"Win={p['win_rate']:.1f}% Trades={p['total_trades']} "
                f"CAGR={p['cagr']:+.1f}% PF={p['profit_factor']:.2f}"
            )
    else:
        print("  No strategy-ticker combinations achieved 80%+ win rate.")
        print("  Listing closest results:")
        all_results = []
        for period_name in PERIODS:
            for strat_name in STRATEGIES:
                for ticker in etf_data:
                    r = output.get(period_name, {}).get(strat_name, {}).get(ticker)
                    if r:
                        all_results.append({
                            "period": period_name,
                            "strategy": strat_name,
                            "ticker": ticker,
                            **r,
                        })
        all_results.sort(key=lambda x: x["win_rate"], reverse=True)
        for p in all_results[:6]:
            print(
                f"  [{p['period']}] {p['strategy']} / {p['ticker']}: "
                f"Win={p['win_rate']:.1f}% Trades={p['total_trades']} "
                f"CAGR={p['cagr']:+.1f}% PF={p['profit_factor']:.2f}"
            )

    output["passing_80pct"] = passing

    out_path = os.path.join(os.path.dirname(__file__), "leveraged_etf_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
