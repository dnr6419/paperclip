"""
Backtest leveraged ETF strategies on real SOXL/TQQQ historical data (DOF-153).
Downloads via Yahoo Finance chart API, runs all 4 strategy variants,
and includes volatility decay analysis.
"""
import sys, os, json, warnings, time
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.leveraged_etf_oversold import (
    LETFTightTPStrategy,
    LETFMicroScalpStrategy,
    LETFTrendDipStrategy,
    LETFMomentumBurstStrategy,
)
from backtesting.engine import run_backtest

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
    "in_sample":     ("2020-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
}

IC = 100_000


def fetch_yahoo_data(ticker: str, start: str = "2019-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": int(pd.Timestamp(end).timestamp()),
        "interval": "1d",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(3):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to fetch {ticker}: HTTP {resp.status_code}")

    result = resp.json()["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open":   quote["open"],
        "high":   quote["high"],
        "low":    quote["low"],
        "close":  quote["close"],
        "volume": quote["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize())

    df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="first")]
    return df


def analyze_volatility_decay(df: pd.DataFrame, ticker: str) -> dict:
    close = df["close"]
    daily_ret = close.pct_change().dropna()

    ann_vol = daily_ret.std() * np.sqrt(252)
    realized_cagr_years = len(close) / 252
    realized_cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / max(realized_cagr_years, 0.01)) - 1)

    monthly = close.resample("M").last().dropna()
    monthly_ret = monthly.pct_change().dropna()

    rolling_30d_vol = daily_ret.rolling(30).std() * np.sqrt(252)
    rolling_60d_ret = close.pct_change(60)

    high_vol_mask = rolling_30d_vol > rolling_30d_vol.quantile(0.75)
    low_vol_mask = rolling_30d_vol < rolling_30d_vol.quantile(0.25)

    aligned_ret = rolling_60d_ret.reindex(rolling_30d_vol.index)
    high_vol_ret = aligned_ret[high_vol_mask].mean()
    low_vol_ret = aligned_ret[low_vol_mask].mean()

    by_year = {}
    for year in sorted(df.index.year.unique()):
        yr_data = daily_ret[daily_ret.index.year == year]
        if len(yr_data) < 20:
            continue
        yr_close = close[close.index.year == year]
        yr_ret = (yr_close.iloc[-1] / yr_close.iloc[0] - 1) * 100
        yr_vol = yr_data.std() * np.sqrt(252) * 100
        by_year[str(year)] = {
            "return_pct": round(yr_ret, 2),
            "ann_vol_pct": round(yr_vol, 1),
            "max_drawdown_pct": round(
                ((yr_close / yr_close.cummax()) - 1).min() * 100, 1
            ),
        }

    return {
        "ticker": ticker,
        "ann_volatility": round(ann_vol * 100, 1),
        "realized_cagr": round(realized_cagr * 100, 2),
        "avg_60d_ret_high_vol_regime": round(float(high_vol_ret * 100) if pd.notna(high_vol_ret) else 0, 2),
        "avg_60d_ret_low_vol_regime": round(float(low_vol_ret * 100) if pd.notna(low_vol_ret) else 0, 2),
        "decay_spread_pct": round(
            float((low_vol_ret - high_vol_ret) * 100) if pd.notna(high_vol_ret) and pd.notna(low_vol_ret) else 0, 2
        ),
        "by_year": by_year,
    }


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
    print("LEVERAGED ETF BACKTEST — REAL DATA (DOF-153)")
    print("Target: Win Rate 80%+ (Momentum Burst focus)")
    print("=" * 70)

    tickers = ["TQQQ", "SOXL"]
    etf_data = {}

    for ticker in tickers:
        print(f"\nFetching {ticker} from Yahoo Finance...")
        df = fetch_yahoo_data(ticker, "2019-01-01", "2025-01-01")
        etf_data[ticker] = df
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        vol = df["close"].pct_change().std() * np.sqrt(252) * 100
        print(f"  {ticker}: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  ${s:.2f} -> ${e:.2f}, AnnVol={vol:.1f}%, {len(df)} bars")
        time.sleep(2)

    # Volatility decay analysis
    print(f"\n{'='*70}")
    print("VOLATILITY DECAY ANALYSIS")
    print(f"{'='*70}")
    vol_decay = {}
    for ticker, df in etf_data.items():
        analysis = analyze_volatility_decay(df, ticker)
        vol_decay[ticker] = analysis
        print(f"\n  {ticker}:")
        print(f"    Ann. Volatility: {analysis['ann_volatility']}%")
        print(f"    Realized CAGR: {analysis['realized_cagr']}%")
        print(f"    60d avg return in HIGH vol regime: {analysis['avg_60d_ret_high_vol_regime']}%")
        print(f"    60d avg return in LOW vol regime:  {analysis['avg_60d_ret_low_vol_regime']}%")
        print(f"    Decay spread (low-high vol): {analysis['decay_spread_pct']}%")
        print(f"    By year:")
        for yr, stats in analysis["by_year"].items():
            print(f"      {yr}: ret={stats['return_pct']:+.1f}% vol={stats['ann_vol_pct']:.1f}% mdd={stats['max_drawdown_pct']:.1f}%")

    # Backtest
    output = {"volatility_decay": vol_decay}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*70}")
        print(f"[{period_name.upper().replace('_', '-')}] {start} ~ {end}")
        print(f"{'='*70}")

        period_data = {}
        for k, v in etf_data.items():
            sliced = v.loc[start:end]
            if len(sliced) > 0:
                period_data[k] = sliced

        if not period_data:
            print("  No data available for this period")
            output[period_name] = {}
            continue

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

                    # Momentum Burst: monthly trade distribution
                    if strat_name == "D: Momentum Burst" and r.trades:
                        monthly_trades = {}
                        for t in r.trades:
                            m = t.entry_date.strftime("%Y-%m")
                            monthly_trades[m] = monthly_trades.get(m, 0) + 1
                        strat_results[ticker]["monthly_trade_count"] = monthly_trades

            period_results[strat_name] = strat_results

        output[period_name] = period_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — 80%+ Win Rate Strategies (Real Data)")
    print("=" * 70)
    passing = []
    for period_name in PERIODS:
        for strat_name in STRATEGIES:
            for ticker in tickers:
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
                for ticker in tickers:
                    r = output.get(period_name, {}).get(strat_name, {}).get(ticker)
                    if r:
                        all_results.append({
                            "period": period_name,
                            "strategy": strat_name,
                            "ticker": ticker,
                            **r,
                        })
        all_results.sort(key=lambda x: x["win_rate"], reverse=True)
        for p in all_results[:8]:
            print(
                f"  [{p['period']}] {p['strategy']} / {p['ticker']}: "
                f"Win={p['win_rate']:.1f}% Trades={p['total_trades']} "
                f"CAGR={p['cagr']:+.1f}% PF={p['profit_factor']:.2f}"
            )

    output["passing_80pct"] = passing

    # Momentum Burst deep dive
    print("\n" + "=" * 70)
    print("MOMENTUM BURST — DEEP DIVE (Real Data)")
    print("=" * 70)
    mb_summary = {}
    for period_name in PERIODS:
        mb = output.get(period_name, {}).get("D: Momentum Burst", {})
        for ticker in tickers:
            r = mb.get(ticker)
            if r:
                key = f"{period_name}/{ticker}"
                mb_summary[key] = r
                print(
                    f"  [{period_name}] {ticker}: "
                    f"Win={r['win_rate']:.1f}% CAGR={r['cagr']:+.1f}% "
                    f"Sharpe={r['sharpe']:.3f} PF={r['profit_factor']:.2f} "
                    f"MDD={r['mdd']:.1f}% Trades={r['total_trades']}"
                )
    output["momentum_burst_summary"] = mb_summary

    out_path = os.path.join(os.path.dirname(__file__), "leveraged_etf_real_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
