"""
ADX Trend Following Momentum backtest.
In-Sample: 2019-2022, Out-of-Sample: 2023-2024.
Parameter optimization: adx_period, adx_threshold, sma_period, stop_loss, take_profit.
"""
import sys, os, json, warnings, itertools
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.adx_trend import ADXTrendStrategy
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics
from backtesting.generate_data import generate_all_data

IC = 100_000
PERIODS = {
    "in_sample":     ("2019-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
}

PARAM_GRID = {
    "adx_period":     [10, 14, 20],
    "adx_threshold":  [15, 20, 25],
    "sma_period":     [30, 50, 70],
    "stop_loss":      [0.04, 0.05, 0.07],
    "take_profit":    [0.15, 0.20, 0.25],
}


def cross_sectional_aggregate(per_stock_results, strategy_name="ADX Trend"):
    if not per_stock_results:
        return None
    all_eq, all_trades = [], []
    for r in per_stock_results:
        if r.equity_curve is not None and len(r.equity_curve) > 0 and r.equity_curve.iloc[0] > 0:
            all_eq.append(r.equity_curve / r.equity_curve.iloc[0])
        all_trades.extend(r.trades)
    if not all_eq:
        return None
    combined = pd.concat(all_eq, axis=1).mean(axis=1) * IC
    agg = BacktestResult(strategy_name=strategy_name, ticker="portfolio", period="", trades=all_trades)
    agg.equity_curve = combined
    return _compute_metrics(agg, combined, IC)


def run_on_universe(strategy, stock_data):
    results = []
    for ticker, df in stock_data.items():
        if len(df) < 250:
            continue
        try:
            signals = strategy.generate_signals(df)
            if (signals == 1).sum() == 0:
                continue
            params = strategy.get_signal_params()
            r = run_backtest(
                df, signals,
                position_size=1.0,
                initial_capital=IC,
                stop_loss=params.stop_loss,
                take_profit=params.take_profit,
            )
            r.ticker = ticker
            results.append(r)
        except Exception:
            pass
    return results


def optimize(stock_data):
    best_score, best_params = -999, None
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    print(f"  Optimizing over {len(combos)} parameter combinations...")
    for combo in combos:
        params = dict(zip(keys, combo))
        sl = params.pop("stop_loss")
        tp = params.pop("take_profit")
        strategy = ADXTrendStrategy(**params)
        strategy._override_sl = sl
        strategy._override_tp = tp
        strategy.get_signal_params = lambda s=strategy, sl_=sl, tp_=tp: type(
            strategy.get_signal_params()
        )(direction=1, stop_loss=sl_, take_profit=tp_, position_size=0.025 / sl_)
        results = run_on_universe(strategy, stock_data)
        agg = cross_sectional_aggregate(results)
        if agg is None:
            continue
        if agg.total_trades < 20:
            continue
        score = agg.sharpe
        if score > best_score:
            best_score = score
            best_params = {**params, "stop_loss": sl, "take_profit": tp}
    return best_params, best_score


def main():
    print("=" * 60)
    print("ADX Trend Following Momentum — Backtest")
    print("=" * 60)
    print("\nGenerating synthetic market data (2019-2024)...", flush=True)
    all_data = generate_all_data("2019-01-01", "2024-12-31")
    all_data.pop("SP500", None)
    print(f"  {len(all_data)} stocks generated")

    print("\n[IN-SAMPLE OPTIMIZATION] 2019-2022")
    is_data = {k: v.loc["2019-01-01":"2022-12-31"] for k, v in all_data.items()}
    best_params, best_score = optimize(is_data)

    if best_params is None:
        print("  No valid parameter set found. Using defaults.")
        best_params = {
            "adx_period": 14,
            "adx_threshold": 15,
            "sma_period": 50,
            "stop_loss": 0.04,
            "take_profit": 0.25,
        }

    print(f"\n  Best params: {best_params}  (Sharpe={best_score:.3f})")

    output = {"best_params": best_params}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n[{period_name.upper().replace('_', '-')}] {start} ~ {end}")
        period_data = {k: v.loc[start:end] for k, v in all_data.items()}
        sl = best_params["stop_loss"]
        tp = best_params["take_profit"]
        strat_params = {k: v for k, v in best_params.items() if k not in ("stop_loss", "take_profit")}
        strategy = ADXTrendStrategy(**strat_params)
        from strategies.base import Signal
        strategy.get_signal_params = lambda sl_=sl, tp_=tp: Signal(
            direction=1, stop_loss=sl_, take_profit=tp_, position_size=0.025 / sl_
        )
        results = run_on_universe(strategy, period_data)
        agg = cross_sectional_aggregate(results)
        if agg:
            pass_fail = "PASS" if (agg.sharpe > 1.0 and agg.mdd < 20.0 and agg.cagr > 10.0) else "fail"
            print(f"  CAGR={agg.cagr:+.1f}% MDD={agg.mdd:.1f}% Sharpe={agg.sharpe:.2f}"
                  f" Win={agg.win_rate:.1f}% PF={agg.profit_factor:.2f}"
                  f" Trades={agg.total_trades} [{pass_fail}]")
            output[period_name] = {
                "cagr": round(agg.cagr, 2),
                "mdd": round(agg.mdd, 2),
                "sharpe": round(agg.sharpe, 3),
                "win_rate": round(agg.win_rate, 1),
                "profit_factor": round(min(agg.profit_factor, 99.0), 2),
                "total_trades": agg.total_trades,
                "avg_holding_days": round(agg.avg_holding_days, 1),
            }
        else:
            print("  no signals")
            output[period_name] = None

    within_60d = True
    for r_key in ["in_sample", "out_of_sample"]:
        if output.get(r_key) and output[r_key].get("avg_holding_days", 999) > 60:
            within_60d = False
    output["within_60_day_holding"] = within_60d

    out_path = os.path.join(os.path.dirname(__file__), "adx_trend_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    main()
