"""
Parameter grid search for top-tier strategies: 52W Breakout, EMA Crossover, ADX Trend.
Target: maximize CAGR while keeping MDD < 20%, Sharpe > 1.0 in OOS (2023-2024).
"""
import sys, os, warnings, inspect
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    EMACrossoverStrategy, ADXTrendStrategy, High52WBreakoutStrategy,
)
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics
from backtesting.generate_data import generate_all_data

IC = 100_000
OOS_START, OOS_END = "2023-01-01", "2024-12-31"
PASS_CRIT = dict(sharpe=1.0, mdd=20.0, cagr=10.0)


def cross_sectional_aggregate(per_stock_results):
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
    agg = BacktestResult(strategy_name="", ticker="portfolio", period="", trades=all_trades)
    agg.equity_curve = combined
    return _compute_metrics(agg, combined, IC)


def run_on_universe(strategy, params, stock_data, sp500_data):
    results = []
    for ticker, df in stock_data.items():
        if len(df) < 100:
            continue
        try:
            sig_params = inspect.signature(strategy.generate_signals)
            if "market_close" in sig_params.parameters:
                mc = sp500_data["close"].reindex(df.index, method="ffill") if sp500_data is not None else None
                signals = strategy.generate_signals(df, market_close=mc)
            else:
                signals = strategy.generate_signals(df)
            if (signals == 1).sum() == 0:
                continue
            r = run_backtest(df, signals, position_size=1.0, initial_capital=IC, **params)
            results.append(r)
        except Exception:
            pass
    return results


def passes(agg):
    return agg.sharpe > PASS_CRIT["sharpe"] and agg.mdd < PASS_CRIT["mdd"] and agg.cagr > PASS_CRIT["cagr"]


def grid_search(name, make_strategy_fn, param_grid, stock_data, sp500_data):
    """Exhaustively search strategy constructor params × backtest params."""
    best = None
    results = []

    bt_keys = ["stop_loss", "take_profit"]
    strat_params_grid = {k: param_grid[k] for k in param_grid if k not in bt_keys}
    bt_params_grid = {k: param_grid[k] for k in bt_keys if k in param_grid}

    def product_dicts(d):
        import itertools
        keys = list(d.keys())
        for vals in itertools.product(*d.values()):
            yield dict(zip(keys, vals))

    for sp in product_dicts(strat_params_grid) if strat_params_grid else [{}]:
        strategy = make_strategy_fn(**sp)
        for bp in product_dicts(bt_params_grid) if bt_params_grid else [{}]:
            per_stock = run_on_universe(strategy, bp, stock_data, sp500_data)
            agg = cross_sectional_aggregate(per_stock)
            if agg is None:
                continue
            label = f"strat={sp} bt={bp}"
            status = "PASS" if passes(agg) else "fail"
            results.append({
                "label": label, "cagr": agg.cagr, "mdd": agg.mdd,
                "sharpe": agg.sharpe, "win_rate": agg.win_rate,
                "trades": agg.total_trades, "status": status,
                "strat_params": sp, "bt_params": bp,
            })
            if best is None or agg.cagr > best["cagr"]:
                best = results[-1]

    results.sort(key=lambda x: x["cagr"], reverse=True)
    print(f"\n{'='*60}")
    print(f"[{name}] Top 5 configs (OOS):")
    for r in results[:5]:
        print(f"  [{r['status']}] CAGR={r['cagr']:+.1f}% MDD={r['mdd']:.1f}% "
              f"Sharpe={r['sharpe']:.2f} Trades={r['trades']}")
        print(f"    {r['label']}")
    return best, results


def main():
    print("Generating synthetic data (2019-2024)...")
    all_data = generate_all_data("2019-01-01", "2024-12-31")
    sp500_full = all_data.pop("SP500")

    oos_data = {k: v.loc[OOS_START:OOS_END] for k, v in all_data.items()}
    sp500_oos = sp500_full.loc[OOS_START:OOS_END]
    print(f"  {len(oos_data)} stocks, OOS period only\n")

    # --- 52W Breakout (best performer) ---
    best_52w, _ = grid_search(
        "52W Breakout",
        lambda lookback=252, vol_multiplier=2.0: High52WBreakoutStrategy(
            lookback=lookback, vol_multiplier=vol_multiplier),
        {
            "lookback":       [200, 252],
            "vol_multiplier": [1.5, 2.0],
            "stop_loss":      [0.06, 0.08],
            "take_profit":    [0.15, 0.20, 0.25],
        },
        oos_data, sp500_oos,
    )

    # --- EMA Crossover (most stable) ---
    best_ema, _ = grid_search(
        "EMA Crossover",
        lambda fast=12, slow=26, vol_multiplier=1.5: EMACrossoverStrategy(fast, slow, vol_multiplier),
        {
            "fast":           [10, 12, 15, 20],
            "slow":           [20, 26, 30, 50],
            "vol_multiplier": [1.2, 1.5],
            "stop_loss":      [0.04, 0.05],
            "take_profit":    [0.10, 0.15],
        },
        oos_data, sp500_oos,
    )

    # --- ADX Trend (highest potential) ---
    best_adx, _ = grid_search(
        "ADX Trend",
        lambda adx_threshold=25, sma_period=50, vol_threshold=1.5: ADXTrendStrategy(
            adx_threshold=adx_threshold, sma_period=sma_period, vol_threshold=vol_threshold),
        {
            "adx_threshold": [20, 22, 25],
            "sma_period":    [20, 50, 100],
            "vol_threshold": [1.2, 1.5],
            "stop_loss":     [0.04, 0.05],
            "take_profit":   [0.15, 0.20],
        },
        oos_data, sp500_oos,
    )

    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    for name, best in [
        ("52W Breakout", best_52w),
        ("EMA Crossover", best_ema),
        ("ADX Trend", best_adx),
    ]:
        if best:
            status = "PASS" if passes(best) else "fail"
            print(f"\n[{name}] Best OOS: CAGR={best['cagr']:+.1f}% MDD={best['mdd']:.1f}% "
                  f"Sharpe={best['sharpe']:.2f} [{status}]")
            print(f"  strat_params: {best['strat_params']}")
            print(f"  bt_params:    {best['bt_params']}")

    return {"breakout_52w": best_52w, "ema": best_ema, "adx": best_adx}


if __name__ == "__main__":
    main()
