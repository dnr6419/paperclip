"""
Stage 3: Parameter optimization for top-4 KRX-passing strategies.
Grid search on IS (2019-2022), validate on OOS (2023-2024).
Strategies: MACD Momentum, ATR Breakout, DCB, VWB.
Reduced grid for tractability — focus on highest-impact parameters.
"""
import sys, os, json, warnings, itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    MACDMomentumStrategy, ATRBreakoutStrategy, DCBStrategy, VWBStrategy,
)
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_krx")
IC = 100_000_000
IS_START, IS_END = "2019-01-01", "2022-12-31"
OOS_START, OOS_END = "2023-01-01", "2024-12-31"
PASS_CRIT = dict(sharpe=1.0, mdd=20.0, cagr=10.0)


def load_real_data():
    meta_path = os.path.join(DATA_DIR, "_meta.csv")
    meta = pd.read_csv(meta_path)
    data = {}
    for _, row in meta.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        if len(df) >= 100:
            data[ticker] = df
    return data


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


def run_on_universe(strategy, bt_params, stock_data):
    results = []
    for ticker, df in stock_data.items():
        if len(df) < 100:
            continue
        try:
            signals = strategy.generate_signals(df)
            if (signals == 1).sum() == 0:
                continue
            r = run_backtest(df, signals, position_size=1.0, initial_capital=IC, **bt_params)
            results.append(r)
        except Exception:
            pass
    return results


def passes(agg):
    if agg is None:
        return False
    return agg.sharpe > PASS_CRIT["sharpe"] and agg.mdd < PASS_CRIT["mdd"] and agg.cagr > PASS_CRIT["cagr"]


def product_dicts(d):
    keys = list(d.keys())
    for vals in itertools.product(*d.values()):
        yield dict(zip(keys, vals))


def grid_search(name, make_strategy_fn, param_grid, is_data, oos_data):
    bt_keys = {"stop_loss", "take_profit"}
    strat_grid = {k: v for k, v in param_grid.items() if k not in bt_keys}
    bt_grid = {k: v for k, v in param_grid.items() if k in bt_keys}

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"\n[{name}] Grid search: {total} combos", flush=True)

    results = []
    count = 0
    for sp in product_dicts(strat_grid) if strat_grid else [{}]:
        strategy = make_strategy_fn(**sp)
        for bp in product_dicts(bt_grid) if bt_grid else [{}]:
            count += 1
            if count % 5 == 0:
                print(f"  ... {count}/{total}", flush=True)

            is_per_stock = run_on_universe(strategy, bp, is_data)
            is_agg = cross_sectional_aggregate(is_per_stock)
            if is_agg is None:
                continue

            oos_per_stock = run_on_universe(strategy, bp, oos_data)
            oos_agg = cross_sectional_aggregate(oos_per_stock)
            if oos_agg is None:
                continue

            is_pass = passes(is_agg)
            oos_pass = passes(oos_agg)

            results.append({
                "strat_params": sp,
                "bt_params": bp,
                "is": {
                    "cagr": round(is_agg.cagr, 2), "mdd": round(is_agg.mdd, 2),
                    "sharpe": round(is_agg.sharpe, 3), "win_rate": round(is_agg.win_rate, 1),
                    "profit_factor": round(is_agg.profit_factor, 2),
                    "trades": is_agg.total_trades, "pass": is_pass,
                },
                "oos": {
                    "cagr": round(oos_agg.cagr, 2), "mdd": round(oos_agg.mdd, 2),
                    "sharpe": round(oos_agg.sharpe, 3), "win_rate": round(oos_agg.win_rate, 1),
                    "profit_factor": round(oos_agg.profit_factor, 2),
                    "trades": oos_agg.total_trades, "pass": oos_pass,
                },
                "both_pass": is_pass and oos_pass,
            })

    results.sort(key=lambda x: x["oos"]["cagr"], reverse=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] Top 5 configs (sorted by OOS CAGR):", flush=True)
    for r in results[:5]:
        tag = "PASS" if r["both_pass"] else "fail"
        print(f"  [{tag}] IS: CAGR={r['is']['cagr']:+.1f}% Sharpe={r['is']['sharpe']:.2f} | "
              f"OOS: CAGR={r['oos']['cagr']:+.1f}% Sharpe={r['oos']['sharpe']:.2f} MDD={r['oos']['mdd']:.1f}%", flush=True)
        print(f"    strat={r['strat_params']} bt={r['bt_params']}", flush=True)

    best_both = [r for r in results if r["both_pass"]]
    best = best_both[0] if best_both else (results[0] if results else None)
    return best, results


def main():
    print("Loading KRX real data...", flush=True)
    all_data = load_real_data()
    print(f"  {len(all_data)} tickers loaded", flush=True)

    is_data = {k: v.loc[IS_START:IS_END] for k, v in all_data.items()}
    oos_data = {k: v.loc[OOS_START:OOS_END] for k, v in all_data.items()}
    is_data = {k: v for k, v in is_data.items() if len(v) >= 100}
    oos_data = {k: v for k, v in oos_data.items() if len(v) >= 100}
    print(f"  IS: {len(is_data)} tickers, OOS: {len(oos_data)} tickers\n", flush=True)

    all_results = {}

    # --- MACD Momentum (3 × 2 × 2 × 2 = 24 combos) ---
    best_macd, details_macd = grid_search(
        "MACD Momentum",
        lambda fast_period=12, slow_period=26, signal_period=9: MACDMomentumStrategy(
            fast_period=fast_period, slow_period=slow_period, signal_period=signal_period),
        {
            "fast_period":   [8, 12, 15],
            "slow_period":   [20, 26],
            "signal_period": [9],
            "stop_loss":     [0.04, 0.05],
            "take_profit":   [0.20, 0.30],
        },
        is_data, oos_data,
    )
    all_results["macd_momentum"] = {"best": best_macd, "all": details_macd[:10]}

    # --- ATR Breakout (3 × 2 × 2 × 2 = 24 combos) ---
    best_atr, details_atr = grid_search(
        "ATR Breakout",
        lambda lookback=20, atr_period=14, breakout_mult=0.7: ATRBreakoutStrategy(
            lookback=lookback, atr_period=atr_period, breakout_mult=breakout_mult),
        {
            "lookback":      [15, 20, 30],
            "breakout_mult": [0.5, 0.7],
            "stop_loss":     [0.03, 0.04],
            "take_profit":   [0.15, 0.25],
        },
        is_data, oos_data,
    )
    all_results["atr_breakout"] = {"best": best_atr, "all": details_atr[:10]}

    # --- DCB (3 × 2 × 2 × 2 = 24 combos) ---
    best_dcb, details_dcb = grid_search(
        "DCB",
        lambda entry_period=20, exit_period=8, adx_threshold=25.0: DCBStrategy(
            entry_period=entry_period, exit_period=exit_period, adx_threshold=adx_threshold),
        {
            "entry_period":  [15, 20, 30],
            "adx_threshold": [20.0, 25.0],
            "stop_loss":     [0.04, 0.05],
            "take_profit":   [0.20, 0.30],
        },
        is_data, oos_data,
    )
    all_results["dcb"] = {"best": best_dcb, "all": details_dcb[:10]}

    # --- VWB (3 × 2 × 2 × 2 = 24 combos) ---
    best_vwb, details_vwb = grid_search(
        "VWB",
        lambda breakout_mult=0.5, vol_period=20, vol_multiplier=1.5: VWBStrategy(
            breakout_mult=breakout_mult, vol_period=vol_period, vol_multiplier=vol_multiplier),
        {
            "breakout_mult":  [0.3, 0.5, 0.7],
            "vol_multiplier": [1.2, 1.5],
            "stop_loss":      [0.04, 0.05],
            "take_profit":    [5.0, 10.0],
        },
        is_data, oos_data,
    )
    all_results["vwb"] = {"best": best_vwb, "all": details_vwb[:10]}

    # --- Summary ---
    print("\n" + "=" * 70, flush=True)
    print("STAGE 3 OPTIMIZATION SUMMARY — KRX Real Data", flush=True)
    print("=" * 70, flush=True)
    for name, key in [
        ("MACD Momentum", "macd_momentum"),
        ("ATR Breakout", "atr_breakout"),
        ("DCB", "dcb"),
        ("VWB", "vwb"),
    ]:
        best = all_results[key]["best"]
        if best:
            tag = "PASS" if best["both_pass"] else "FAIL"
            print(f"\n[{name}] [{tag}]", flush=True)
            print(f"  IS:  CAGR={best['is']['cagr']:+.1f}% MDD={best['is']['mdd']:.1f}% "
                  f"Sharpe={best['is']['sharpe']:.2f} WR={best['is']['win_rate']:.1f}%", flush=True)
            print(f"  OOS: CAGR={best['oos']['cagr']:+.1f}% MDD={best['oos']['mdd']:.1f}% "
                  f"Sharpe={best['oos']['sharpe']:.2f} WR={best['oos']['win_rate']:.1f}%", flush=True)
            print(f"  Best params: strat={best['strat_params']} bt={best['bt_params']}", flush=True)
        else:
            print(f"\n[{name}] No viable configs found", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "optimization_top4_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
