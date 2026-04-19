"""
VIX Spike Reversion — Portfolio Overlay Analysis (DOF-180)

Analyses Strategy 13 (VIX Spike Reversion) as a hedge overlay on top of the
5-strategy core portfolio:
  - DCB
  - Candle+RSI
  - MTM
  - MACD Momentum
  - ATR Breakout

Steps:
  1. Run each of the 5 core strategies independently → equity curves
  2. Run VIX Spike Reversion → equity curve
  3. Compute pairwise correlation matrix of daily returns
  4. Sweep overlay weights [0%, 5%, 10%, 15%, 20%] for VIX strategy
  5. Compute blended portfolio metrics at each weight
  6. Report optimal weight and save → vix_overlay_results.json
"""

import sys, os, json, warnings, inspect
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    DCBStrategy, CandleRSIStrategy, MTMStrategy,
    MACDMomentumStrategy, ATRBreakoutStrategy,
)
from strategies.vix_spike_reversion import VIXSpikeReversionStrategy
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics
from backtesting.generate_data import generate_all_data

IC = 100_000

CORE_STRATEGIES = {
    "DCB":           (DCBStrategy(),          {"stop_loss": 0.05, "take_profit": 0.25}),
    "Candle+RSI":    (CandleRSIStrategy(),    {"stop_loss": 0.04, "take_profit": 0.20}),
    "MTM":           (MTMStrategy(),          {"stop_loss": 0.05, "take_profit": 0.20}),
    "MACD Momentum": (MACDMomentumStrategy(), {"stop_loss": 0.05, "take_profit": 0.25}),
    "ATR Breakout":  (ATRBreakoutStrategy(),  {"stop_loss": 0.04, "take_profit": 0.20}),
}

VIX_STRATEGY = VIXSpikeReversionStrategy()

OVERLAY_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]

PERIODS = {
    "in_sample":     ("2019-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
    "full":          ("2019-01-01", "2024-12-31"),
}


def run_strategy_universe(strategy, params, stock_data, sp500_data=None):
    """Run strategy on all stocks, return list of (ticker, equity_curve, metrics)."""
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
            bt_params = dict(params)
            if hasattr(strategy, "last_atr") and strategy.last_atr is not None:
                bt_params["trailing_atr_series"] = strategy.last_atr
                bt_params["trailing_atr_mult"]   = strategy.atr_mult
            r = run_backtest(df, signals, position_size=1.0, initial_capital=IC, **bt_params)
            r.strategy_name = ""
            r.ticker = ticker
            results.append(r)
        except Exception:
            pass
    return results


def aggregate_equity_curve(per_stock_results):
    """Cross-sectional mean-normalised equity curve."""
    curves = []
    for r in per_stock_results:
        if r.equity_curve is not None and len(r.equity_curve) > 0 and r.equity_curve.iloc[0] > 0:
            curves.append(r.equity_curve / r.equity_curve.iloc[0])
    if not curves:
        return None
    return pd.concat(curves, axis=1).mean(axis=1) * IC


def aggregate_metrics(per_stock_results, equity_curve):
    if equity_curve is None or len(equity_curve) == 0:
        return None
    agg = BacktestResult(strategy_name="", ticker="portfolio", period="", trades=[])
    for r in per_stock_results:
        agg.trades.extend(r.trades)
    agg.equity_curve = equity_curve
    return _compute_metrics(agg, equity_curve, IC)


def compute_correlation_matrix(equity_curves: dict) -> pd.DataFrame:
    """Compute Pearson correlation matrix of daily returns across strategies."""
    daily_returns = {}
    for name, curve in equity_curves.items():
        if curve is not None and len(curve) > 1:
            daily_returns[name] = curve.pct_change().dropna()
    df = pd.DataFrame(daily_returns)
    return df.corr()


def blend_portfolio(core_curves: dict, vix_curve, vix_weight: float):
    """
    Blend core portfolio (equal-weight within core) with VIX overlay.
    core_weight = 1 - vix_weight, split equally among core strategies.
    """
    if vix_curve is None:
        return None

    n_core = len(core_curves)
    core_w = (1.0 - vix_weight) / n_core

    # Align all curves on common index
    all_curves = {}
    for name, curve in core_curves.items():
        if curve is not None:
            all_curves[name] = curve / curve.iloc[0]
    all_curves["VIX_Spike"] = vix_curve / vix_curve.iloc[0]

    combined = pd.DataFrame(all_curves).dropna()
    if combined.empty:
        return None

    weights = {name: core_w for name in core_curves if name in combined.columns}
    if "VIX_Spike" in combined.columns:
        weights["VIX_Spike"] = vix_weight

    blended = sum(combined[name] * w for name, w in weights.items()) * IC
    return blended


def metrics_from_curve(curve, trades=None):
    """Compute basic metrics from equity curve."""
    if curve is None or len(curve) < 2:
        return None
    ret = curve.pct_change().dropna()
    total_return = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
    n_years = len(curve) / 252
    cagr = ((curve.iloc[-1] / curve.iloc[0]) ** (1.0 / n_years) - 1) * 100 if n_years > 0 else 0
    vol = ret.std() * np.sqrt(252) * 100
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    roll_max = curve.cummax()
    mdd = ((curve - roll_max) / roll_max).min() * 100
    return {
        "cagr": round(cagr, 2),
        "total_return_pct": round(total_return, 2),
        "annual_vol_pct": round(vol, 2),
        "sharpe": round(sharpe, 3),
        "mdd": round(mdd, 2),
    }


def main():
    print("=" * 70)
    print("VIX SPIKE REVERSION — Portfolio Overlay Analysis (DOF-180)")
    print("=" * 70)

    print("\nGenerating synthetic market data (2019-2024)...", flush=True)
    all_data = generate_all_data("2019-01-01", "2024-12-31")
    sp500_full = all_data.pop("SP500")
    print(f"  {len(all_data)} stocks generated")

    output = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*70}")
        print(f"[{period_name.upper()}] {start} ~ {end}")
        print(f"{'='*70}")

        stock_data = {k: v.loc[start:end] for k, v in all_data.items()}
        sp500 = sp500_full.loc[start:end]

        # ── Step 1: Run core strategies ──────────────────────────────────────
        print("\nRunning core strategies...")
        core_curves = {}
        core_metrics = {}
        for strat_name, (strategy, params) in CORE_STRATEGIES.items():
            results = run_strategy_universe(strategy, params, stock_data, sp500)
            curve = aggregate_equity_curve(results)
            m = aggregate_metrics(results, curve)
            core_curves[strat_name] = curve
            if m:
                core_metrics[strat_name] = {
                    "cagr": round(m.cagr, 2),
                    "mdd": round(m.mdd, 2),
                    "sharpe": round(m.sharpe, 3),
                    "win_rate": round(m.win_rate, 1),
                    "total_trades": m.total_trades,
                }
                print(f"  {strat_name:<16} CAGR={m.cagr:+.1f}%  MDD={m.mdd:.1f}%  Sharpe={m.sharpe:.2f}")
            else:
                core_metrics[strat_name] = None
                print(f"  {strat_name:<16} no signals")

        # ── Step 2: Run VIX Spike Reversion ─────────────────────────────────
        print("\nRunning VIX Spike Reversion...")
        vix_results = run_strategy_universe(VIX_STRATEGY, {"stop_loss": 0.04, "take_profit": 0.08}, stock_data, sp500)
        vix_curve = aggregate_equity_curve(vix_results)
        vix_m = aggregate_metrics(vix_results, vix_curve)
        if vix_m:
            vix_metrics = {
                "cagr": round(vix_m.cagr, 2),
                "mdd": round(vix_m.mdd, 2),
                "sharpe": round(vix_m.sharpe, 3),
                "win_rate": round(vix_m.win_rate, 1),
                "total_trades": vix_m.total_trades,
            }
            print(f"  {'VIX Spike Reversion':<16} CAGR={vix_m.cagr:+.1f}%  MDD={vix_m.mdd:.1f}%  Sharpe={vix_m.sharpe:.2f}")
        else:
            vix_metrics = None
            print("  VIX Spike Reversion: no signals")

        # ── Step 3: Correlation matrix ───────────────────────────────────────
        all_equity_curves = dict(core_curves)
        all_equity_curves["VIX Spike Reversion"] = vix_curve

        corr_matrix = compute_correlation_matrix(all_equity_curves)
        print("\nCorrelation matrix (daily returns):")
        print(corr_matrix.round(3).to_string())

        vix_corr_with_core = {}
        if "VIX Spike Reversion" in corr_matrix.columns:
            for name in CORE_STRATEGIES:
                if name in corr_matrix.index:
                    vix_corr_with_core[name] = round(corr_matrix.loc["VIX Spike Reversion", name], 3)

        # ── Step 4: Overlay weight sweep ─────────────────────────────────────
        print("\nOverlay weight sweep:")
        print(f"  {'VIX Weight':>10}  {'CAGR':>8}  {'MDD':>7}  {'Sharpe':>8}  {'AnnVol':>8}")
        print("  " + "-" * 50)

        overlay_results = []
        for vix_w in OVERLAY_WEIGHTS:
            blended = blend_portfolio(core_curves, vix_curve, vix_w)
            m = metrics_from_curve(blended)
            if m:
                overlay_results.append({"vix_weight": vix_w, **m})
                print(f"  {vix_w*100:>8.0f}%  {m['cagr']:+7.2f}%  {m['mdd']:6.2f}%  {m['sharpe']:8.3f}  {m['annual_vol_pct']:7.2f}%")
            else:
                overlay_results.append({"vix_weight": vix_w, "error": "no data"})

        # ── Step 5: Find optimal weight ──────────────────────────────────────
        valid = [r for r in overlay_results if "sharpe" in r]
        optimal = max(valid, key=lambda r: r["sharpe"]) if valid else None
        if optimal:
            print(f"\n  Optimal overlay weight: {optimal['vix_weight']*100:.0f}% VIX  (Sharpe={optimal['sharpe']:.3f})")

        output[period_name] = {
            "core_strategy_metrics": core_metrics,
            "vix_spike_reversion_metrics": vix_metrics,
            "correlation_with_vix": vix_corr_with_core,
            "correlation_matrix": corr_matrix.round(3).to_dict() if not corr_matrix.empty else {},
            "overlay_sweep": overlay_results,
            "optimal_vix_weight": optimal["vix_weight"] if optimal else None,
            "optimal_metrics": {k: v for k, v in optimal.items() if k != "vix_weight"} if optimal else None,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Optimal VIX Overlay Weight by Period")
    for period_name, result in output.items():
        w = result.get("optimal_vix_weight")
        m = result.get("optimal_metrics")
        if w is not None and m:
            print(f"  {period_name:<16}: {w*100:.0f}% VIX  (CAGR={m['cagr']:+.2f}%, Sharpe={m['sharpe']:.3f}, MDD={m['mdd']:.2f}%)")

    out_path = os.path.join(os.path.dirname(__file__), "vix_overlay_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")
    return output


if __name__ == "__main__":
    main()
