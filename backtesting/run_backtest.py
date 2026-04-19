"""
Main backtest runner: 12 strategies on synthetic S&P 500 market data.
In-Sample (2019-2022) / Out-of-Sample (2023-2024).
"""
import sys, os, json, warnings, inspect
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import (
    EMACrossoverStrategy, RSIReversalStrategy,
    CandleRSIStrategy, ADXTrendStrategy,
    High52WBreakoutStrategy, ATRBreakoutStrategy,
    VWBStrategy, MTMStrategy, DCBStrategy, MACDMomentumStrategy,
    GapFillStrategy, DualMomentumStrategy,
)
from backtesting.engine import run_backtest, BacktestResult, _compute_metrics
from backtesting.generate_data import generate_all_data

STRATEGIES = {
    "EMA Crossover":    (EMACrossoverStrategy(),    {"stop_loss": 0.04, "take_profit": 0.30}),
    "RSI Reversal":     (RSIReversalStrategy(),     {"stop_loss": 0.04, "take_profit": 0.25}),
    "Candle+RSI":       (CandleRSIStrategy(),       {"stop_loss": 0.04, "take_profit": 0.20}),
    "ADX Trend":        (ADXTrendStrategy(),        {"stop_loss": 0.04, "take_profit": 0.25}),
    "52W Breakout":     (High52WBreakoutStrategy(), {"stop_loss": 0.06, "take_profit": 0.20}),
    "ATR Breakout":     (ATRBreakoutStrategy(),     {"stop_loss": 0.04, "take_profit": 0.20}),
    "VWB":              (VWBStrategy(),             {"stop_loss": 0.05, "take_profit": 10.0}),
    "MTM":              (MTMStrategy(),             {"stop_loss": 0.05, "take_profit": 0.20}),
    "DCB":              (DCBStrategy(),             {"stop_loss": 0.05, "take_profit": 0.25}),
    "MACD Momentum":    (MACDMomentumStrategy(),    {"stop_loss": 0.05, "take_profit": 0.25}),
    "Gap Fill":         (GapFillStrategy(),         {"stop_loss": 0.03, "take_profit": 0.05}),
    "Dual Momentum":    (DualMomentumStrategy(),    {"stop_loss": 0.05, "take_profit": 0.30}),
}

PERIODS = {
    "in_sample":     ("2019-01-01", "2022-12-31"),
    "out_of_sample": ("2023-01-01", "2024-12-31"),
}
IC = 100_000


def cross_sectional_aggregate(per_stock_results):
    """Average equity curves (normalized) then compute portfolio metrics."""
    if not per_stock_results:
        return None
    all_eq = []
    all_trades = []
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
            bt_params = dict(params)
            if hasattr(strategy, "last_atr") and strategy.last_atr is not None:
                bt_params["trailing_atr_series"] = strategy.last_atr
                bt_params["trailing_atr_mult"] = strategy.atr_mult
            r = run_backtest(df, signals, position_size=1.0, initial_capital=IC, **bt_params)
            r.strategy_name = ""
            r.ticker = ticker
            results.append(r)
        except Exception:
            pass
    return results


PORTFOLIO_STRATEGIES = ["DCB", "ATR Breakout", "MTM", "Candle+RSI", "Dual Momentum"]
MAX_WEIGHT = 0.35


def compute_portfolio_weights(oos_results: dict) -> dict:
    """Method D: Sharpe 60% + Inverse-MDD 40%, capped at MAX_WEIGHT per strategy."""
    candidates = {k: v for k, v in oos_results.items() if k in PORTFOLIO_STRATEGIES and v}

    sharpe_scores = {k: max(v["sharpe"], 0.0) for k, v in candidates.items()}
    inv_mdd_scores = {k: 1.0 / max(v["mdd"], 0.1) for k, v in candidates.items()}

    total_sharpe = sum(sharpe_scores.values()) or 1.0
    total_inv_mdd = sum(inv_mdd_scores.values()) or 1.0

    raw = {
        k: 0.60 * (sharpe_scores[k] / total_sharpe) + 0.40 * (inv_mdd_scores[k] / total_inv_mdd)
        for k in candidates
    }

    # Iterative cap: redistribute excess from capped strategies
    weights = dict(raw)
    for _ in range(10):
        capped = {k: min(w, MAX_WEIGHT) for k, w in weights.items()}
        excess = sum(w - MAX_WEIGHT for w in weights.values() if w > MAX_WEIGHT)
        uncapped_keys = [k for k, w in weights.items() if w < MAX_WEIGHT]
        if excess <= 0 or not uncapped_keys:
            weights = capped
            break
        total_uncapped = sum(capped[k] for k in uncapped_keys) or 1.0
        for k in uncapped_keys:
            capped[k] += excess * (capped[k] / total_uncapped)
        weights = capped

    total = sum(weights.values()) or 1.0
    return {k: round(w / total, 4) for k, w in weights.items()}


def main():
    print("=" * 60)
    print("BACKTESTING RUNNER — 12 Active Strategies, Synthetic S&P 500 Universe")
    print("=" * 60)
    print("\nGenerating synthetic market data (2019-2024)...", flush=True)
    all_data = generate_all_data("2019-01-01", "2024-12-31")
    sp500_full = all_data.pop("SP500")
    print(f"  {len(all_data)} stocks generated")

    output = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n[{period_name.upper().replace('_', '-')}] {start} ~ {end}")
        stock_data = {k: v.loc[start:end] for k, v in all_data.items()}
        sp500 = sp500_full.loc[start:end]

        period_out = {}
        for strat_name, (strategy, params) in STRATEGIES.items():
            print(f"  {strat_name}...", end="", flush=True)
            results = run_on_universe(strategy, params, stock_data, sp500)
            agg = cross_sectional_aggregate(results)
            if agg:
                agg.strategy_name = strat_name
                pass_fail = "PASS" if (agg.sharpe > 1.0 and agg.mdd < 20.0 and agg.cagr > 10.0) else "fail"
                print(f" CAGR={agg.cagr:+.1f}% MDD={agg.mdd:.1f}% Sharpe={agg.sharpe:.2f}"
                      f" Win={agg.win_rate:.1f}% PF={agg.profit_factor:.2f}"
                      f" Trades={agg.total_trades} [{pass_fail}]")
                period_out[strat_name] = {
                    "cagr": round(agg.cagr, 2),
                    "mdd": round(agg.mdd, 2),
                    "sharpe": round(agg.sharpe, 3),
                    "win_rate": round(agg.win_rate, 1),
                    "profit_factor": round(min(agg.profit_factor, 99.0), 2),
                    "total_trades": agg.total_trades,
                    "avg_holding_days": round(agg.avg_holding_days, 1),
                }
            else:
                print(" no signals")
                period_out[strat_name] = None

        output[period_name] = period_out

    # Portfolio weight computation (OOS results, Method D)
    oos = output.get("out_of_sample", {})
    weights = compute_portfolio_weights(oos)
    output["portfolio_weights"] = weights

    print("\n" + "=" * 60)
    print("PORTFOLIO WEIGHTS (Method D: Sharpe 60% + Inverse-MDD 40%)")
    print("=" * 60)
    for strat, w in sorted(weights.items(), key=lambda x: -x[1]):
        metrics = oos.get(strat, {}) or {}
        sharpe = metrics.get("sharpe", "-")
        mdd = metrics.get("mdd", "-")
        print(f"  {strat:<20} {w*100:5.1f}%  (Sharpe={sharpe}, MDD={mdd}%)")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    main()
