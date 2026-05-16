# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Paperclip is a stock-trading backtesting framework with a FastAPI dashboard. It defines technical-analysis strategies (~30+, e.g. EMA Crossover, RSI Reversal, Aroon Oscillator, Connors RSI), runs them through an event-driven single-stock engine, aggregates per-strategy metrics across a universe (synthetic S&P 500 or real KRX data), and surfaces results in a web UI. Postgres stores per-run trades/metrics; Elasticsearch optionally mirrors OHLCV data; results are also written to `backtesting/results.json` for the dashboard's static view.

## Common commands

```bash
# Tests
pytest                            # runs all tests under tests/
pytest tests/test_ema_crossover.py::TestEMACrossoverSignals::test_signals_have_correct_values

# Backtests (each script is independently runnable; they print metrics and
# write a *_results.json sibling file into backtesting/)
python -m backtesting.run_backtest                # 12 strategies on synthetic data, IS/OOS
python -m backtesting.run_backtest_periods        # 13 strategies × short/medium/long_term periods
python -m backtesting.run_backtest_<strategy>     # e.g. run_backtest_aroon, run_backtest_connors_rsi
python -m backtesting.optimize_params             # grid search for top trend-following strategies
python -m backtesting.optimize_top4               # KRX top-4 parameter optimization

# Data
python -m backtesting.generate_data               # synthetic GBM-calibrated S&P 500
python -m backtesting.collect_data                # yfinance → Postgres + ES
python -m backtesting.collect_data_krx            # pykrx → local CSVs / DB
python -m backtesting.seed_db                     # seed Postgres + ES with synthetic data
python -m backtesting.db                          # run SQL migrations in migrations/

# Dashboard (local)
uvicorn dashboard.app:app --reload --port 8000

# Docker stack (Postgres + dashboard at http://localhost:9000)
docker-compose up -d

# Daily pipeline used in production
./scripts/run-and-publish.sh                      # run main backtest, commit results.json, push
./scripts/deploy.sh                               # pull on Synology host, rebuild dashboard container
```

`pytest.ini` sets `pythonpath = .`, so all imports use the repo root (`from strategies...`, `from backtesting...`). The runner scripts also `sys.path.insert(0, repo_root)` so they work when invoked directly as well as via `-m`.

## Architecture

Three layers, loosely coupled:

**`strategies/`** — One module per strategy, each subclassing `BaseStrategy` (`strategies/base.py`):
- `generate_signals(df) -> pd.Series` returning `1` (buy), `-1` (sell), `0` (hold) aligned to `df.index`. `df` is OHLCV with lowercase columns (`open/high/low/close/volume`) and a `DatetimeIndex`.
- `get_signal_params() -> Signal` returns default `stop_loss` / `take_profit` / `position_size` (all as fractions).
- Optional `market_close` kwarg on `generate_signals` for strategies that need the SP500 benchmark (detected via `inspect.signature` in runners).
- Optional `last_atr` attribute + `atr_mult` (set during `generate_signals`) enables trailing-ATR stops in the engine.
- New strategies must be exported from `strategies/__init__.py` to be importable as `from strategies import XYZ`.

**`backtesting/`** — Engine + runners + data:
- `engine.run_backtest(df, signals, stop_loss, take_profit, ...)` is the single event-driven backtester. Iterates bars, opens on next bar's open after a `1` signal, exits on SL/TP/sell-signal/end. Returns `BacktestResult` with `trades`, `equity_curve`, and metrics (`cagr`, `mdd`, `sharpe`, `win_rate`, `profit_factor`, `total_trades`, `avg_holding_days`). `_compute_metrics` is reused by aggregators.
- `generate_data.generate_all_data(start, end)` returns a `{ticker: DataFrame}` of GBM-calibrated synthetic S&P 500 + a `"SP500"` index, using `ANNUAL_MARKET_RETURNS` / `ANNUAL_MARKET_VOL` per year.
- `run_backtest.py` / `run_backtest_periods.py` are the main multi-strategy drivers. They iterate the strategy registry, call `run_on_universe`, then `cross_sectional_aggregate` (mean of normalized equity curves) to derive portfolio-level metrics. They merge into `backtesting/results.json` rather than overwriting — preserve other keys when editing this file.
- `run_backtest.py` also computes `portfolio_weights` for `PORTFOLIO_STRATEGIES` using Method D (Sharpe 60% + inverse-MDD 40%, iteratively capped at `MAX_WEIGHT = 0.35`).
- Single-strategy `run_backtest_<name>.py` scripts each do their own grid search and write `<name>_results.json` sibling files. Use these as templates when adding a new strategy.
- `db.py` provides `get_connection()` (Postgres context manager) and `run_migrations()` (applies `migrations/*.sql` in order, idempotent via `_migrations` tracking table).
- `elk.py` / `collect_data_elk.py` / `ingest_daily_elk.py` are the Elasticsearch path; `db.py` also exposes `get_es_client()` for dual-writes with `ohlcv_daily` / `stocks` indices.
- KRX-specific runners (`*_krx.py`) read CSVs from `backtesting/data_krx/` (gitignored) with a `_meta.csv` index.

**`dashboard/`** — FastAPI app (`dashboard/app.py`) with Jinja2 templates:
- Reads two data sources side-by-side: static `backtesting/results.json` (always present) and Postgres (`backtest_runs` + `backtest_metrics` + `backtest_trades` + `backtest_equity_curve`, available only if DB is reachable — failures are silently swallowed).
- `STRATEGY_FACTORY` / `STRATEGY_PARAMS` / `STRATEGY_MANUALS` are hardcoded registries used by the live trade-replay endpoints (`/strategy/<name>/trades`, `/api/strategy/<name>/trades`) and the manuals UI. When adding a strategy that should appear in the dashboard, register it in all three plus `PERIOD_RANGES` if needed.
- Strategy manuals: hardcoded buy/sell/params dict in Korean (the project's primary documentation language), plus user-uploaded Markdown stored in `dashboard/manuals/` keyed by sanitized strategy name.
- Module-level caches: `_market_data_cache` (synthetic data, generated once per process) and `_trades_cache` (per `(strategy, period)`). Restart the app after data changes.

**`migrations/`** — Plain SQL files run in lexical order by `backtesting/db.py`. Add new ones with the next `00N_` prefix. Schema covers `stocks`, `ohlcv_daily`, `corporate_actions`, plus the four `backtest_*` tables used by the dashboard.

## Conventions

- Strategies operate on lowercase OHLCV columns and a `DatetimeIndex`. Never mutate `df` in place — return a fresh `pd.Series` aligned to `df.index`.
- All risk parameters (`stop_loss`, `take_profit`, `position_size`) are **fractions** (`0.04` = 4%), not percentages.
- Pass `position_size=1.0` to `run_backtest` when aggregating across a universe — the cross-sectional aggregator normalizes equity curves and a sub-unit position size would underweight the result.
- The runners catch and swallow per-ticker exceptions inside `run_on_universe`. When debugging a new strategy, run it on a single ticker outside that loop to see real tracebacks.
- `backtesting/results.json` is gitignored locally but committed by `scripts/run-and-publish.sh` — be aware that local edits won't show up via `git status` unless you `git add -f`.
- Commit messages follow `feat: <description> (DOF-NNN)` for ticketed work and `data: daily backtest results YYYY-MM-DD` for automated result publishes. Recent strategy commits are numbered against `DOF-` Linear tickets.
- Tests use `tests/conftest.py` helpers (`make_ohlcv`, `make_trending_ohlcv`, `make_falling_ohlcv`) — reuse these for new strategy tests rather than re-rolling synthetic data.
