import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

RESULTS_JSON = Path(__file__).resolve().parent.parent / "backtesting" / "results.json"

DB_CONFIG = {
    "host": os.getenv("BACKTEST_DB_HOST", "localhost"),
    "port": int(os.getenv("BACKTEST_DB_PORT", "5432")),
    "dbname": os.getenv("BACKTEST_DB_NAME", "backtesting"),
    "user": os.getenv("BACKTEST_DB_USER", "backtest"),
    "password": os.getenv("BACKTEST_DB_PASSWORD", "backtest"),
}


def _get_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception:
        return None


def _load_static_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _load_db_runs(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT r.id, r.strategy_name, r.start_date, r.end_date, r.parameters,
               r.status, r.created_at,
               m.annualized_return AS cagr, m.max_drawdown AS mdd,
               m.sharpe_ratio AS sharpe, m.win_rate,
               m.profit_factor, m.total_trades, m.avg_holding_days
        FROM backtest_runs r
        LEFT JOIN backtest_metrics m ON m.run_id = r.id
        WHERE r.status = 'completed'
        ORDER BY r.created_at DESC
    """)
    return cur.fetchall()


def _load_db_equity_curve(conn, run_id: str):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT date, portfolio_value, drawdown
        FROM backtest_equity_curve
        WHERE run_id = %s
        ORDER BY date
    """, (run_id,))
    return cur.fetchall()


def _load_db_trades(conn, run_id: str):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT entry_date, exit_date, entry_price, exit_price,
               pnl_pct, exit_reason, holding_days
        FROM backtest_trades
        WHERE run_id = %s
        ORDER BY entry_date
    """, (run_id,))
    return cur.fetchall()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Backtesting Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    static = _load_static_results()
    db_available = False
    db_runs = []
    conn = _get_db()
    if conn:
        try:
            db_runs = _load_db_runs(conn)
            db_available = True
        except Exception:
            pass
        finally:
            conn.close()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "static_results": static,
            "db_runs": db_runs,
            "db_available": db_available,
        },
    )


@app.get("/api/results")
async def api_results():
    return _load_static_results()


@app.get("/api/runs")
async def api_runs():
    conn = _get_db()
    if not conn:
        return {"error": "Database not available", "runs": []}
    try:
        runs = _load_db_runs(conn)
        for r in runs:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
                elif isinstance(v, (int, float, str, bool, type(None), dict, list)):
                    pass
                else:
                    r[k] = str(v)
        return {"runs": runs}
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/equity")
async def api_equity(run_id: str):
    conn = _get_db()
    if not conn:
        return {"error": "Database not available"}
    try:
        rows = _load_db_equity_curve(conn, run_id)
        return {
            "dates": [r["date"].isoformat() for r in rows],
            "values": [float(r["portfolio_value"]) for r in rows],
            "drawdowns": [float(r["drawdown"]) if r["drawdown"] else 0 for r in rows],
        }
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/trades")
async def api_trades(run_id: str):
    conn = _get_db()
    if not conn:
        return {"error": "Database not available"}
    try:
        rows = _load_db_trades(conn, run_id)
        for r in rows:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
                elif isinstance(v, (int, float, str, bool, type(None))):
                    pass
                else:
                    r[k] = str(v)
        return {"trades": rows}
    finally:
        conn.close()
