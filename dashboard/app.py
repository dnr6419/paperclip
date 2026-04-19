import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

RESULTS_JSON = Path(__file__).resolve().parent.parent / "backtesting" / "results.json"
MANUALS_DIR = Path(__file__).resolve().parent / "manuals"
MANUALS_DIR.mkdir(exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("BACKTEST_DB_HOST", "localhost"),
    "port": int(os.getenv("BACKTEST_DB_PORT", "5432")),
    "dbname": os.getenv("BACKTEST_DB_NAME", "backtesting"),
    "user": os.getenv("BACKTEST_DB_USER", "backtest"),
    "password": os.getenv("BACKTEST_DB_PASSWORD", "backtest"),
}

STRATEGY_MANUALS = {
    "EMA Crossover": {
        "buy": [
            "EMA(8)이 EMA(20)을 상향 돌파 (골든크로스)",
            "거래량이 20일 평균의 1.2배 이상 (거래량 확인)",
            "전일 종가가 EMA(20) 위에 위치",
        ],
        "sell": [
            "EMA(8)이 EMA(20)을 하향 돌파 (데드크로스)",
            "손절: 진입가 대비 -4%",
            "익절: 진입가 대비 +30%",
        ],
        "params": {"fast_ema": 8, "slow_ema": 20, "vol_multiplier": 1.2,
                   "stop_loss": "4%", "take_profit": "30%"},
    },
    "RSI Reversal": {
        "buy": [
            "RSI(14)가 30 아래에서 다시 30 위로 상향 돌파 (과매도 탈출)",
            "종가가 SMA(200) 위에 위치 (장기 상승 추세 확인)",
        ],
        "sell": [
            "RSI(14)가 65 이상 도달",
            "손절: 진입가 대비 -4%",
            "익절: 진입가 대비 +25%",
        ],
        "params": {"rsi_period": 14, "oversold": 30, "overbought_exit": 65,
                   "sma_filter": 200, "stop_loss": "4%", "take_profit": "25%"},
    },
    "Candle+RSI": {
        "buy": [
            "강세 반전 캔들 패턴 감지 (Hammer, Morning Star, Bullish Engulfing)",
            "RSI(14) < 40 확인 (과매도 근처)",
        ],
        "sell": [
            "손절: 패턴 저점 -1%",
            "1차 익절: +6% (부분 청산)",
            "2차 익절: +10%",
        ],
        "params": {"rsi_threshold": 40, "stop_loss": "4%",
                   "take_profit_1": "6%", "take_profit_2": "10%"},
    },
    "ADX Trend": {
        "buy": [
            "ADX > 15 (추세 존재 확인)",
            "+DI가 -DI를 상향 돌파",
            "종가가 SMA(50) 위에 위치",
        ],
        "sell": [
            "익절: 진입가 대비 +25%",
            "손절: 진입가 대비 -4%",
            "시그널 기반 매도 없음 — TP/SL만 사용",
        ],
        "params": {"adx_threshold": 15, "sma_filter": 50,
                   "stop_loss": "4%", "take_profit": "25%"},
    },
    "52W Breakout": {
        "buy": [
            "종가가 52주(252거래일) 최고가를 돌파",
            "거래량이 평균의 2배 이상",
            "시장(SP500)이 200일 이평선 위 (상승 추세)",
        ],
        "sell": [
            "종가가 돌파일 저점 아래로 하락 시 손절",
            "1차 익절: +15% (부분 청산)",
            "2차 익절: +25%",
        ],
        "params": {"lookback": 252, "vol_multiplier": 2.0,
                   "stop_loss": "6%", "take_profit": "20%"},
    },
    "ATR Breakout": {
        "buy": [
            "종가가 20일 최고가 + ATR(14) × 0.5 이상으로 돌파",
            "변동성 조정된 확장 확인",
        ],
        "sell": [
            "트레일링 스탑: 최고가 - ATR × 2.0",
            "손절: 진입가 대비 -5%",
            "익절: 진입가 대비 +20%",
        ],
        "params": {"high_period": 20, "atr_period": 14, "breakout_mult": 0.5,
                   "atr_mult": 2.0, "stop_loss": "5%", "take_profit": "20%"},
    },
    "BB Mean Reversion": {
        "buy": [
            "가격이 하단 볼린저밴드(20, 2σ) 아래에서 다시 위로 복귀",
            "RSI(14) < 50 (과매수 아님)",
            "종가가 SMA(50) 위 (중기 상승 추세 필터)",
        ],
        "sell": [
            "손절: 진입가 대비 -6%",
            "익절: 진입가 대비 +12%",
        ],
        "params": {"bb_period": 20, "bb_std": 2.0, "rsi_threshold": 50,
                   "sma_filter": 50, "stop_loss": "6%", "take_profit": "12%"},
    },
    "VWB": {
        "buy": [
            "종가가 (N일 롤링 최고가 + breakout_mult × ATR) 이상 돌파",
            "거래량이 N일 평균의 vol_multiplier배 이상",
        ],
        "sell": [
            "트레일링 스탑: 최고가 - ATR × 2.0",
            "하드 손절: 진입가 대비 -5%",
        ],
        "params": {"atr_mult": 2.0, "breakout_mult": 0.5,
                   "stop_loss": "5%", "take_profit": "trailing"},
    },
    "MTM": {
        "buy": [
            "주봉: 종가가 MA(10) 위 + RSI > 50",
            "일봉: 종가가 MA(20) 위 + RSI > 50",
            "두 타임프레임 동시 충족 시 진입",
        ],
        "sell": [
            "일봉 약세: 종가가 MA(20) 아래 OR RSI < 45",
            "주봉 반전: 약세 전환 시",
            "손절: 진입가 대비 -5%",
            "익절: 진입가 대비 +20%",
        ],
        "params": {"daily_ma": 20, "weekly_ma": 10, "rsi_threshold": 50,
                   "stop_loss": "5%", "take_profit": "20%"},
    },
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


def _load_strategy_tickers(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT r.strategy_name,
               s.ticker,
               s.name AS stock_name,
               COUNT(*) AS trade_count,
               SUM(CASE WHEN t.direction = 'long' THEN 1 ELSE 0 END) AS long_count,
               SUM(CASE WHEN t.direction = 'short' THEN 1 ELSE 0 END) AS short_count,
               ROUND(AVG(t.pnl_pct)::numeric, 4) AS avg_pnl_pct
        FROM backtest_trades t
        JOIN backtest_runs r ON r.id = t.run_id
        JOIN stocks s ON s.id = t.stock_id
        WHERE r.status = 'completed'
        GROUP BY r.strategy_name, s.ticker, s.name
        ORDER BY r.strategy_name, trade_count DESC
    """)
    rows = cur.fetchall()
    result = {}
    for row in rows:
        strat = row["strategy_name"]
        if strat not in result:
            result[strat] = []
        result[strat].append({
            "ticker": row["ticker"],
            "name": row["stock_name"],
            "trade_count": row["trade_count"],
            "long_count": row["long_count"],
            "short_count": row["short_count"],
            "avg_pnl_pct": float(row["avg_pnl_pct"]) if row["avg_pnl_pct"] else 0,
        })
    return result


def _load_uploaded_manual(strategy_name: str) -> str | None:
    safe_name = strategy_name.replace(" ", "_").replace("/", "_")
    path = MANUALS_DIR / f"{safe_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


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
    strategy_tickers = {}
    conn = _get_db()
    if conn:
        try:
            db_runs = _load_db_runs(conn)
            strategy_tickers = _load_strategy_tickers(conn)
            db_available = True
        except Exception:
            pass
        finally:
            conn.close()

    manuals_with_uploads = {}
    for strat_name, manual in STRATEGY_MANUALS.items():
        entry = dict(manual)
        uploaded = _load_uploaded_manual(strat_name)
        if uploaded:
            entry["uploaded_manual"] = uploaded
        manuals_with_uploads[strat_name] = entry

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "static_results": static,
            "db_runs": db_runs,
            "db_available": db_available,
            "strategy_manuals": manuals_with_uploads,
            "strategy_tickers": strategy_tickers,
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


@app.get("/api/strategies/manuals")
async def api_strategy_manuals():
    result = {}
    for name, manual in STRATEGY_MANUALS.items():
        entry = dict(manual)
        uploaded = _load_uploaded_manual(name)
        if uploaded:
            entry["uploaded_manual"] = uploaded
        result[name] = entry
    return result


@app.get("/api/strategies/tickers")
async def api_strategy_tickers():
    conn = _get_db()
    if not conn:
        return {"error": "Database not available", "tickers": {}}
    try:
        tickers = _load_strategy_tickers(conn)
        return {"tickers": tickers}
    finally:
        conn.close()


@app.post("/api/strategies/{strategy_name}/manual")
async def upload_manual(strategy_name: str, file: UploadFile = File(...)):
    if strategy_name not in STRATEGY_MANUALS:
        return JSONResponse(status_code=404, content={"error": "Unknown strategy"})
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"error": "File must be UTF-8 text"})
    if len(text) > 100_000:
        return JSONResponse(status_code=400, content={"error": "File too large (max 100KB)"})
    safe_name = strategy_name.replace(" ", "_").replace("/", "_")
    path = MANUALS_DIR / f"{safe_name}.md"
    path.write_text(text, encoding="utf-8")
    return {"ok": True, "strategy": strategy_name, "size": len(text)}
