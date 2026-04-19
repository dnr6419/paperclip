"""
Korean market (KOSPI/KOSDAQ) OHLCV data collector using pykrx.
Collects 6 years of daily data for 50 representative KRX tickers.
Dual-writes to PostgreSQL and Elasticsearch.
"""
import os
import sys
import time

import pandas as pd
from pykrx import stock as krx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.db import get_connection, get_es_client, ensure_es_indices, OHLCV_INDEX, STOCKS_INDEX

START_DATE = "20190101"
END_DATE = "20241231"
BATCH_SLEEP = 1.0

KRX_TICKERS = [
    # KOSPI — 대형주 (시가총액 상위)
    ("005930", "삼성전자", "KOSPI"),
    ("000660", "SK하이닉스", "KOSPI"),
    ("373220", "LG에너지솔루션", "KOSPI"),
    ("207940", "삼성바이오로직스", "KOSPI"),
    ("005380", "현대차", "KOSPI"),
    ("000270", "기아", "KOSPI"),
    ("006400", "삼성SDI", "KOSPI"),
    ("051910", "LG화학", "KOSPI"),
    ("035420", "NAVER", "KOSPI"),
    ("005490", "POSCO홀딩스", "KOSPI"),
    # KOSPI — 금융
    ("055550", "신한지주", "KOSPI"),
    ("105560", "KB금융", "KOSPI"),
    ("086790", "하나금융지주", "KOSPI"),
    ("316140", "우리금융지주", "KOSPI"),
    ("000810", "삼성화재", "KOSPI"),
    # KOSPI — 산업/소재
    ("010130", "고려아연", "KOSPI"),
    ("012330", "현대모비스", "KOSPI"),
    ("009150", "삼성전기", "KOSPI"),
    ("034730", "SK", "KOSPI"),
    ("003670", "포스코퓨처엠", "KOSPI"),
    # KOSPI — 소비/유통
    ("030200", "KT", "KOSPI"),
    ("017670", "SK텔레콤", "KOSPI"),
    ("032830", "삼성생명", "KOSPI"),
    ("066570", "LG전자", "KOSPI"),
    ("003550", "LG", "KOSPI"),
    # KOSPI — 중형주
    ("028260", "삼성물산", "KOSPI"),
    ("010950", "S-Oil", "KOSPI"),
    ("036570", "엔씨소프트", "KOSPI"),
    ("011200", "HMM", "KOSPI"),
    ("034020", "두산에너빌리티", "KOSPI"),
    ("000720", "현대건설", "KOSPI"),
    ("018260", "삼성에스디에스", "KOSPI"),
    ("033780", "KT&G", "KOSPI"),
    ("015760", "한국전력", "KOSPI"),
    ("009540", "한국조선해양", "KOSPI"),
    # KOSDAQ — 대표 종목
    ("247540", "에코프로비엠", "KOSDAQ"),
    ("086520", "에코프로", "KOSDAQ"),
    ("035720", "카카오", "KOSPI"),
    ("035900", "JYP Ent.", "KOSDAQ"),
    ("041510", "에스엠", "KOSPI"),
    ("263750", "펄어비스", "KOSDAQ"),
    ("293490", "카카오게임즈", "KOSDAQ"),
    ("112040", "위메이드", "KOSDAQ"),
    ("068270", "셀트리온", "KOSPI"),
    ("326030", "SK바이오팜", "KOSPI"),
    ("352820", "하이브", "KOSPI"),
    ("259960", "크래프톤", "KOSPI"),
    ("003490", "대한항공", "KOSPI"),
    ("180640", "한진칼", "KOSPI"),
    ("069500", "KODEX 200", "KOSPI"),
]


def fetch_ohlcv_krx(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = krx.get_market_ohlcv_by_date(start, end, ticker)
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"
    df = df[df["volume"] > 0]
    return df


def upsert_stock(cur, ticker, name, market):
    cur.execute("""
        INSERT INTO stocks (ticker, name, market)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name, market = EXCLUDED.market, updated_at = NOW()
        RETURNING id
    """, (ticker, name, market))
    return cur.fetchone()[0]


def insert_ohlcv_batch(cur, stock_id, df: pd.DataFrame):
    rows = []
    for date, row in df.iterrows():
        rows.append((
            stock_id, date.strftime("%Y-%m-%d"),
            float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
            int(row["volume"]),
            None, None, None, float(row["close"]), 1.0,
            "pykrx",
        ))
    if not rows:
        return 0
    from psycopg2.extras import execute_values
    execute_values(cur, """
        INSERT INTO ohlcv_daily (stock_id, date, open, high, low, close, volume,
                                  adj_open, adj_high, adj_low, adj_close, adj_factor, source)
        VALUES %s
        ON CONFLICT (stock_id, date) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume,
            adj_open = EXCLUDED.adj_open, adj_high = EXCLUDED.adj_high,
            adj_low = EXCLUDED.adj_low, adj_close = EXCLUDED.adj_close,
            adj_factor = EXCLUDED.adj_factor, source = EXCLUDED.source
    """, rows)
    return len(rows)


def index_stock_es(es, ticker, name, market, stock_id):
    es.index(index=STOCKS_INDEX, id=ticker, document={
        "ticker": ticker,
        "name": name,
        "market": market,
        "stock_id": stock_id,
        "is_active": True,
    })


def index_ohlcv_es(es, ticker, stock_id, df: pd.DataFrame):
    from elasticsearch.helpers import bulk

    actions = []
    for date, row in df.iterrows():
        doc_id = f"{ticker}_{date.strftime('%Y-%m-%d')}"
        actions.append({
            "_index": OHLCV_INDEX,
            "_id": doc_id,
            "_source": {
                "ticker": ticker,
                "stock_id": stock_id,
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "adj_close": float(row["close"]),
                "adj_factor": 1.0,
                "source": "pykrx",
            },
        })

    if actions:
        bulk(es, actions, raise_on_error=False)
    return len(actions)


def collect_krx(tickers, start, end):
    print(f"\n{'='*60}")
    print(f"Collecting KRX — {len(tickers)} tickers")
    print(f"Period: {start} ~ {end}")
    print(f"{'='*60}")

    total_rows = 0
    es_rows = 0
    failed = []

    es = get_es_client()
    ensure_es_indices(es)
    print("  Elasticsearch connected and indices ready")

    with get_connection() as conn:
        cur = conn.cursor()
        for i, (ticker, name, market) in enumerate(tickers, 1):
            try:
                stock_id = upsert_stock(cur, ticker, name, market)

                df = fetch_ohlcv_krx(ticker, start, end)
                if df is None or df.empty:
                    print(f"  [{i}/{len(tickers)}] {ticker}: no data")
                    continue

                count = insert_ohlcv_batch(cur, stock_id, df)
                total_rows += count

                index_stock_es(es, ticker, name, market, stock_id)
                es_count = index_ohlcv_es(es, ticker, stock_id, df)
                es_rows += es_count

                print(f"  [{i}/{len(tickers)}] {ticker} ({name}): {count} PG + {es_count} ES rows")

                conn.commit()
                time.sleep(BATCH_SLEEP)

            except Exception as e:
                failed.append((ticker, str(e)))
                print(f"  [{i}/{len(tickers)}] {ticker}: FAILED — {e}")
                conn.rollback()

    return total_rows, es_rows, failed


def main():
    print("OHLCV Data Collection — KRX (50 representative tickers)")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {len(KRX_TICKERS)}")
    print(f"Targets: PostgreSQL + Elasticsearch\n")

    pg_rows, es_rows, failed = collect_krx(KRX_TICKERS, START_DATE, END_DATE)

    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"  PostgreSQL: {pg_rows} rows")
    print(f"  Elasticsearch: {es_rows} rows")
    print(f"  Failures: {len(failed)}")

    if failed:
        print("\nFailed tickers:")
        for ticker, err in failed:
            print(f"  {ticker}: {err}")


if __name__ == "__main__":
    main()
