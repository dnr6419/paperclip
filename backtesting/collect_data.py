"""
S&P 500 OHLCV data collector using yfinance.
Collects 6 years of daily data for 50 representative S&P 500 tickers.
Dual-writes to PostgreSQL and Elasticsearch.
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.db import get_connection, get_es_client, ensure_es_indices, OHLCV_INDEX, STOCKS_INDEX

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
BATCH_SLEEP = 2.0
MAX_RETRIES = 3

SP500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD", "ORCL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "V", "MA", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "CVS", "CI", "HUM",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
    # Consumer Staples / Discretionary
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX",
    # Industrials
    "BA", "CAT", "HON", "GE", "MMM",
    # Utilities / Telecom
    "T", "VZ", "NEE",
]

_NASDAQ_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD",
    "ORCL", "COST", "MRK", "SBUX",
}


def get_market_for_ticker(ticker: str) -> str:
    return "NASDAQ" if ticker in _NASDAQ_TICKERS else "NYSE"


def upsert_stock(cur, ticker, name, market):
    cur.execute("""
        INSERT INTO stocks (ticker, name, market)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name, market = EXCLUDED.market, updated_at = NOW()
        RETURNING id
    """, (ticker, name, market))
    return cur.fetchone()[0]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def fetch_ohlcv_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=True, group_by="ticker", threads=False)
    result = {}
    if raw.empty:
        return result
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "date"
            result[ticker] = df
        except (KeyError, TypeError):
            continue
    return result


def insert_ohlcv_batch(cur, stock_id, df: pd.DataFrame):
    rows = []
    for date, row in df.iterrows():
        rows.append((
            stock_id, date.strftime("%Y-%m-%d"),
            float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
            int(row["volume"]),
            None, None, None, float(row["close"]), 1.0,
            "yfinance",
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
                "source": "yfinance",
            },
        })

    if actions:
        bulk(es, actions, raise_on_error=False)
    return len(actions)


def collect_sp500(tickers, start, end):
    print(f"\n{'='*60}")
    print(f"Collecting S&P 500 — {len(tickers)} tickers")
    print(f"Period: {start} ~ {end}")
    print(f"{'='*60}")

    total_rows = 0
    es_rows = 0
    failed = []

    es = get_es_client()
    ensure_es_indices(es)
    print("  Elasticsearch connected and indices ready")

    batch_size = 10
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        print(f"\n  Batch {batch_num}/{total_batches}: {', '.join(batch)}")

        for attempt in range(MAX_RETRIES):
            try:
                data = fetch_ohlcv_batch(batch, start, end)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = BATCH_SLEEP * (2 ** (attempt + 1))
                    print(f"    Download failed, retry in {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    print(f"    Batch download FAILED: {e}")
                    for t in batch:
                        failed.append((t, str(e)))
                    data = {}

        if not data:
            print(f"    No data returned for this batch")
            for t in batch:
                if t not in [f[0] for f in failed]:
                    failed.append((t, "no data"))
            time.sleep(BATCH_SLEEP)
            continue

        with get_connection() as conn:
            cur = conn.cursor()
            for i, ticker in enumerate(batch, batch_start + 1):
                df = data.get(ticker)
                if df is None or df.empty:
                    print(f"    [{i}/{len(tickers)}] {ticker}: no data")
                    continue

                try:
                    market = get_market_for_ticker(ticker)
                    stock_id = upsert_stock(cur, ticker, ticker, market)
                    count = insert_ohlcv_batch(cur, stock_id, df)
                    total_rows += count

                    try:
                        index_stock_es(es, ticker, ticker, market, stock_id)
                        es_count = index_ohlcv_es(es, ticker, stock_id, df)
                        es_rows += es_count
                    except Exception as es_err:
                        es_count = 0
                        print(f"    [{i}/{len(tickers)}] {ticker}: ES write failed ({es_err}), PG ok")

                    print(f"    [{i}/{len(tickers)}] {ticker}: {count} PG + {es_count} ES rows")
                    conn.commit()

                except Exception as e:
                    failed.append((ticker, str(e)))
                    print(f"    [{i}/{len(tickers)}] {ticker}: FAILED — {e}")
                    conn.rollback()

        time.sleep(BATCH_SLEEP)

    return total_rows, es_rows, failed


def main():
    print("OHLCV Data Collection — S&P 500 (50 representative tickers)")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {len(SP500_TICKERS)}")
    print(f"Targets: PostgreSQL + Elasticsearch\n")

    pg_rows, es_rows, failed = collect_sp500(SP500_TICKERS, START_DATE, END_DATE)

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
