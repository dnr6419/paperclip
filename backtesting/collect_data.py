"""
S&P 500 OHLCV data collector using yfinance.
Collects 6 years of daily data for 50 representative S&P 500 tickers.
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.db import get_connection

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
BATCH_SLEEP = 0.3

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


def collect_sp500(tickers, start, end):
    print(f"\n{'='*60}")
    print(f"Collecting S&P 500 — {len(tickers)} tickers")
    print(f"Period: {start} ~ {end}")
    print(f"{'='*60}")

    total_rows = 0
    failed = []

    with get_connection() as conn:
        cur = conn.cursor()
        for i, ticker in enumerate(tickers, 1):
            try:
                info = yf.Ticker(ticker).info
                name = info.get("shortName") or info.get("longName") or ticker
                market = get_market_for_ticker(ticker)
                stock_id = upsert_stock(cur, ticker, name, market)

                df = fetch_ohlcv(ticker, start, end)
                if df is None or df.empty:
                    print(f"  [{i}/{len(tickers)}] {ticker}: no data")
                    continue

                count = insert_ohlcv_batch(cur, stock_id, df)
                total_rows += count
                print(f"  [{i}/{len(tickers)}] {ticker} ({name}): {count} rows")

                conn.commit()
                time.sleep(BATCH_SLEEP)

            except Exception as e:
                failed.append((ticker, str(e)))
                print(f"  [{i}/{len(tickers)}] {ticker}: FAILED — {e}")
                conn.rollback()

    return total_rows, failed


def main():
    print("OHLCV Data Collection — S&P 500 (50 representative tickers)")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {len(SP500_TICKERS)}\n")

    rows, failed = collect_sp500(SP500_TICKERS, START_DATE, END_DATE)

    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"  Total: {rows} rows, {len(failed)} failures")

    if failed:
        print("\nFailed tickers:")
        for ticker, err in failed:
            print(f"  {ticker}: {err}")


if __name__ == "__main__":
    main()
