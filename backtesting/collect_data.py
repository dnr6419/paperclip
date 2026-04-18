"""
KOSPI/KOSDAQ OHLCV data collector using pykrx.
Collects 5 years of daily data for KOSPI 200 + KOSDAQ 150 tickers.
"""
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock as pykrx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.db import get_connection

START_DATE = "20190101"
END_DATE = "20241231"
BATCH_SLEEP = 0.5


def get_kospi200_tickers():
    return pykrx.get_index_portfolio_deposit_file("1028")


def get_kosdaq150_tickers():
    return pykrx.get_index_portfolio_deposit_file("2203")


def get_stock_name(ticker):
    return pykrx.get_market_ticker_name(ticker)


def upsert_stock(cur, ticker, name, market):
    cur.execute("""
        INSERT INTO stocks (ticker, name, market)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name, market = EXCLUDED.market, updated_at = NOW()
        RETURNING id
    """, (ticker, name, market))
    return cur.fetchone()[0]


def fetch_ohlcv(ticker, start, end):
    df = pykrx.get_market_ohlcv_by_date(start, end, ticker)
    if df.empty:
        return None
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def fetch_adj_ohlcv(ticker, start, end):
    df = pykrx.get_market_ohlcv_by_date(start, end, ticker, adjusted=True)
    if df.empty:
        return None
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df


def compute_adj_factor(raw_close, adj_close):
    if raw_close == 0:
        return 1.0
    return round(adj_close / raw_close, 6)


def insert_ohlcv_batch(cur, stock_id, raw_df, adj_df):
    rows = []
    for date in raw_df.index:
        r = raw_df.loc[date]
        adj_factor = 1.0
        adj_open = adj_high = adj_low = adj_close = None

        if adj_df is not None and date in adj_df.index:
            a = adj_df.loc[date]
            adj_factor = compute_adj_factor(r["close"], a["close"])
            adj_open = float(a["open"])
            adj_high = float(a["high"])
            adj_low = float(a["low"])
            adj_close = float(a["close"])

        rows.append((
            stock_id, date.strftime("%Y-%m-%d"),
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
            int(r["volume"]),
            adj_open, adj_high, adj_low, adj_close, adj_factor,
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


def collect_market(market, tickers, start, end):
    print(f"\n{'='*60}")
    print(f"Collecting {market} — {len(tickers)} tickers")
    print(f"Period: {start} ~ {end}")
    print(f"{'='*60}")

    total_rows = 0
    failed = []

    with get_connection() as conn:
        cur = conn.cursor()
        for i, ticker in enumerate(tickers, 1):
            try:
                name = get_stock_name(ticker)
                stock_id = upsert_stock(cur, ticker, name, market)

                raw_df = fetch_ohlcv(ticker, start, end)
                if raw_df is None or raw_df.empty:
                    print(f"  [{i}/{len(tickers)}] {ticker} ({name}): no data")
                    continue

                adj_df = fetch_adj_ohlcv(ticker, start, end)
                count = insert_ohlcv_batch(cur, stock_id, raw_df, adj_df)
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
    print("OHLCV Data Collection — KOSPI 200 + KOSDAQ 150")
    print(f"Period: {START_DATE} ~ {END_DATE}\n")

    kospi_tickers = get_kospi200_tickers()
    kosdaq_tickers = get_kosdaq150_tickers()

    print(f"KOSPI 200 tickers: {len(kospi_tickers)}")
    print(f"KOSDAQ 150 tickers: {len(kosdaq_tickers)}")

    kospi_rows, kospi_failed = collect_market("KOSPI", kospi_tickers, START_DATE, END_DATE)
    kosdaq_rows, kosdaq_failed = collect_market("KOSDAQ", kosdaq_tickers, START_DATE, END_DATE)

    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"  KOSPI: {kospi_rows} rows, {len(kospi_failed)} failures")
    print(f"  KOSDAQ: {kosdaq_rows} rows, {len(kosdaq_failed)} failures")
    print(f"  Total: {kospi_rows + kosdaq_rows} rows")

    if kospi_failed or kosdaq_failed:
        print("\nFailed tickers:")
        for ticker, err in kospi_failed + kosdaq_failed:
            print(f"  {ticker}: {err}")


if __name__ == "__main__":
    main()
