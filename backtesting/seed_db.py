"""
Seed PostgreSQL and Elasticsearch with synthetic S&P 500 data.
Uses GBM-calibrated data from generate_data.py when yfinance is unavailable.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.generate_data import generate_all_data, STOCK_CONFIGS
from backtesting.db import get_connection, get_es_client, ensure_es_indices, OHLCV_INDEX, STOCKS_INDEX
from backtesting.collect_data import (
    upsert_stock, insert_ohlcv_batch, index_stock_es, index_ohlcv_es, get_market_for_ticker,
)

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def main():
    print("Seeding DB with synthetic S&P 500 data (GBM-calibrated)")
    print(f"Period: {START_DATE} ~ {END_DATE}\n")

    data = generate_all_data(START_DATE, END_DATE)
    tickers = [c["name"] for c in STOCK_CONFIGS] + ["SP500"]

    es = get_es_client()
    ensure_es_indices(es)
    print("Elasticsearch connected and indices ready\n")

    total_pg = 0
    total_es = 0

    with get_connection() as conn:
        cur = conn.cursor()
        for i, ticker in enumerate(tickers, 1):
            df = data[ticker]
            market = get_market_for_ticker(ticker) if ticker != "SP500" else "NYSE"
            stock_id = upsert_stock(cur, ticker, ticker, market)

            count = insert_ohlcv_batch(cur, stock_id, df)
            total_pg += count

            try:
                index_stock_es(es, ticker, ticker, market, stock_id)
                es_count = index_ohlcv_es(es, ticker, stock_id, df)
                total_es += es_count
            except Exception as e:
                es_count = 0
                print(f"  [{i}/{len(tickers)}] {ticker}: ES failed ({e}), PG ok")

            print(f"  [{i}/{len(tickers)}] {ticker}: {count} PG + {es_count} ES rows")
            conn.commit()

    print(f"\nSeed complete: {total_pg} PG rows, {total_es} ES rows across {len(tickers)} tickers")


if __name__ == "__main__":
    main()
