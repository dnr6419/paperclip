"""
S&P 500 OHLCV data collector — writes to Elasticsearch (ohlcv-daily index).
Fetches daily data via yfinance and bulk-inserts into ELK.
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.elk import INDEX, ensure_index, get_client

# RSI strategy needs SMA(200), so at least 200 trading days before 2019-01-01
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


def build_actions(ticker: str, df: pd.DataFrame):
    for date, row in df.iterrows():
        doc_id = f"{ticker}_{date.strftime('%Y-%m-%d')}"
        yield {
            "_index": INDEX,
            "_id": doc_id,
            "_source": {
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            },
        }


def collect_sp500(tickers, start, end, client: Elasticsearch):
    print(f"\n{'='*60}")
    print(f"Collecting S&P 500 — {len(tickers)} tickers")
    print(f"Period: {start} ~ {end}")
    print(f"Target: Elasticsearch index '{INDEX}'")
    print(f"{'='*60}")

    total_rows = 0
    failed = []

    for i, ticker in enumerate(tickers, 1):
        try:
            df = fetch_ohlcv(ticker, start, end)
            if df is None or df.empty:
                print(f"  [{i}/{len(tickers)}] {ticker}: no data")
                continue

            actions = list(build_actions(ticker, df))
            success, errors = bulk(client, actions, raise_on_error=False)
            if errors:
                print(f"  [{i}/{len(tickers)}] {ticker}: {success} ok, {len(errors)} errors")
            else:
                print(f"  [{i}/{len(tickers)}] {ticker}: {success} rows indexed")
            total_rows += success
            time.sleep(BATCH_SLEEP)

        except Exception as e:
            failed.append((ticker, str(e)))
            print(f"  [{i}/{len(tickers)}] {ticker}: FAILED — {e}")

    return total_rows, failed


def main():
    print("OHLCV Data Collection → Elasticsearch")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Tickers: {len(SP500_TICKERS)}\n")

    client = get_client()
    ensure_index(client)

    rows, failed = collect_sp500(SP500_TICKERS, START_DATE, END_DATE, client)

    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"  Total indexed: {rows} rows, {len(failed)} failures")

    if failed:
        print("\nFailed tickers:")
        for ticker, err in failed:
            print(f"  {ticker}: {err}")


if __name__ == "__main__":
    main()
