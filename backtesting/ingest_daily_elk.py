"""
Daily incremental OHLCV ingestion into Elasticsearch.

Fetches the previous trading day's data for all tracked tickers and
bulk-upserts into the ohlcv-daily index. Sends a Slack notification
on completion (success or failure) via SLACK_WEBHOOK_URL env var.

Usage:
    python backtesting/ingest_daily_elk.py                   # ingest yesterday via yfinance
    python backtesting/ingest_daily_elk.py --date 2024-12-31
    python backtesting/ingest_daily_elk.py --gap-check       # backfill missing dates
    python backtesting/ingest_daily_elk.py --from-ticker-prices  # sync from ticker-prices-* indices
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from elasticsearch.helpers import bulk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.collect_data_elk import SP500_TICKERS, fetch_ohlcv
from backtesting.elk import INDEX, ensure_index, get_client

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def send_slack(message: str) -> None:
    if not SLACK_WEBHOOK_URL:
        print(f"[slack] SLACK_WEBHOOK_URL not set, skipping notification")
        return
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"[slack] unexpected status {resp.status}")
    except Exception as e:
        print(f"[slack] failed to send notification: {e}")


def prev_trading_day(ref: date) -> date:
    """Return the most recent weekday before ref."""
    d = ref - timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def get_indexed_dates(client, ticker: str) -> set[str]:
    """Return the set of dates already indexed for a ticker."""
    resp = client.search(
        index=INDEX,
        body={
            "size": 0,
            "query": {"term": {"ticker": ticker}},
            "aggs": {"dates": {"terms": {"field": "date", "size": 5000}}},
        },
    )
    return {b["key_as_string"][:10] for b in resp["aggregations"]["dates"]["buckets"]}


def ingest_date_range(client, tickers: list[str], start: str, end: str) -> dict:
    """Ingest OHLCV for all tickers between start and end. Returns stats."""
    stats = {"indexed": 0, "skipped": 0, "failed": []}

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, start, end)
            if df is None or df.empty:
                stats["skipped"] += 1
                print(f"  {ticker}: no data for {start}~{end}")
                continue

            actions = [
                {
                    "_index": INDEX,
                    "_id": f"{ticker}_{date_val.strftime('%Y-%m-%d')}",
                    "_source": {
                        "ticker": ticker,
                        "date": date_val.strftime("%Y-%m-%d"),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    },
                }
                for date_val, row in df.iterrows()
            ]
            success, errors = bulk(client, actions, raise_on_error=False)
            stats["indexed"] += success
            if errors:
                print(f"  {ticker}: {success} ok, {len(errors)} errors")
            else:
                print(f"  {ticker}: {success} rows indexed")

        except Exception as e:
            stats["failed"].append((ticker, str(e)))
            print(f"  {ticker}: FAILED — {e}")

    return stats


def gap_check_and_backfill(client, tickers: list[str], start: str, end: str) -> dict:
    """Find and backfill missing dates for each ticker."""
    # Build expected trading days from yfinance market calendar approximation
    all_days = pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()
    stats = {"indexed": 0, "skipped": 0, "failed": []}

    for ticker in tickers:
        indexed = get_indexed_dates(client, ticker)
        missing = [d for d in all_days if d not in indexed]
        if not missing:
            print(f"  {ticker}: no gaps")
            stats["skipped"] += 1
            continue

        print(f"  {ticker}: {len(missing)} missing dates, backfilling...")
        s = ingest_date_range(client, [ticker], missing[0], missing[-1])
        stats["indexed"] += s["indexed"]
        stats["failed"].extend(s["failed"])

    return stats


def sync_from_ticker_prices(client) -> dict:
    """
    Sync OHLCV data from existing ticker-prices-{date} indices into ohlcv-daily.

    ticker-prices-* fields: symbol, trade_date, open, high, low, last (=close), volume
    Filters to SP500_TICKERS only.
    """
    ticker_set = set(SP500_TICKERS)
    stats = {"indexed": 0, "skipped": 0, "failed": []}

    # Discover all ticker-prices-* indices
    resp = client.cat.indices(index="ticker-prices-*", h="index", format="json")
    price_indices = sorted([r["index"] for r in resp])
    print(f"Found {len(price_indices)} ticker-prices-* indices: {price_indices[0]} ~ {price_indices[-1]}")

    for idx_name in price_indices:
        try:
            result = client.search(
                index=idx_name,
                body={
                    "size": 10000,
                    "query": {"terms": {"symbol": list(ticker_set)}},
                    "_source": ["symbol", "trade_date", "open", "high", "low", "last", "volume"],
                },
            )
            hits = result["hits"]["hits"]
            if not hits:
                print(f"  {idx_name}: no matching tickers")
                stats["skipped"] += 1
                continue

            actions = []
            for h in hits:
                s = h["_source"]
                ticker = s.get("symbol", "")
                trade_date = s.get("trade_date", "")
                if not ticker or not trade_date:
                    continue
                actions.append({
                    "_index": INDEX,
                    "_id": f"{ticker}_{trade_date}",
                    "_source": {
                        "ticker": ticker,
                        "date": trade_date,
                        "open": float(s.get("open") or 0),
                        "high": float(s.get("high") or 0),
                        "low": float(s.get("low") or 0),
                        "close": float(s.get("last") or 0),
                        "volume": int(s.get("volume") or 0),
                    },
                })

            success, errors = bulk(client, actions, raise_on_error=False)
            stats["indexed"] += success
            print(f"  {idx_name}: {success} rows indexed" + (f", {len(errors)} errors" if errors else ""))

        except Exception as e:
            stats["failed"].append((idx_name, str(e)))
            print(f"  {idx_name}: FAILED — {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Daily OHLCV ELK ingestion")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--gap-check", action="store_true", help="Backfill missing dates since 2019-01-01")
    parser.add_argument("--from-ticker-prices", action="store_true",
                        help="Sync from existing ticker-prices-* indices into ohlcv-daily")
    args = parser.parse_args()

    client = get_client()
    ensure_index(client)

    if args.from_ticker_prices:
        print("Syncing from ticker-prices-* indices...")
        stats = sync_from_ticker_prices(client)
        mode = "ticker-prices-sync"
    elif args.gap_check:
        target_start = "2019-01-01"
        target_end = date.today().strftime("%Y-%m-%d")
        print(f"Gap check & backfill: {target_start} ~ {target_end}")
        stats = gap_check_and_backfill(client, SP500_TICKERS, target_start, target_end)
        mode = "gap-check"
    else:
        if args.date:
            target_date = args.date
        else:
            target_date = prev_trading_day(date.today()).strftime("%Y-%m-%d")

        print(f"Daily ingestion for {target_date} ({len(SP500_TICKERS)} tickers)")
        stats = ingest_date_range(client, SP500_TICKERS, target_date, target_date)
        mode = f"daily ({target_date})"

    # Summary
    failed_count = len(stats["failed"])
    print(f"\nDone — indexed: {stats['indexed']}, skipped: {stats['skipped']}, failed: {failed_count}")

    if failed_count == 0:
        msg = (
            f":white_check_mark: *ELK OHLCV 일봉 적재 완료* [{mode}]\n"
            f"• 인덱싱: {stats['indexed']}건 | 스킵: {stats['skipped']}건 | 실패: 0건\n"
            f"• 인덱스: `{INDEX}`"
        )
    else:
        failed_list = "\n".join(f"  - {t}: {e}" for t, e in stats["failed"])
        msg = (
            f":warning: *ELK OHLCV 적재 일부 실패* [{mode}]\n"
            f"• 인덱싱: {stats['indexed']}건 | 실패: {failed_count}건\n"
            f"• 실패 종목:\n{failed_list}"
        )

    send_slack(msg)


if __name__ == "__main__":
    main()
