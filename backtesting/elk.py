"""
Elasticsearch connection and OHLCV query utilities.
Reads config from environment variables.

Required env vars for the external ES instance:
  ELK_HOST      host (default: localhost)
  ELK_PORT      port (default: 9200)
  ELK_SCHEME    http or https (default: https)
  ELK_USER      basic-auth username (default: elastic)
  ELK_PASSWORD  basic-auth password (default: "")
"""
import os
import urllib3
import pandas as pd
from elasticsearch import Elasticsearch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INDEX = "ohlcv-daily"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "ticker": {"type": "keyword"},
            "date": {"type": "date", "format": "yyyy-MM-dd"},
            "open": {"type": "double"},
            "high": {"type": "double"},
            "low": {"type": "double"},
            "close": {"type": "double"},
            "volume": {"type": "long"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

ELK_CONFIG = {
    "host": os.getenv("ELK_HOST", "localhost"),
    "port": int(os.getenv("ELK_PORT", "9200")),
    "scheme": os.getenv("ELK_SCHEME", "https"),
    "user": os.getenv("ELK_USER", "elastic"),
    "password": os.getenv("ELK_PASSWORD", ""),
}


def get_client() -> Elasticsearch:
    url = f"{ELK_CONFIG['scheme']}://{ELK_CONFIG['host']}:{ELK_CONFIG['port']}"
    kwargs = {"verify_certs": False, "ssl_show_warn": False}
    if ELK_CONFIG["user"]:
        kwargs["basic_auth"] = (ELK_CONFIG["user"], ELK_CONFIG["password"])
    return Elasticsearch(url, **kwargs)


def ensure_index(client: Elasticsearch = None) -> None:
    """Create ohlcv-daily index with mapping if it doesn't exist."""
    if client is None:
        client = get_client()
    if not client.indices.exists(index=INDEX):
        client.indices.create(index=INDEX, body=INDEX_MAPPING)
        print(f"Created index: {INDEX}")
    else:
        print(f"Index already exists: {INDEX}")


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    client: Elasticsearch = None,
) -> pd.DataFrame:
    """
    Query OHLCV data for a ticker between start and end dates (inclusive).

    Returns a DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    Returns empty DataFrame if no data found.
    """
    if client is None:
        client = get_client()

    query = {
        "size": 10000,
        "sort": [{"date": {"order": "asc"}}],
        "query": {
            "bool": {
                "filter": [
                    {"term": {"ticker": ticker}},
                    {"range": {"date": {"gte": start, "lte": end}}},
                ]
            }
        },
    }

    resp = client.search(index=INDEX, body=query)
    hits = resp["hits"]["hits"]
    if not hits:
        return pd.DataFrame()

    records = [h["_source"] for h in hits]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": int})
    return df


def fetch_tickers(client: Elasticsearch = None) -> list[str]:
    """Return all distinct tickers loaded in the index."""
    if client is None:
        client = get_client()

    resp = client.search(
        index=INDEX,
        body={
            "size": 0,
            "aggs": {"tickers": {"terms": {"field": "ticker", "size": 1000}}},
        },
    )
    return [b["key"] for b in resp["aggregations"]["tickers"]["buckets"]]
