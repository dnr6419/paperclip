"""
Database connection and migration utilities.
Reads config from environment variables or .env file.
Supports PostgreSQL and Elasticsearch dual-write.
"""
import os
import glob
import psycopg2
from contextlib import contextmanager
from elasticsearch import Elasticsearch

DB_CONFIG = {
    "host": os.getenv("BACKTEST_DB_HOST", "localhost"),
    "port": int(os.getenv("BACKTEST_DB_PORT", "5432")),
    "dbname": os.getenv("BACKTEST_DB_NAME", "backtesting"),
    "user": os.getenv("BACKTEST_DB_USER", "backtest"),
    "password": os.getenv("BACKTEST_DB_PASSWORD", "backtest"),
}

ES_CONFIG = {
    "hosts": [os.getenv("BACKTEST_ES_URL", "https://192.168.45.75:9200")],
    "basic_auth": (
        os.getenv("BACKTEST_ES_USER", "elastic"),
        os.getenv("BACKTEST_ES_PASSWORD", "DOF2026!"),
    ),
    "verify_certs": False,
}

OHLCV_INDEX = "ohlcv_daily"
STOCKS_INDEX = "stocks"


def get_es_client() -> Elasticsearch:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return Elasticsearch(**ES_CONFIG)


def ensure_es_indices(es: Elasticsearch):
    if not es.indices.exists(index=STOCKS_INDEX):
        es.indices.create(index=STOCKS_INDEX, body={
            "mappings": {
                "properties": {
                    "ticker": {"type": "keyword"},
                    "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "market": {"type": "keyword"},
                    "sector": {"type": "keyword"},
                    "industry": {"type": "keyword"},
                    "is_active": {"type": "boolean"},
                }
            }
        })

    if not es.indices.exists(index=OHLCV_INDEX):
        es.indices.create(index=OHLCV_INDEX, body={
            "mappings": {
                "properties": {
                    "ticker": {"type": "keyword"},
                    "stock_id": {"type": "long"},
                    "date": {"type": "date", "format": "yyyy-MM-dd"},
                    "open": {"type": "float"},
                    "high": {"type": "float"},
                    "low": {"type": "float"},
                    "close": {"type": "float"},
                    "volume": {"type": "long"},
                    "adj_close": {"type": "float"},
                    "adj_factor": {"type": "float"},
                    "source": {"type": "keyword"},
                }
            }
        })


@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations(migrations_dir: str = None):
    if migrations_dir is None:
        migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")

    sql_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    if not sql_files:
        print(f"No migration files found in {migrations_dir}")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("SELECT filename FROM _migrations")
        applied = {row[0] for row in cur.fetchall()}

        for filepath in sql_files:
            filename = os.path.basename(filepath)
            if filename in applied:
                print(f"  Skip (already applied): {filename}")
                continue

            print(f"  Applying: {filename}")
            with open(filepath) as f:
                sql = f.read()
            cur.execute(sql)
            cur.execute("INSERT INTO _migrations (filename) VALUES (%s)", (filename,))

        conn.commit()
    print("Migrations complete.")


if __name__ == "__main__":
    run_migrations()
