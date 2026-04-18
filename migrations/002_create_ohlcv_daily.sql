CREATE TABLE IF NOT EXISTS ohlcv_daily (
    id          BIGSERIAL PRIMARY KEY,
    stock_id    BIGINT      NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date        DATE        NOT NULL,
    open        NUMERIC(18,2) NOT NULL,
    high        NUMERIC(18,2) NOT NULL,
    low         NUMERIC(18,2) NOT NULL,
    close       NUMERIC(18,2) NOT NULL,
    volume      BIGINT       NOT NULL,
    adj_open    NUMERIC(18,4),
    adj_high    NUMERIC(18,4),
    adj_low     NUMERIC(18,4),
    adj_close   NUMERIC(18,4),
    adj_factor  NUMERIC(14,6) NOT NULL DEFAULT 1.0,
    source      VARCHAR(30),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (stock_id, date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_stock_date ON ohlcv_daily(stock_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_daily(date DESC);
