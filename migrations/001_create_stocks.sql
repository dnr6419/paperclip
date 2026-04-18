CREATE TABLE IF NOT EXISTS stocks (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL UNIQUE,
    name        VARCHAR(100) NOT NULL,
    market      VARCHAR(10)  NOT NULL CHECK (market IN ('KOSPI','KOSDAQ')),
    sector      VARCHAR(100),
    industry    VARCHAR(100),
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    listed_at   DATE,
    delisted_at DATE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market);
CREATE INDEX IF NOT EXISTS idx_stocks_ticker ON stocks(ticker);
