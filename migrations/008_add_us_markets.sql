-- Extend stocks.market to accept NYSE and NASDAQ
ALTER TABLE stocks DROP CONSTRAINT IF EXISTS stocks_market_check;
ALTER TABLE stocks ADD CONSTRAINT stocks_market_check
    CHECK (market IN ('KOSPI', 'KOSDAQ', 'NYSE', 'NASDAQ'));
