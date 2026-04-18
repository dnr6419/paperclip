CREATE TABLE IF NOT EXISTS corporate_actions (
    id          BIGSERIAL PRIMARY KEY,
    stock_id    BIGINT      NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    action_date DATE        NOT NULL,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('split','reverse_split','dividend','rights')),
    split_ratio NUMERIC(10,4),
    div_amount  NUMERIC(18,2),
    adj_factor  NUMERIC(14,6),
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_stock_date ON corporate_actions(stock_id, action_date DESC);
