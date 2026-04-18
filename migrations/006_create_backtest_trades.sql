CREATE TABLE IF NOT EXISTS backtest_trades (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID    NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    stock_id     BIGINT  NOT NULL REFERENCES stocks(id),
    direction    VARCHAR(5) NOT NULL CHECK (direction IN ('long','short')),
    entry_date   DATE    NOT NULL,
    entry_price  NUMERIC(18,4) NOT NULL,
    exit_date    DATE,
    exit_price   NUMERIC(18,4),
    quantity     INTEGER NOT NULL,
    pnl          NUMERIC(18,2),
    pnl_pct      NUMERIC(12,6),
    commission   NUMERIC(18,2),
    holding_days INTEGER,
    exit_reason  VARCHAR(30) CHECK (exit_reason IN
                     ('signal','stop_loss','take_profit','trailing_stop','end_of_period','delisted')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_run_id ON backtest_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_run_stock ON backtest_trades(run_id, stock_id);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON backtest_trades(entry_date);
