CREATE TABLE IF NOT EXISTS backtest_equity_curve (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID  NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    date            DATE  NOT NULL,
    portfolio_value NUMERIC(18,2) NOT NULL,
    cash            NUMERIC(18,2) NOT NULL,
    invested        NUMERIC(18,2) NOT NULL,
    daily_return    NUMERIC(12,6),
    drawdown        NUMERIC(12,6),
    UNIQUE (run_id, date)
);

CREATE INDEX IF NOT EXISTS idx_equity_run_date ON backtest_equity_curve(run_id, date);
