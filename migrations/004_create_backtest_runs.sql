CREATE TABLE IF NOT EXISTS backtest_runs (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name    VARCHAR(100) NOT NULL,
    strategy_version VARCHAR(30),
    start_date       DATE         NOT NULL,
    end_date         DATE         NOT NULL,
    universe         JSONB        NOT NULL DEFAULT '[]',
    parameters       JSONB        NOT NULL DEFAULT '{}',
    initial_capital  NUMERIC(18,2) NOT NULL,
    commission_rate  NUMERIC(8,6)  NOT NULL DEFAULT 0.00015,
    slippage_rate    NUMERIC(8,6)  NOT NULL DEFAULT 0.0,
    engine           VARCHAR(30)   DEFAULT 'vectorbt',
    status           VARCHAR(20)   NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running','completed','failed','cancelled')),
    error_message    TEXT,
    run_by           VARCHAR(100),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy ON backtest_runs(strategy_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON backtest_runs(created_at DESC);
