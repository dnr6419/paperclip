"""
Synthetic leveraged ETF data generator for SOXL (3x Semiconductors) and TQQQ (3x NASDAQ-100).
Models daily-rebalanced 3x leverage with volatility decay.
"""
import numpy as np
import pandas as pd
from typing import Dict
from backtesting.generate_data import generate_trading_dates, simulate_returns_gbm, generate_ohlcv

NASDAQ_RETURNS = {2019: 0.358, 2020: 0.435, 2021: 0.213, 2022: -0.329, 2023: 0.437, 2024: 0.285}
NASDAQ_VOL     = {2019: 0.17,  2020: 0.36,  2021: 0.19,  2022: 0.30,   2023: 0.19,  2024: 0.17}

SOX_RETURNS = {2019: 0.607, 2020: 0.517, 2021: 0.413, 2022: -0.356, 2023: 0.652, 2024: 0.193}
SOX_VOL     = {2019: 0.22,  2020: 0.40,  2021: 0.24,  2022: 0.35,   2023: 0.24,  2024: 0.22}

LEVERAGED_ETFS = {
    "TQQQ": {
        "underlying_returns": NASDAQ_RETURNS,
        "underlying_vol": NASDAQ_VOL,
        "leverage": 3,
        "expense_ratio": 0.0086,
        "start_price": 20.0,
        "seed": 100,
    },
    "SOXL": {
        "underlying_returns": SOX_RETURNS,
        "underlying_vol": SOX_VOL,
        "leverage": 3,
        "expense_ratio": 0.0076,
        "start_price": 15.0,
        "seed": 200,
    },
}


def generate_leveraged_etf(dates, config) -> pd.DataFrame:
    underlying_ret = simulate_returns_gbm(
        dates,
        config["underlying_returns"],
        config["underlying_vol"],
        seed=config["seed"],
    )
    leverage = config["leverage"]
    daily_expense = config["expense_ratio"] / 252

    leveraged_ret = leverage * underlying_ret - daily_expense

    price = config["start_price"] * np.cumprod(1 + leveraged_ret)
    ps = pd.Series(price, index=dates)

    avg_vol = np.mean(list(config["underlying_vol"].values())) * leverage
    return generate_ohlcv(ps, ann_vol=avg_vol)


def generate_leveraged_etf_data(start: str, end: str) -> Dict[str, pd.DataFrame]:
    dates = generate_trading_dates(start, end)
    data = {}
    for name, config in LEVERAGED_ETFS.items():
        data[name] = generate_leveraged_etf(dates, config)
    return data


if __name__ == "__main__":
    data = generate_leveraged_etf_data("2019-01-01", "2024-12-31")
    for name, df in data.items():
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        years = len(df) / 252
        cagr = ((e / s) ** (1 / years) - 1) * 100
        vol = df["close"].pct_change().std() * np.sqrt(252) * 100
        print(f"  {name}: start=${s:.2f} end=${e:.2f} CAGR={cagr:+.1f}% AnnVol={vol:.1f}%")
