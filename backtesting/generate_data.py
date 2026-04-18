"""
Synthetic S&P 500 market data generator.
Uses GBM calibrated to actual S&P 500 annual returns.
"""
import numpy as np
import pandas as pd
from typing import Dict

# S&P 500 annual total returns
ANNUAL_MARKET_RETURNS = {2019: 0.289, 2020: 0.163, 2021: 0.269, 2022: -0.194, 2023: 0.242, 2024: 0.233}
ANNUAL_MARKET_VOL     = {2019: 0.15,  2020: 0.34,  2021: 0.17,  2022: 0.26,   2023: 0.17,  2024: 0.15}

STOCK_CONFIGS = [
    {"name": "AAPL",  "beta": 1.2, "alpha": 0.05,  "vol_idio": 0.22, "seed": 1,  "start_price": 40.0},
    {"name": "MSFT",  "beta": 1.1, "alpha": 0.06,  "vol_idio": 0.20, "seed": 2,  "start_price": 100.0},
    {"name": "NVDA",  "beta": 1.5, "alpha": 0.12,  "vol_idio": 0.40, "seed": 3,  "start_price": 35.0},
    {"name": "GOOGL", "beta": 1.1, "alpha": 0.04,  "vol_idio": 0.22, "seed": 4,  "start_price": 1050.0},
    {"name": "META",  "beta": 1.2, "alpha": 0.05,  "vol_idio": 0.30, "seed": 5,  "start_price": 130.0},
    {"name": "AMZN",  "beta": 1.2, "alpha": 0.07,  "vol_idio": 0.28, "seed": 6,  "start_price": 85.0},
    {"name": "TSLA",  "beta": 1.8, "alpha": 0.10,  "vol_idio": 0.55, "seed": 7,  "start_price": 20.0},
    {"name": "JPM",   "beta": 1.1, "alpha": 0.03,  "vol_idio": 0.22, "seed": 8,  "start_price": 100.0},
    {"name": "V",     "beta": 0.9, "alpha": 0.04,  "vol_idio": 0.18, "seed": 9,  "start_price": 140.0},
    {"name": "JNJ",   "beta": 0.6, "alpha": 0.02,  "vol_idio": 0.14, "seed": 10, "start_price": 130.0},
    {"name": "UNH",   "beta": 0.8, "alpha": 0.05,  "vol_idio": 0.18, "seed": 11, "start_price": 250.0},
    {"name": "XOM",   "beta": 0.9, "alpha": 0.02,  "vol_idio": 0.24, "seed": 12, "start_price": 70.0},
    {"name": "WMT",   "beta": 0.5, "alpha": 0.03,  "vol_idio": 0.14, "seed": 13, "start_price": 95.0},
    {"name": "MA",    "beta": 1.0, "alpha": 0.04,  "vol_idio": 0.19, "seed": 14, "start_price": 200.0},
    {"name": "PG",    "beta": 0.4, "alpha": 0.02,  "vol_idio": 0.12, "seed": 15, "start_price": 90.0},
]


def generate_trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def simulate_returns_gbm(dates, ann_returns_by_year, ann_vol_by_year, seed=42):
    """Generate daily returns using GBM, calibrated to annual targets."""
    rng = np.random.RandomState(seed)
    daily_returns = []
    for date in dates:
        y = date.year
        mu = ann_returns_by_year.get(y, 0.10)
        sigma = ann_vol_by_year.get(y, 0.18)
        dt = 1/252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        z = rng.randn()
        ret = drift + diffusion * z
        daily_returns.append(ret)
    return np.array(daily_returns)


def generate_stock_returns(dates, beta, alpha_ann, vol_idio, market_returns, seed):
    """Stock returns = alpha + beta * market + idiosyncratic."""
    rng = np.random.RandomState(seed * 13 + 77)
    idio_sigma = vol_idio / np.sqrt(252)
    alpha_daily = alpha_ann / 252
    idio = rng.randn(len(dates)) * idio_sigma
    stock_ret = alpha_daily + beta * market_returns + idio
    return stock_ret


def generate_ohlcv(price: pd.Series, ann_vol: float = 0.20) -> pd.DataFrame:
    rng = np.random.RandomState(int(abs(price.values[0]) * 10) % 10000 + 42)
    n = len(price)
    closes = price.values
    daily_vol = ann_vol / np.sqrt(252)
    opens, highs, lows, volumes = [], [], [], []
    for i in range(n):
        close = closes[i]
        prev = closes[i-1] if i > 0 else close
        open_price = prev * np.exp(rng.normal(0, daily_vol * 0.4))
        ir = abs(rng.normal(0, daily_vol * 1.2))
        high = max(close, open_price) * (1 + ir * 0.7)
        low  = min(close, open_price) * (1 - ir * 0.3)
        move = abs(close / prev - 1)
        vol_base = 5_000_000
        volume = int(vol_base * (1 + move * 25) * np.exp(rng.normal(0, 0.4)))
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(max(volume, 10_000))
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=price.index)


def generate_all_data(start: str, end: str) -> Dict[str, pd.DataFrame]:
    dates = generate_trading_dates(start, end)
    mkt_ret = simulate_returns_gbm(dates, ANNUAL_MARKET_RETURNS, ANNUAL_MARKET_VOL, seed=42)

    data = {}
    for config in STOCK_CONFIGS:
        s_ret = generate_stock_returns(dates, config["beta"], config["alpha"], config["vol_idio"], mkt_ret, config["seed"])
        price = config["start_price"] * np.cumprod(1 + s_ret)
        ps = pd.Series(price, index=dates)
        data[config["name"]] = generate_ohlcv(ps, ann_vol=config["vol_idio"])

    # S&P 500 index starting ~2500 (approximate level at start of 2019)
    mkt_price = 2500 * np.cumprod(1 + mkt_ret)
    data["SP500"] = generate_ohlcv(pd.Series(mkt_price, index=dates), ann_vol=0.15)
    return data


if __name__ == "__main__":
    data = generate_all_data("2019-01-01", "2024-12-31")
    for name, df in data.items():
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        years = len(df)/252
        cagr = ((e/s)**(1/years)-1)*100
        print(f"  {name}: CAGR {cagr:+.1f}%")
