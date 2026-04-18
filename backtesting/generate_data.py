"""
Synthetic KOSPI/KOSDAQ market data generator.
Uses GBM calibrated to actual Korean market statistics.
"""
import numpy as np
import pandas as pd
from typing import Dict

# Actual KOSPI annual total returns (approximate)
ANNUAL_MARKET_RETURNS = {2019: 0.077, 2020: 0.308, 2021: 0.036, 2022: -0.249, 2023: 0.187, 2024: -0.096}
ANNUAL_MARKET_VOL     = {2019: 0.14,  2020: 0.32,  2021: 0.18,  2022: 0.22,   2023: 0.16,  2024: 0.17}

STOCK_CONFIGS = [
    {"name": "Samsung",    "beta": 1.1, "alpha": 0.03,  "vol_idio": 0.18, "seed": 1},
    {"name": "SKHynix",    "beta": 1.3, "alpha": 0.06,  "vol_idio": 0.26, "seed": 2},
    {"name": "NAVER",      "beta": 1.1, "alpha": 0.04,  "vol_idio": 0.22, "seed": 3},
    {"name": "Hyundai",    "beta": 1.0, "alpha": 0.02,  "vol_idio": 0.20, "seed": 4},
    {"name": "LGChem",     "beta": 1.2, "alpha": 0.04,  "vol_idio": 0.24, "seed": 5},
    {"name": "SamsungSDI", "beta": 1.2, "alpha": 0.05,  "vol_idio": 0.28, "seed": 6},
    {"name": "Kakao",      "beta": 1.1, "alpha": 0.03,  "vol_idio": 0.30, "seed": 7},
    {"name": "SamsungCT",  "beta": 0.8, "alpha": 0.01,  "vol_idio": 0.16, "seed": 8},
    {"name": "KBFinancial","beta": 0.9, "alpha": 0.02,  "vol_idio": 0.18, "seed": 9},
    {"name": "Shinhan",    "beta": 0.8, "alpha": 0.02,  "vol_idio": 0.17, "seed": 10},
    {"name": "SKTelecom",  "beta": 0.6, "alpha": 0.02,  "vol_idio": 0.13, "seed": 11},
    {"name": "KT",         "beta": 0.7, "alpha": 0.01,  "vol_idio": 0.15, "seed": 12},
    {"name": "LGCorp",     "beta": 0.9, "alpha": 0.02,  "vol_idio": 0.16, "seed": 13},
    {"name": "SamsungBio", "beta": 1.0, "alpha": 0.07,  "vol_idio": 0.28, "seed": 14},
    {"name": "LGElec",     "beta": 1.0, "alpha": 0.02,  "vol_idio": 0.22, "seed": 15},
]


def generate_trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def simulate_returns_gbm(dates, ann_returns_by_year, ann_vol_by_year, seed=42):
    """Generate daily returns using GBM, calibrated to annual targets."""
    rng = np.random.RandomState(seed)
    daily_returns = []
    for date in dates:
        y = date.year
        mu = ann_returns_by_year.get(y, 0.05)
        sigma = ann_vol_by_year.get(y, 0.18)
        dt = 1/252
        # GBM: r = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
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
    rng = np.random.RandomState(int(price.values[0]) % 10000 + 42)
    n = len(price)
    closes = price.values
    daily_vol = ann_vol / np.sqrt(252)
    opens, highs, lows, volumes = [], [], [], []
    for i in range(n):
        close = closes[i]
        prev = closes[i-1] if i > 0 else close
        # Open: small gap from prev close
        open_price = prev * np.exp(rng.normal(0, daily_vol * 0.4))
        # High/Low: based on intraday range
        ir = abs(rng.normal(0, daily_vol * 1.2))
        high = max(close, open_price) * (1 + ir * 0.7)
        low  = min(close, open_price) * (1 - ir * 0.3)
        # Volume
        move = abs(close / prev - 1)
        vol_base = 500_000
        volume = int(vol_base * (1 + move * 25) * np.exp(rng.normal(0, 0.4)))
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(max(volume, 1000))
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=price.index)


def generate_all_data(start: str, end: str) -> Dict[str, pd.DataFrame]:
    dates = generate_trading_dates(start, end)
    mkt_ret = simulate_returns_gbm(dates, ANNUAL_MARKET_RETURNS, ANNUAL_MARKET_VOL, seed=42)

    data = {}
    for config in STOCK_CONFIGS:
        s_ret = generate_stock_returns(dates, config["beta"], config["alpha"], config["vol_idio"], mkt_ret, config["seed"])
        price = 10000 * np.cumprod(1 + s_ret)
        ps = pd.Series(price, index=dates)
        data[config["name"]] = generate_ohlcv(ps, ann_vol=config["vol_idio"])

    # Market index (starting at KOSPI level ~2200)
    mkt_price = 2200 * np.cumprod(1 + mkt_ret)
    data["KOSPI"] = generate_ohlcv(pd.Series(mkt_price, index=dates), ann_vol=0.14)
    return data


if __name__ == "__main__":
    data = generate_all_data("2019-01-01", "2024-12-31")
    for name, df in data.items():
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        years = len(df)/252
        cagr = ((e/s)**(1/years)-1)*100
        print(f"  {name}: CAGR {cagr:+.1f}%")
