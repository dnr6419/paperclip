"""
Synthetic Korean market (KRX) data generator.
Uses GBM calibrated to actual KOSPI annual returns.
For offline backtesting before pykrx real data collection completes.
"""
import numpy as np
import pandas as pd
from typing import Dict

ANNUAL_KOSPI_RETURNS = {2019: 0.078, 2020: 0.306, 2021: 0.037, 2022: -0.247, 2023: 0.187, 2024: -0.098}
ANNUAL_KOSPI_VOL     = {2019: 0.16,  2020: 0.30,  2021: 0.15,  2022: 0.24,   2023: 0.14,  2024: 0.18}

KRX_STOCK_CONFIGS = [
    {"name": "005930", "label": "삼성전자",       "beta": 1.0, "alpha": 0.03, "vol_idio": 0.25, "seed": 101, "start_price": 44000},
    {"name": "000660", "label": "SK하이닉스",     "beta": 1.4, "alpha": 0.05, "vol_idio": 0.35, "seed": 102, "start_price": 65000},
    {"name": "005380", "label": "현대차",         "beta": 1.1, "alpha": 0.02, "vol_idio": 0.28, "seed": 103, "start_price": 120000},
    {"name": "000270", "label": "기아",           "beta": 1.2, "alpha": 0.04, "vol_idio": 0.30, "seed": 104, "start_price": 42000},
    {"name": "035420", "label": "NAVER",          "beta": 1.3, "alpha": 0.06, "vol_idio": 0.32, "seed": 105, "start_price": 130000},
    {"name": "035720", "label": "카카오",         "beta": 1.3, "alpha": 0.04, "vol_idio": 0.35, "seed": 106, "start_price": 120000},
    {"name": "051910", "label": "LG화학",         "beta": 1.1, "alpha": 0.03, "vol_idio": 0.28, "seed": 107, "start_price": 350000},
    {"name": "006400", "label": "삼성SDI",        "beta": 1.2, "alpha": 0.05, "vol_idio": 0.30, "seed": 108, "start_price": 220000},
    {"name": "068270", "label": "셀트리온",       "beta": 0.8, "alpha": 0.03, "vol_idio": 0.32, "seed": 109, "start_price": 200000},
    {"name": "055550", "label": "신한지주",       "beta": 0.9, "alpha": 0.02, "vol_idio": 0.20, "seed": 110, "start_price": 42000},
    {"name": "105560", "label": "KB금융",         "beta": 0.9, "alpha": 0.02, "vol_idio": 0.20, "seed": 111, "start_price": 48000},
    {"name": "005490", "label": "POSCO홀딩스",    "beta": 1.0, "alpha": 0.01, "vol_idio": 0.26, "seed": 112, "start_price": 260000},
    {"name": "066570", "label": "LG전자",         "beta": 1.0, "alpha": 0.02, "vol_idio": 0.25, "seed": 113, "start_price": 65000},
    {"name": "009150", "label": "삼성전기",       "beta": 1.2, "alpha": 0.03, "vol_idio": 0.30, "seed": 114, "start_price": 100000},
    {"name": "033780", "label": "KT&G",           "beta": 0.4, "alpha": 0.02, "vol_idio": 0.14, "seed": 115, "start_price": 95000},
]


def generate_trading_dates_krx(start: str, end: str) -> pd.DatetimeIndex:
    dates = pd.bdate_range(start=start, end=end)
    kr_holidays = pd.to_datetime([
        "2019-01-01", "2019-02-04", "2019-02-05", "2019-02-06", "2019-03-01",
        "2019-05-06", "2019-06-06", "2019-08-15", "2019-09-12", "2019-09-13",
        "2019-10-03", "2019-10-09", "2019-12-25",
        "2020-01-01", "2020-01-24", "2020-01-27", "2020-03-01", "2020-04-15",
        "2020-04-30", "2020-05-05", "2020-06-06", "2020-08-17", "2020-09-30",
        "2020-10-01", "2020-10-02", "2020-10-09", "2020-12-25",
        "2021-01-01", "2021-02-11", "2021-02-12", "2021-03-01", "2021-05-05",
        "2021-05-19", "2021-06-06", "2021-08-16", "2021-09-20", "2021-09-21",
        "2021-10-04", "2021-10-11", "2021-12-31",
        "2022-01-31", "2022-02-01", "2022-02-02", "2022-03-01", "2022-03-09",
        "2022-05-05", "2022-06-01", "2022-06-06", "2022-08-15", "2022-09-09",
        "2022-09-12", "2022-10-03", "2022-10-10",
        "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-29",
        "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29", "2023-10-03",
        "2023-10-09", "2023-12-25",
        "2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01", "2024-04-10",
        "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
        "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25",
    ])
    return dates.difference(kr_holidays)


def simulate_returns_gbm(dates, ann_returns_by_year, ann_vol_by_year, seed=42):
    rng = np.random.RandomState(seed)
    daily_returns = []
    for date in dates:
        y = date.year
        mu = ann_returns_by_year.get(y, 0.05)
        sigma = ann_vol_by_year.get(y, 0.18)
        dt = 1 / 252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        z = rng.randn()
        daily_returns.append(drift + diffusion * z)
    return np.array(daily_returns)


def generate_stock_returns(dates, beta, alpha_ann, vol_idio, market_returns, seed):
    rng = np.random.RandomState(seed * 17 + 53)
    idio_sigma = vol_idio / np.sqrt(252)
    alpha_daily = alpha_ann / 252
    idio = rng.randn(len(dates)) * idio_sigma
    return alpha_daily + beta * market_returns + idio


def generate_ohlcv(price: pd.Series, ann_vol: float = 0.20) -> pd.DataFrame:
    rng = np.random.RandomState(int(abs(price.values[0]) * 10) % 10000 + 42)
    n = len(price)
    closes = price.values
    daily_vol = ann_vol / np.sqrt(252)
    opens, highs, lows, volumes = [], [], [], []
    for i in range(n):
        close = closes[i]
        prev = closes[i - 1] if i > 0 else close
        open_price = prev * np.exp(rng.normal(0, daily_vol * 0.4))
        ir = abs(rng.normal(0, daily_vol * 1.2))
        high = max(close, open_price) * (1 + ir * 0.7)
        low = min(close, open_price) * (1 - ir * 0.3)
        move = abs(close / prev - 1)
        vol_base = 3_000_000
        volume = int(vol_base * (1 + move * 25) * np.exp(rng.normal(0, 0.4)))
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(max(volume, 10_000))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=price.index,
    )


def generate_all_krx_data(start: str, end: str) -> Dict[str, pd.DataFrame]:
    dates = generate_trading_dates_krx(start, end)
    mkt_ret = simulate_returns_gbm(dates, ANNUAL_KOSPI_RETURNS, ANNUAL_KOSPI_VOL, seed=200)

    data = {}
    for config in KRX_STOCK_CONFIGS:
        s_ret = generate_stock_returns(
            dates, config["beta"], config["alpha"], config["vol_idio"], mkt_ret, config["seed"],
        )
        price = config["start_price"] * np.cumprod(1 + s_ret)
        ps = pd.Series(price, index=dates)
        data[config["name"]] = generate_ohlcv(ps, ann_vol=config["vol_idio"])

    kospi_price = 2040 * np.cumprod(1 + mkt_ret)
    data["KOSPI"] = generate_ohlcv(pd.Series(kospi_price, index=dates), ann_vol=0.16)
    return data


if __name__ == "__main__":
    data = generate_all_krx_data("2019-01-01", "2024-12-31")
    for name, df in data.items():
        s, e = df["close"].iloc[0], df["close"].iloc[-1]
        years = len(df) / 252
        cagr = ((e / s) ** (1 / years) - 1) * 100
        label = name
        print(f"  {label}: CAGR {cagr:+.1f}%")
