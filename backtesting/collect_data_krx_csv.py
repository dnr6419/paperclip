"""
Korean market (KOSPI/KOSDAQ) OHLCV data collector using pykrx.
Saves to CSV files (DB-independent) for immediate backtesting.
"""
import os
import sys
import time

import pandas as pd
from pykrx import stock as krx

START_DATE = "20190101"
END_DATE = "20241231"
BATCH_SLEEP = 1.5
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_krx")

KRX_TICKERS = [
    ("005930", "삼성전자", "KOSPI"),
    ("000660", "SK하이닉스", "KOSPI"),
    ("207940", "삼성바이오로직스", "KOSPI"),
    ("005380", "현대차", "KOSPI"),
    ("000270", "기아", "KOSPI"),
    ("006400", "삼성SDI", "KOSPI"),
    ("051910", "LG화학", "KOSPI"),
    ("035420", "NAVER", "KOSPI"),
    ("005490", "POSCO홀딩스", "KOSPI"),
    ("055550", "신한지주", "KOSPI"),
    ("105560", "KB금융", "KOSPI"),
    ("086790", "하나금융지주", "KOSPI"),
    ("316140", "우리금융지주", "KOSPI"),
    ("000810", "삼성화재", "KOSPI"),
    ("010130", "고려아연", "KOSPI"),
    ("012330", "현대모비스", "KOSPI"),
    ("009150", "삼성전기", "KOSPI"),
    ("034730", "SK", "KOSPI"),
    ("003670", "포스코퓨처엠", "KOSPI"),
    ("030200", "KT", "KOSPI"),
    ("017670", "SK텔레콤", "KOSPI"),
    ("032830", "삼성생명", "KOSPI"),
    ("066570", "LG전자", "KOSPI"),
    ("003550", "LG", "KOSPI"),
    ("028260", "삼성물산", "KOSPI"),
    ("010950", "S-Oil", "KOSPI"),
    ("036570", "엔씨소프트", "KOSPI"),
    ("011200", "HMM", "KOSPI"),
    ("034020", "두산에너빌리티", "KOSPI"),
    ("000720", "현대건설", "KOSPI"),
    ("018260", "삼성에스디에스", "KOSPI"),
    ("033780", "KT&G", "KOSPI"),
    ("015760", "한국전력", "KOSPI"),
    ("009540", "한국조선해양", "KOSPI"),
    ("068270", "셀트리온", "KOSPI"),
    ("326030", "SK바이오팜", "KOSPI"),
    ("352820", "하이브", "KOSPI"),
    ("259960", "크래프톤", "KOSPI"),
    ("003490", "대한항공", "KOSPI"),
    ("180640", "한진칼", "KOSPI"),
    ("035720", "카카오", "KOSPI"),
    ("041510", "에스엠", "KOSPI"),
    ("247540", "에코프로비엠", "KOSDAQ"),
    ("086520", "에코프로", "KOSDAQ"),
    ("035900", "JYP Ent.", "KOSDAQ"),
    ("263750", "펄어비스", "KOSDAQ"),
    ("293490", "카카오게임즈", "KOSDAQ"),
    ("112040", "위메이드", "KOSDAQ"),
    ("069500", "KODEX 200", "KOSPI"),
    ("122630", "KODEX 레버리지", "KOSPI"),
]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = krx.get_market_ohlcv_by_date(start, end, ticker)
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "시가": "open", "고가": "high", "저가": "low",
        "종가": "close", "거래량": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "date"
    df = df[df["volume"] > 0]
    return df


def collect_to_csv():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"Collecting KRX OHLCV → CSV — {len(KRX_TICKERS)} tickers")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print(f"Output: {DATA_DIR}/")
    print(f"{'='*60}")

    success = 0
    failed = []
    meta_rows = []

    for i, (ticker, name, market) in enumerate(KRX_TICKERS, 1):
        try:
            df = fetch_ohlcv(ticker, START_DATE, END_DATE)
            if df is None or df.empty:
                print(f"  [{i}/{len(KRX_TICKERS)}] {ticker} ({name}): no data")
                failed.append((ticker, "no data"))
                continue

            csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            df.to_csv(csv_path)
            success += 1
            meta_rows.append({
                "ticker": ticker, "name": name, "market": market,
                "rows": len(df),
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d"),
            })
            print(f"  [{i}/{len(KRX_TICKERS)}] {ticker} ({name}): {len(df)} rows")
            time.sleep(BATCH_SLEEP)

        except Exception as e:
            failed.append((ticker, str(e)))
            print(f"  [{i}/{len(KRX_TICKERS)}] {ticker} ({name}): FAILED — {e}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(DATA_DIR, "_meta.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"  Success: {success}/{len(KRX_TICKERS)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print("\nFailed tickers:")
        for ticker, err in failed:
            print(f"  {ticker}: {err}")

    return success, failed


if __name__ == "__main__":
    collect_to_csv()
