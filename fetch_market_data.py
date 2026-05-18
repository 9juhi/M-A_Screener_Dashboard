import yfinance as yf
import pandas as pd
import time
import os
from tqdm import tqdm
from config import RAW_DIR, BATCH_SIZE, BATCH_SLEEP

OUTPUT_PATH = f"{RAW_DIR}/market_data.csv"
FAILED_PATH = f"{RAW_DIR}/failed_market_tickers.csv"
MAX_RETRIES = 3
REQUEST_SLEEP = 5
CONSECUTIVE_FAILURE_LIMIT = 3


def get_fast_info_value(fast_info, *names):

    for name in names:
        try:
            if isinstance(fast_info, dict) and name in fast_info:
                return fast_info[name]

            value = getattr(fast_info, name)
            if value is not None:
                return value
        except Exception:
            continue

    return None


def fetch_ticker_data(ticker: str) -> dict:


    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(
                period="1y",
                auto_adjust=False,
                actions=False,
            )

            if history.empty or "Close" not in history:
                raise ValueError("No price history returned")

            close = history["Close"].dropna()
            high = history["High"].dropna() if "High" in history else close
            low = history["Low"].dropna() if "Low" in history else close

            if close.empty:
                raise ValueError("No close price returned")


            try:
                fast_info = stock.fast_info
            except Exception:
                fast_info = {}

            return {
                "ticker":             ticker,
                "price":              close.iloc[-1],
                "market_cap":         get_fast_info_value(fast_info, "marketCap", "market_cap"),
                "shares_outstanding": get_fast_info_value(fast_info, "shares"),
                "52w_high":           high.max() if not high.empty else None,
                "52w_low":            low.min() if not low.empty else None,
            }

        except Exception as e:
            error = str(e)
            is_rate_limit = "429" in error or "Too Many Requests" in error
            is_empty_response = "Expecting value" in error

            if attempt < MAX_RETRIES and (is_rate_limit or is_empty_response):
                wait_seconds = 300 * attempt
                print(f"  Yahoo throttled {ticker}; retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue


            print(f"  FAILED: {ticker} — {e}")
            return None


def fetch_all_market_data(tickers: list) -> pd.DataFrame:


    results = []
    failed  = []
    consecutive_failures = 0
    os.makedirs(RAW_DIR, exist_ok=True)

    for i, ticker in enumerate(tqdm(tickers, desc="Fetching market data")):
        data = fetch_ticker_data(ticker)

        if data:
            results.append(data)
            consecutive_failures = 0
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        else:
            failed.append(ticker)
            consecutive_failures += 1

        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            failed_df = pd.DataFrame({"ticker": failed})
            failed_df.to_csv(FAILED_PATH, index=False)
            raise RuntimeError(
                "Yahoo Finance is still returning empty/throttled responses. "
                f"Stopped after {CONSECUTIVE_FAILURE_LIMIT} consecutive failures "
                "to avoid burning through all 503 tickers. Wait longer or switch "
                "networks, then rerun this script."
            )

        time.sleep(REQUEST_SLEEP)

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_SLEEP)

    df = pd.DataFrame(results)


    if failed:
        failed_df = pd.DataFrame({"ticker": failed})
        failed_df.to_csv(FAILED_PATH, index=False)
        print(f"\n{len(failed)} tickers failed. Saved to failed_market_tickers.csv")

    return df


if __name__ == "__main__":
    sp500 = pd.read_csv(f"{RAW_DIR}/sp500_companies.csv")
    tickers = sp500["ticker"].tolist()
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        if "price" in existing.columns:
            existing = existing[existing["price"].notna()]
        completed = set(existing["ticker"].dropna())
        tickers = [ticker for ticker in tickers if ticker not in completed]
        print(f"Resuming from existing file with {len(completed)} completed tickers.")
    else:
        existing = pd.DataFrame()

    print(f"Starting market data fetch for {len(tickers)} companies...")

    new_df = fetch_all_market_data(tickers)
    df = pd.concat([existing, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["ticker"], keep="last")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} records to {OUTPUT_PATH}")
    print(df.head())
