# explore_polygon_market_data.py
# --------------------------------------------------------
# Isolated Polygon.io exploration for market snapshot fields.
# This does not feed the main pipeline unless you explicitly decide to.
#
# Required:
#   export POLYGON_API_KEY="your_api_key_here"
#
# Output:
#   data/raw/polygon_market_data_sample.csv
# --------------------------------------------------------

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from config import RAW_DIR


API_KEY_ENV = "POLYGON_API_KEY"
BASE_URL = "https://api.polygon.io"
OUTPUT_PATH = f"{RAW_DIR}/polygon_market_data_sample.csv"
REQUEST_SLEEP = 13


def require_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing Polygon API key. Set it first with:\n"
            f'  export {API_KEY_ENV}="your_api_key_here"'
        )

    return api_key


def polygon_get(path: str, params: dict, api_key: str) -> dict:
    params = dict(params)
    params["apiKey"] = api_key

    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_ticker_overview(ticker: str, api_key: str) -> dict:
    data = polygon_get(f"/v3/reference/tickers/{ticker}", {}, api_key)
    return data.get("results", {})


def fetch_daily_bars(ticker: str, api_key: str) -> list:
    end_date = date.today()
    start_date = end_date - timedelta(days=370)

    data = polygon_get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}",
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        },
        api_key,
    )
    return data.get("results", [])


def fetch_polygon_market_row(ticker: str, api_key: str) -> dict:
    overview = fetch_ticker_overview(ticker, api_key)
    time.sleep(REQUEST_SLEEP)

    bars = fetch_daily_bars(ticker, api_key)
    if not bars:
        raise ValueError("No daily bars returned")

    closes = [bar["c"] for bar in bars if "c" in bar]
    highs = [bar["h"] for bar in bars if "h" in bar]
    lows = [bar["l"] for bar in bars if "l" in bar]

    return {
        "ticker": ticker,
        "price": closes[-1] if closes else None,
        "market_cap": overview.get("market_cap"),
        "shares_outstanding": overview.get("weighted_shares_outstanding"),
        "52w_high": max(highs) if highs else None,
        "52w_low": min(lows) if lows else None,
        "polygon_name": overview.get("name"),
        "polygon_primary_exchange": overview.get("primary_exchange"),
    }


def run_sample(limit: int = 10):
    api_key = require_api_key()
    sp500 = pd.read_csv(f"{RAW_DIR}/sp500_companies.csv")
    tickers = sp500["ticker"].head(limit).tolist()

    rows = []
    failed = []

    os.makedirs(RAW_DIR, exist_ok=True)

    for ticker in tqdm(tickers, desc="Exploring Polygon market data"):
        try:
            rows.append(fetch_polygon_market_row(ticker, api_key))
        except Exception as e:
            failed.append({"ticker": ticker, "error": str(e)})
            print(f"FAILED: {ticker} - {e}")

        time.sleep(REQUEST_SLEEP)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} sample rows to {OUTPUT_PATH}")
    if failed:
        print("Failures:")
        print(pd.DataFrame(failed).to_string(index=False))

    return df


if __name__ == "__main__":
    run_sample(limit=10)
