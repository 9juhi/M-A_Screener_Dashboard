# fetch_sp500.py
# ─────────────────────────────────────────────────────────
# Pulls the S&P 500 constituent list from Wikipedia.
# Output: data/raw/sp500_companies.csv
#
# Columns we care about:
#   Symbol   → ticker (e.g. AAPL, MSFT)
#   Security → company name
#   GICS Sector → sector (e.g. Technology, Healthcare)
#   GICS Sub-Industry → more specific industry
# ─────────────────────────────────────────────────────────

import pandas as pd
import os
from io import StringIO

import requests
from config import RAW_DIR

def fetch_sp500_list():
    print("Fetching S&P 500 constituent list from Wikipedia...")

    # Wikipedia maintains a live table of all S&P 500 companies.
    # pd.read_html() can directly scrape HTML tables — no API needed.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Wikipedia blocks some default Python URL fetches, so request the page
    # with a browser-like User-Agent and let pandas parse the HTML text.
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()

    # read_html returns a LIST of all tables on the page.
    # The S&P 500 table is always the first one (index 0).
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]

    # Rename columns to cleaner names we'll use throughout the project
    df = df.rename(columns={
        "Symbol":           "ticker",
        "Security":         "company_name",
        "GICS Sector":      "sector",
        "GICS Sub-Industry":"sub_industry",
        "Headquarters Location": "headquarters",
        "Date added":       "date_added",
        "CIK":              "cik",
        "Founded":          "founded"
    })

    # Some tickers have a dot (e.g. BRK.B) — yfinance uses a dash (BRK-B)
    # Replace dots with dashes to match yfinance format
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    # Keep only the columns we need
    df = df[["ticker", "company_name", "sector", "sub_industry"]]

    # Save to raw data folder
    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = f"{RAW_DIR}/sp500_companies.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} companies to {output_path}")
    print(f"\nSector breakdown:")
    print(df["sector"].value_counts().to_string())

    return df


if __name__ == "__main__":
    df = fetch_sp500_list()
    print("\nFirst 5 rows:")
    print(df.head())
