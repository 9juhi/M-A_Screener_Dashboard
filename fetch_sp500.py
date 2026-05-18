import pandas as pd
import os
from io import StringIO

import requests
from config import RAW_DIR

def fetch_sp500_list():
    print("Fetching S&P 500 constituent list from Wikipedia...")


    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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


    tables = pd.read_html(StringIO(response.text))
    df = tables[0]


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


    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)


    df = df[["ticker", "company_name", "sector", "sub_industry"]]


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
