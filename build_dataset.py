import pandas as pd
import numpy as np
import os
from config import RAW_DIR, PROCESSED_DIR


SP500_PATH = f"{RAW_DIR}/sp500_companies.csv"
MARKET_DATA_PATH = f"{RAW_DIR}/market_data.csv"
FINANCIALS_PATH = f"{RAW_DIR}/financials.csv"
OUTPUT_PATH = f"{PROCESSED_DIR}/master_dataset.parquet"

MARKET_COLUMNS = [
    "ticker",
    "price",
    "market_cap",
    "shares_outstanding",
    "52w_high",
    "52w_low",
]

FINANCIAL_COLUMNS = [
    "ticker",
    "fiscal_year",
    "revenue",
    "gross_profit",
    "ebitda",
    "ebit",
    "net_income",
    "total_assets",
    "total_equity",
    "total_debt",
    "cash_and_equivalents",
    "operating_cf",
    "capex",
    "free_cash_flow",
]


def safe_divide(numerator, denominator):

    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df


def load_market_data() -> pd.DataFrame:
    market = pd.read_csv(MARKET_DATA_PATH)
    market = ensure_columns(market, MARKET_COLUMNS)


    market = market[MARKET_COLUMNS]
    market = market[market["price"].notna()]
    market = market.drop_duplicates(subset=["ticker"], keep="last")

    return market


def load_financials() -> pd.DataFrame:
    financials = pd.read_csv(FINANCIALS_PATH)
    financials = ensure_columns(financials, FINANCIAL_COLUMNS)
    financials = financials[FINANCIAL_COLUMNS]
    financials = financials.drop_duplicates(subset=["ticker"], keep="last")

    return financials


def add_calculated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["cash_and_equivalents"] = df["cash_and_equivalents"].fillna(0)
    df["total_debt"] = df["total_debt"].fillna(0)
    df["net_debt"] = df["total_debt"] - df["cash_and_equivalents"]

    df["enterprise_value"] = (
        df["market_cap"] + df["total_debt"] - df["cash_and_equivalents"]
    )

    df["pe_ratio"] = safe_divide(df["market_cap"], df["net_income"])
    df["pb_ratio"] = safe_divide(df["market_cap"], df["total_equity"])
    df["ev_to_revenue"] = safe_divide(df["enterprise_value"], df["revenue"])
    df["ev_to_ebitda"] = safe_divide(df["enterprise_value"], df["ebitda"])
    df["ev_to_ebit"] = safe_divide(df["enterprise_value"], df["ebit"])
    df["fcf_yield"] = safe_divide(df["free_cash_flow"], df["market_cap"])
    df["net_debt_to_ebitda"] = safe_divide(df["net_debt"], df["ebitda"])

    return df


def build_master_dataset():
    print("Loading raw files...")

    sp500      = pd.read_csv(SP500_PATH)
    market     = load_market_data()
    financials = load_financials()

    print(f"  S&P 500 list:  {len(sp500)} companies")
    print(f"  Market data:   {len(market)} companies")
    print(f"  Financials:    {len(financials)} companies")


    df = sp500.merge(market,     on="ticker", how="left")
    df = df.merge(financials,    on="ticker", how="left")
    df = add_calculated_metrics(df)

    print(f"\nMerged dataset: {len(df)} rows × {len(df.columns)} columns")


    print("\nData quality report:")

    key_cols = ["price", "market_cap", "total_debt", "cash_and_equivalents",
                "revenue", "ebitda", "ebit", "net_income", "free_cash_flow",
                "enterprise_value", "ev_to_ebitda"]

    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct     = missing / len(df) * 100
            print(f"  {col:<28} missing: {missing:>3} ({pct:.1f}%)")


    df["data_complete"] = (
        df["market_cap"].notna() &
        df["revenue"].notna() &
        df["ebitda"].notna()
    )

    complete = df["data_complete"].sum()
    print(f"\nCompanies with complete core data: {complete}/{len(df)}")


    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved master dataset to {OUTPUT_PATH}")
    print(f"Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    df = build_master_dataset()

    print("\nSample — first 5 rows (key columns only):")
    preview_cols = ["ticker", "company_name", "sector",
                    "market_cap", "revenue", "total_debt", "free_cash_flow"]
    print(df[preview_cols].head(10).to_string(index=False))
