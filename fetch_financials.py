import simfin as sf
from simfin.names import *
import pandas as pd
import os
from config import SIMFIN_API_KEY, RAW_DIR


def normalize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:

    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def setup_simfin():

    sf.set_api_key(SIMFIN_API_KEY)


    sf.set_data_dir(f"{RAW_DIR}/simfin_cache")
    print("SimFin configured.")


def fetch_income_statements() -> pd.DataFrame:


    print("Downloading income statements...")


    df = sf.load(dataset='income',
                 variant='annual',
                 market='us',
                 refresh_days=30)


    df = df.reset_index()


    df = df.rename(columns={
        TICKER:       "ticker",
        FISCAL_YEAR:  "fiscal_year",
        REVENUE:      "revenue",
        GROSS_PROFIT: "gross_profit",
        "Operating Income (Loss)": "ebit",
        DEPR_AMOR:    "depreciation_amortization",
        NET_INCOME:   "net_income",
    })
    df = normalize_ticker_column(df)

    if "ebit" not in df.columns:
        df["ebit"] = pd.NA

    if "depreciation_amortization" not in df.columns:
        df["depreciation_amortization"] = 0

    df["ebitda"] = (
        df["ebit"].fillna(0) +
        df["depreciation_amortization"].fillna(0)
    )


    df = (df.sort_values("fiscal_year", ascending=False)
            .groupby("ticker")
            .first()
            .reset_index())

    return df[["ticker", "fiscal_year", "revenue",
               "gross_profit", "ebitda", "ebit", "net_income"]]


def fetch_balance_sheets() -> pd.DataFrame:


    print("Downloading balance sheets...")

    df = sf.load(dataset='balance',
                 variant='annual',
                 market='us',
                 refresh_days=30)

    df = df.reset_index()

    df = df.rename(columns={
        TICKER:           "ticker",
        FISCAL_YEAR:      "fiscal_year",
        TOTAL_ASSETS:     "total_assets",
        TOTAL_EQUITY:     "total_equity",
        LT_DEBT:          "long_term_debt",
        ST_DEBT:          "short_term_debt",
        CASH_EQUIV_ST_INVEST: "cash_and_equivalents",
    })
    df = normalize_ticker_column(df)

    df = (df.sort_values("fiscal_year", ascending=False)
            .groupby("ticker")
            .first()
            .reset_index())


    df["total_debt"] = (df["long_term_debt"].fillna(0) +
                        df["short_term_debt"].fillna(0))

    return df[["ticker", "total_assets", "total_equity",
               "total_debt", "cash_and_equivalents"]]


def fetch_cashflows() -> pd.DataFrame:


    print("Downloading cash flow statements...")

    df = sf.load(dataset='cashflow',
                 variant='annual',
                 market='us',
                 refresh_days=30)

    df = df.reset_index()

    df = df.rename(columns={
        TICKER:               "ticker",
        FISCAL_YEAR:          "fiscal_year",
        NET_CASH_OPS:         "operating_cf",
        CAPEX:                "capex_raw",
    })
    df = normalize_ticker_column(df)

    df = (df.sort_values("fiscal_year", ascending=False)
            .groupby("ticker")
            .first()
            .reset_index())


    df["capex"] = df["capex_raw"].abs().fillna(0)


    df["free_cash_flow"] = df["operating_cf"] - df["capex"]

    return df[["ticker", "operating_cf", "capex", "free_cash_flow"]]


def merge_all_financials() -> pd.DataFrame:

    income  = fetch_income_statements()
    balance = fetch_balance_sheets()
    cashflow = fetch_cashflows()


    df = income.merge(balance,   on="ticker", how="left")
    df = df.merge(cashflow,      on="ticker", how="left")

    return df


if __name__ == "__main__":
    setup_simfin()

    df = merge_all_financials()

    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = f"{RAW_DIR}/financials.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} records to {output_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
