import simfin as sf
from simfin.names import *
import pandas as pd
import os
from config import SIMFIN_API_KEY, RAW_DIR


def setup_simfin():
    sf.set_api_key(SIMFIN_API_KEY)
    sf.set_data_dir(f"{RAW_DIR}/simfin_cache")


def normalize_ticker(df: pd.DataFrame) -> pd.DataFrame:

    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def fetch_revenue_history() -> pd.DataFrame:


    print("Fetching multi-year revenue history from SimFin cache...")

    df = sf.load(
        dataset='income',
        variant='annual',
        market='us',
        refresh_days=30
    )
    df = df.reset_index()


    df = df.rename(columns={
        TICKER:      "ticker",
        FISCAL_YEAR: "fiscal_year",
        REVENUE:     "revenue",
    })
    df = normalize_ticker(df)


    df = df[df["revenue"] > 0].dropna(subset=["revenue"])


    df = df.sort_values(["ticker", "fiscal_year"])

    output_path = f"{RAW_DIR}/revenue_history.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved revenue history → {output_path}")
    print(f"  Rows: {len(df)} | Companies: {df['ticker'].nunique()} | "
          f"Years per company: {df.groupby('ticker')['fiscal_year'].count().median():.0f} median")

    return df


def fetch_interest_expense() -> pd.DataFrame:


    print("Fetching interest expense from SimFin cache...")

    df = sf.load(
        dataset='income',
        variant='annual',
        market='us',
        refresh_days=30
    )
    df = df.reset_index()


    interest_col = None
    candidates = [
        "Interest Expense, Net",
        "Interest Expense",
        "Net Interest Expense",
    ]


    try:
        candidates.insert(0, INTEREST_EXP)
    except NameError:
        pass

    for candidate in candidates:
        if candidate in df.columns:
            interest_col = candidate
            print(f"  Found interest expense column: '{interest_col}'")
            break

    if interest_col is None:

        print("  WARNING: Could not find interest expense column.")
        print(f"  Available income statement columns: {df.columns.tolist()}")
        print("  Saving empty interest_expense.csv — interest coverage will be imputed.")
        empty = pd.DataFrame(columns=["ticker", "interest_expense"])
        empty.to_csv(f"{RAW_DIR}/interest_expense.csv", index=False)
        return empty

    df = df.rename(columns={
        TICKER:      "ticker",
        FISCAL_YEAR: "fiscal_year",
        interest_col: "interest_expense_raw",
    })
    df = normalize_ticker(df)


    df = (df.sort_values("fiscal_year", ascending=False)
            .groupby("ticker")
            .first()
            .reset_index())


    df["interest_expense"] = df["interest_expense_raw"].abs().fillna(0)

    output = df[["ticker", "interest_expense"]]
    output_path = f"{RAW_DIR}/interest_expense.csv"
    output.to_csv(output_path, index=False)

    non_zero = (output["interest_expense"] > 0).sum()
    print(f"Saved interest expense → {output_path}")
    print(f"  {non_zero} companies have non-zero interest expense")

    return output


if __name__ == "__main__":
    setup_simfin()
    fetch_revenue_history()
    fetch_interest_expense()
    print("\nSupplementary data fetch complete.")
