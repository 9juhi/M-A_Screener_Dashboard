# fetch_supplementary_data.py
# ─────────────────────────────────────────────────────────
# Fetches two things that fetch_financials.py didn't get:
#
#   1. MULTI-YEAR REVENUE HISTORY (all years, not just latest)
#      → needed to compute 5-year Revenue CAGR
#      → your existing fetch_financials.py does .groupby().first()
#        which keeps only the most recent year — we undo that here
#
#   2. INTEREST EXPENSE (most recent year)
#      → needed to compute Interest Coverage = EBIT / Interest Expense
#      → this line item was never fetched in Phase 1
#
# Both datasets come from SimFin which is already cached locally
# from Phase 1, so this runs in ~30 seconds.
#
# Outputs:
#   data/raw/revenue_history.csv     ← multiple rows per ticker
#   data/raw/interest_expense.csv    ← one row per ticker
# ─────────────────────────────────────────────────────────

import simfin as sf
from simfin.names import *
import pandas as pd
import os
from config import SIMFIN_API_KEY, RAW_DIR


def setup_simfin():
    sf.set_api_key(SIMFIN_API_KEY)
    sf.set_data_dir(f"{RAW_DIR}/simfin_cache")


def normalize_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Match the ticker format used throughout the project (dots → dashes)."""
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def fetch_revenue_history() -> pd.DataFrame:
    """
    Download ALL years of income statement data per company — not just latest.
    
    Why do we need all years?
    Revenue CAGR = (Revenue_today / Revenue_5yr_ago) ^ (1/5) - 1
    If we only have this year's revenue, we have nothing to divide it by.
    We need at least the revenue from 5 years ago to measure growth.
    
    We re-use the exact same SimFin download as fetch_financials.py —
    the data is cached locally so there's no additional API cost.
    The only difference is we skip the .groupby().first() step that
    was collapsing all years into one row.
    """
    print("Fetching multi-year revenue history from SimFin cache...")

    df = sf.load(
        dataset='income',
        variant='annual',
        market='us',
        refresh_days=30   # uses local cache if less than 30 days old
    )
    df = df.reset_index()

    # We only need the minimum columns for CAGR — no need to pull everything
    df = df.rename(columns={
        TICKER:      "ticker",
        FISCAL_YEAR: "fiscal_year",
        REVENUE:     "revenue",
    })
    df = normalize_ticker(df)

    # Keep only rows where revenue is a real positive number.
    # Negative or zero revenue (data errors, restatements) would produce
    # nonsensical CAGR values — safer to exclude them.
    df = df[df["revenue"] > 0].dropna(subset=["revenue"])

    # Sort oldest → newest within each company
    df = df.sort_values(["ticker", "fiscal_year"])

    output_path = f"{RAW_DIR}/revenue_history.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved revenue history → {output_path}")
    print(f"  Rows: {len(df)} | Companies: {df['ticker'].nunique()} | "
          f"Years per company: {df.groupby('ticker')['fiscal_year'].count().median():.0f} median")

    return df


def fetch_interest_expense() -> pd.DataFrame:
    """
    Download interest expense for the most recent fiscal year per company.
    
    Why interest expense?
    Interest Coverage Ratio = EBIT / Interest Expense
    This tells us how many times a company can cover its interest payments
    from its operating profit. < 1.5x is danger zone. > 5x is very safe.
    It's one of our 6 acquirability scoring signals.
    
    SimFin reports interest expense as a NEGATIVE number in the income
    statement (it's a cash outflow). We take the absolute value when
    computing interest coverage so the ratio is positive and readable.
    
    SimFin column name: INTEREST_EXP (the constant from simfin.names).
    If that constant doesn't exist in the installed version, we fall back
    to trying common string names that SimFin uses.
    """
    print("Fetching interest expense from SimFin cache...")

    df = sf.load(
        dataset='income',
        variant='annual',
        market='us',
        refresh_days=30
    )
    df = df.reset_index()

    # Try to find the interest expense column.
    # SimFin versions differ on the exact constant name —
    # we try several candidates in order.
    interest_col = None
    candidates = [
        "Interest Expense, Net",   # most common SimFin string
        "Interest Expense",
        "Net Interest Expense",
    ]

    # Also try simfin.names constants if they exist
    try:
        candidates.insert(0, INTEREST_EXP)
    except NameError:
        pass  # constant not in this version of simfin — that's fine

    for candidate in candidates:
        if candidate in df.columns:
            interest_col = candidate
            print(f"  Found interest expense column: '{interest_col}'")
            break

    if interest_col is None:
        # Print available columns to help debug
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

    # Keep only the most recent year per company
    df = (df.sort_values("fiscal_year", ascending=False)
            .groupby("ticker")
            .first()
            .reset_index())

    # SimFin reports this as negative (outflow). Store as positive for clarity.
    # Zero or NaN means no significant debt / not reported.
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