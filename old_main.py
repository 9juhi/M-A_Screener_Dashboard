# main.py
# ─────────────────────────────────────────────────────────
# Phase 1 orchestrator. Runs all steps in sequence.
# Run this file to execute the full pipeline:
#   python main.py
# ─────────────────────────────────────────────────────────

from fetch_sp500       import fetch_sp500_list
from fetch_market_data import fetch_all_market_data
from fetch_financials  import setup_simfin, merge_all_financials
from build_dataset     import build_master_dataset
import pandas as pd
from config import RAW_DIR
import os

def run_phase1():
    print("=" * 50)
    print("PHASE 1: DATA FOUNDATION")
    print("=" * 50)

    # Step 1: Get S&P 500 list
    print("\n[1/4] Fetching S&P 500 company list...")
    sp500 = fetch_sp500_list()

    # Step 2: Fetch market data
    print("\n[2/4] Fetching market data from yfinance...")
    market_df = fetch_all_market_data(sp500["ticker"].tolist())
    market_df.to_csv(f"{RAW_DIR}/market_data.csv", index=False)

    # Step 3: Fetch financials
    print("\n[3/4] Fetching financial statements from SimFin...")
    setup_simfin()
    fin_df = merge_all_financials()
    fin_df.to_csv(f"{RAW_DIR}/financials.csv", index=False)

    # Step 4: Build master dataset
    print("\n[4/4] Building master dataset...")
    master_df = build_master_dataset()

    print("\n" + "=" * 50)
    print("PHASE 1 COMPLETE")
    print(f"Master dataset: {len(master_df)} companies")
    print(f"Saved to: data/processed/master_dataset.parquet")
    print("=" * 50)
    print("\nReady for Phase 2: Metric Engineering")

if __name__ == "__main__":
    run_phase1()