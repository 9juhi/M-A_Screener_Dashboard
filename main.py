# main.py
# ─────────────────────────────────────────────────────────
# Full pipeline orchestrator — all phases.
# ─────────────────────────────────────────────────────────

import os
from config import RAW_DIR, PROCESSED_DIR

from fetch_sp500 import fetch_sp500_list
from fetch_market_data import fetch_all_market_data
from fetch_financials import setup_simfin, merge_all_financials
from build_dataset import build_master_dataset
from fetch_supplementary_data import fetch_revenue_history, fetch_interest_expense
from compute_metrics import run_compute_metrics
from compute_sector_benchmarks import run_sector_benchmarks
from scoring_engine import run_scoring_engine


def run_phase1():
    print("=" * 50)
    print("PHASE 1: DATA FOUNDATION")
    print("=" * 50)
    os.makedirs(RAW_DIR, exist_ok=True)
    sp500 = fetch_sp500_list()
    fetch_all_market_data(sp500["ticker"].tolist())
    setup_simfin()
    fin_df = merge_all_financials()
    fin_df.to_csv(f"{RAW_DIR}/financials.csv", index=False)
    build_master_dataset()


def run_phase2():
    print("\n" + "=" * 50)
    print("PHASE 2: METRIC ENGINEERING")
    print("=" * 50)
    os.makedirs(RAW_DIR, exist_ok=True)
    setup_simfin()
    fetch_revenue_history()
    fetch_interest_expense()
    run_compute_metrics()
    run_sector_benchmarks()


def run_phase3():
    print("\n" + "=" * 50)
    print("PHASE 3: ACQUIRABILITY SCORING ENGINE")
    print("=" * 50)
    run_scoring_engine(preset="default")


if __name__ == "__main__":
    # Phases 1 and 2 already complete — run Phase 3 only
    run_phase3()