# compute_sector_benchmarks.py
# ─────────────────────────────────────────────────────────
# Reads enriched_dataset.parquet and does two things:
#
#   1. SECTOR BENCHMARKS
#      For each sector × metric, compute the median, 25th and
#      75th percentile. These are the reference points the
#      dashboard will show ("your target trades at 8x EV/EBITDA
#      vs a sector median of 13x").
#
#   2. PERCENTILE RANK COLUMNS
#      For each of the 6 scoring signals, rank every company
#      within its sector peer group (0 = worst, 100 = best).
#      Direction-aware — for metrics where LOW is good (e.g.
#      EV/EBITDA, Debt/EBITDA), we invert the rank so that
#      a score of 100 always means "most attractive."
#
# Outputs:
#   data/processed/sector_benchmarks.parquet   ← sector summary table
#   data/processed/scored_dataset.parquet      ← enriched + rank columns
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR


# The 6 signals that feed the acquirability score.
# Key: metric column name
# Value: True = higher raw value is better; False = lower is better
SCORING_SIGNALS = {
    "ev_to_ebitda":     False,   # low multiple = cheap = attractive target
    "revenue_cagr_5yr": True,    # high growth = attractive
    "ebitda_margin":    True,    # high margin = efficient, attractive
    "debt_to_ebitda":   False,   # low leverage = room for deal financing
    "fcf_margin":       True,    # high FCF = cash-generative, attractive
    "interest_coverage": True,   # high coverage = financially safe
}


def compute_sector_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector × metric combination, compute descriptive statistics.
    
    Returns a tidy DataFrame: one row per (sector, metric).
    This table is what the comps dashboard will use to show
    "sector context" alongside any individual company's metrics.
    """
    print("Computing sector benchmarks...")
    records = []

    for sector, group in df.groupby("sector"):
        n_companies = len(group)

        for metric in SCORING_SIGNALS:
            if metric not in group.columns:
                continue

            values = group[metric].dropna()

            # Need at least 3 companies for meaningful statistics.
            # With only 1–2 data points, median/percentile are unstable.
            if len(values) < 3:
                continue

            records.append({
                "sector":          sector,
                "metric":          metric,
                "median":          values.median(),
                "mean":            values.mean(),
                "pct_25":          values.quantile(0.25),
                "pct_75":          values.quantile(0.75),
                "min":             values.min(),
                "max":             values.max(),
                "company_count":   n_companies,
                "data_count":      len(values),  # companies with actual (non-imputed) data
            })

    benchmarks = pd.DataFrame(records)
    print(f"  Benchmarks computed: "
          f"{benchmarks['sector'].nunique()} sectors × "
          f"{benchmarks['metric'].nunique()} metrics")

    return benchmarks


def add_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scoring signal, compute where each company sits within
    its sector peer group on a 0–100 scale.
    
    Why sector-relative and not global ranking?
    A Debt/EBITDA of 2x is excellent in Telecom (where 3–4x is normal)
    but only average in Technology (where 0–1x is common). Comparing
    across sectors with a single global ranking would systematically
    favour low-debt tech companies over all other sectors regardless of
    their actual financial health relative to their peers.
    
    Direction awareness:
    After ranking, we invert metrics where LOW is better.
    Example: a company in the 20th percentile for EV/EBITDA (very cheap
    relative to peers) is in the 80th percentile of ATTRACTIVENESS.
    After inversion, a score of 100 always = "most attractive in sector."
    
    The rank column naming convention:
    ev_to_ebitda → ev_to_ebitda_rank
    revenue_cagr_5yr → revenue_cagr_5yr_rank
    ...etc
    """
    print("Computing sector-relative percentile ranks for all 6 signals...")
    df = df.copy()

    for metric, higher_is_better in SCORING_SIGNALS.items():
        if metric not in df.columns:
            print(f"  SKIP: {metric} not found in dataset")
            continue

        rank_col = f"{metric}_rank"

        # rank(pct=True) gives a value from 0.0 to 1.0 within each group.
        # Multiply by 100 to get 0–100. na_option="keep" means companies
        # with NaN values get NaN rank (they won't affect others' ranks).
        df[rank_col] = (
            df.groupby("sector")[metric]
              .rank(pct=True, na_option="keep")
              * 100
        )

        # Invert for "lower is better" metrics.
        # If EV/EBITDA rank = 20 (cheap, bottom 20%), attractiveness = 80.
        if not higher_is_better:
            df[rank_col] = 100 - df[rank_col]

        valid = df[rank_col].notna().sum()
        print(f"  {metric:<28} rank computed for {valid} companies "
              f"({'higher raw = better' if higher_is_better else 'lower raw = better'})")

    return df


def print_benchmark_summary(benchmarks: pd.DataFrame):
    """Pretty-print a summary of sector benchmarks for the key metric."""
    print("\nSector benchmark summary — EV/EBITDA (sector medians):")
    ev_bench = (
        benchmarks[benchmarks["metric"] == "ev_to_ebitda"]
        [["sector", "pct_25", "median", "pct_75", "company_count"]]
        .sort_values("median", ascending=False)
    )
    print(ev_bench.to_string(index=False))

    print("\nSector benchmark summary — Revenue CAGR 5yr (sector medians):")
    cagr_bench = (
        benchmarks[benchmarks["metric"] == "revenue_cagr_5yr"]
        [["sector", "pct_25", "median", "pct_75", "company_count"]]
        .sort_values("median", ascending=False)
    )
    # Display as percentages
    for col in ["pct_25", "median", "pct_75"]:
        cagr_bench[col] = (cagr_bench[col] * 100).round(1).astype(str) + "%"
    print(cagr_bench.to_string(index=False))


def run_sector_benchmarks():
    print("\n" + "=" * 50)
    print("PHASE 2 STEP 2: SECTOR BENCHMARKS & PERCENTILE RANKS")
    print("=" * 50)

    enriched_path = f"{PROCESSED_DIR}/enriched_dataset.parquet"
    print(f"\nLoading {enriched_path}...")
    df = pd.read_parquet(enriched_path)
    print(f"  {len(df)} companies × {len(df.columns)} columns")

    # Step 1: Sector-level statistics
    benchmarks = compute_sector_benchmarks(df)

    # Step 2: Add percentile rank columns to the company-level dataset
    df_ranked = add_percentile_ranks(df)

    # Save both outputs
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    benchmarks_path = f"{PROCESSED_DIR}/sector_benchmarks.parquet"
    benchmarks.to_parquet(benchmarks_path, index=False)
    print(f"\nSaved sector benchmarks → {benchmarks_path}")

    scored_path = f"{PROCESSED_DIR}/scored_dataset.parquet"
    df_ranked.to_parquet(scored_path, index=False)
    print(f"Saved scored dataset   → {scored_path}")
    print(f"Shape: {df_ranked.shape[0]} companies × {df_ranked.shape[1]} columns")

    print_benchmark_summary(benchmarks)

    return df_ranked, benchmarks


if __name__ == "__main__":
    df_ranked, benchmarks = run_sector_benchmarks()

    print("\nRank columns added to dataset:")
    rank_cols = [c for c in df_ranked.columns if c.endswith("_rank")]
    print("  " + ", ".join(rank_cols))

    print("\nSample — Apple (AAPL) scores:")
    aapl = df_ranked[df_ranked["ticker"] == "AAPL"]
    if len(aapl) > 0:
        sample_cols = ["ticker", "sector"] + rank_cols
        print(aapl[sample_cols].to_string(index=False))