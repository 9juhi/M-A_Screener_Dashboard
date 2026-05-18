import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR


SCORING_SIGNALS = {
    "ev_to_ebitda":     False,
    "revenue_cagr_5yr": True,
    "ebitda_margin":    True,
    "debt_to_ebitda":   False,
    "fcf_margin":       True,
    "interest_coverage": True,
}


def compute_sector_benchmarks(df: pd.DataFrame) -> pd.DataFrame:


    print("Computing sector benchmarks...")
    records = []

    for sector, group in df.groupby("sector"):
        n_companies = len(group)

        for metric in SCORING_SIGNALS:
            if metric not in group.columns:
                continue

            values = group[metric].dropna()


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
                "data_count":      len(values),
            })

    benchmarks = pd.DataFrame(records)
    print(f"  Benchmarks computed: "
          f"{benchmarks['sector'].nunique()} sectors × "
          f"{benchmarks['metric'].nunique()} metrics")

    return benchmarks


def add_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:


    print("Computing sector-relative percentile ranks for all 6 signals...")
    df = df.copy()

    for metric, higher_is_better in SCORING_SIGNALS.items():
        if metric not in df.columns:
            print(f"  SKIP: {metric} not found in dataset")
            continue

        rank_col = f"{metric}_rank"


        df[rank_col] = (
            df.groupby("sector")[metric]
              .rank(pct=True, na_option="keep")
              * 100
        )


        if not higher_is_better:
            df[rank_col] = 100 - df[rank_col]

        valid = df[rank_col].notna().sum()
        print(f"  {metric:<28} rank computed for {valid} companies "
              f"({'higher raw = better' if higher_is_better else 'lower raw = better'})")

    return df


def print_benchmark_summary(benchmarks: pd.DataFrame):

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


    benchmarks = compute_sector_benchmarks(df)


    df_ranked = add_percentile_ranks(df)


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
