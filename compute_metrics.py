# compute_metrics.py
# ─────────────────────────────────────────────────────────
# Reads master_dataset.parquet (Phase 1 output) and adds
# the financial metrics needed for the scoring engine.
#
# What already exists in master_dataset (from build_dataset.py):
#   enterprise_value, ev_to_ebitda, ev_to_revenue, ev_to_ebit,
#   net_debt, net_debt_to_ebitda, pe_ratio, pb_ratio, fcf_yield
#
# What we ADD here:
#   gross_margin        → gross_profit / revenue
#   ebitda_margin       → ebitda / revenue
#   fcf_margin          → free_cash_flow / revenue
#   debt_to_ebitda      → total_debt / ebitda  (gross leverage, not net)
#   interest_coverage   → ebit / interest_expense
#   revenue_cagr_5yr    → compound annual revenue growth over 5 years
#
# Plus display-friendly columns (billions, percentages) for the dashboard.
#
# Output: data/processed/enriched_dataset.parquet
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from config import RAW_DIR, PROCESSED_DIR


# ── Safe arithmetic helpers ────────────────────────────────────────────────────

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Division that returns NaN instead of infinity or nonsense when:
      - denominator is zero
      - denominator is NaN
      - denominator is negative (a negative EBITDA makes ratios meaningless)
    
    Why reject negative denominators?
    A company with negative EBITDA and positive debt would produce a
    negative Debt/EBITDA — which looks like a healthy ratio when it
    actually signals the company is losing money. It's safer to say
    "this metric is not applicable" (NaN) than to produce a misleading number.
    """
    safe_denom = denominator.where(denominator > 0, other=np.nan)
    return numerator / safe_denom


def compute_cagr(start: float, end: float, years: int) -> float:
    """
    Compound Annual Growth Rate.
    
    Formula: (end / start) ^ (1/years) - 1
    
    Returns NaN if start or end is zero/negative (logarithm of negative
    numbers is undefined, and it would produce a meaningless growth rate).
    
    Example: revenue goes from $1B to $1.61B over 5 years
    CAGR = (1.61/1.0)^(1/5) - 1 = 0.10 = 10% per year
    """
    if (pd.isna(start) or pd.isna(end) or
            start <= 0 or end <= 0 or years <= 0):
        return np.nan
    return (end / start) ** (1 / years) - 1


# ── Revenue CAGR from multi-year history ──────────────────────────────────────

def compute_revenue_cagr(
        revenue_history: pd.DataFrame,
        n_years: int = 5
) -> pd.DataFrame:
    """
    For each company in the revenue history, find:
      - their most recent year's revenue  (end point)
      - their revenue n_years ago          (start point)
    Then compute CAGR between those two points.
    
    Why 5-year CAGR and not 1-year growth?
    One year of growth is too noisy — a single good or bad year can
    look like a trend. 5-year CAGR smooths out business cycles and
    gives a truer picture of the underlying growth trajectory.
    Consulting firms use 3–5 year CAGR as the standard.
    
    Fallback logic:
    If a company went public fewer than 5 years ago, we don't have
    5 years of data. Instead of dropping them, we compute CAGR over
    however many years ARE available, and record how many years we used.
    """
    print(f"Computing {n_years}-year Revenue CAGR for all companies...")
    results = []

    for ticker, group in revenue_history.groupby("ticker"):
        group = group.sort_values("fiscal_year")

        # Need at least 2 data points to compute any growth at all
        if len(group) < 2:
            results.append({
                "ticker": ticker,
                "revenue_cagr_5yr": np.nan,
                "cagr_years_used": np.nan,
            })
            continue

        # Most recent data point
        latest = group.iloc[-1]

        # Try to find data from exactly n_years ago.
        # If not found, use the earliest available year.
        target_year = latest["fiscal_year"] - n_years
        earlier_candidates = group[group["fiscal_year"] <= target_year]

        if len(earlier_candidates) == 0:
            # Less history than requested — use earliest available
            earlier = group.iloc[0]
        else:
            # Use the row closest to (but not after) the target year
            earlier = earlier_candidates.iloc[-1]

        actual_years = latest["fiscal_year"] - earlier["fiscal_year"]

        if actual_years == 0:
            results.append({
                "ticker": ticker,
                "revenue_cagr_5yr": np.nan,
                "cagr_years_used": 0,
            })
            continue

        cagr = compute_cagr(earlier["revenue"], latest["revenue"], actual_years)
        results.append({
            "ticker":           ticker,
            "revenue_cagr_5yr": cagr,
            "cagr_years_used":  actual_years,  # audit column — how many years used
        })

    cagr_df = pd.DataFrame(results)
    valid = cagr_df["revenue_cagr_5yr"].notna().sum()
    print(f"  Computed CAGR for {valid}/{len(cagr_df)} companies")
    return cagr_df


# ── Core metric computation ────────────────────────────────────────────────────

def add_missing_metrics(
        df: pd.DataFrame,
        interest_df: pd.DataFrame,
        cagr_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds the 6 metrics that weren't in master_dataset.parquet.
    Works on a copy so the original DataFrame is never mutated.
    """
    print("Adding missing financial metrics...")
    df = df.copy()

    # Merge in the supplementary data we just fetched
    df = df.merge(interest_df, on="ticker", how="left")
    df = df.merge(cagr_df[["ticker", "revenue_cagr_5yr", "cagr_years_used"]],
                  on="ticker", how="left")

    # ── Metric 1: Gross Margin ─────────────────────────────────────────────
    # Gross Profit / Revenue
    # Measures how much revenue remains after paying the direct costs of
    # making the product/service. A SaaS company might have 75% gross margin
    # (software is cheap to deliver). A manufacturer might have 25%
    # (raw materials, factory costs are expensive).
    df["gross_margin"] = safe_divide(df["gross_profit"], df["revenue"])

    # ── Metric 2: EBITDA Margin ────────────────────────────────────────────
    # EBITDA / Revenue
    # How much of every rupee/dollar of revenue becomes operational cash.
    # This is one of our 6 scoring signals — improving margins over time
    # signal operational leverage (growing without proportional cost growth).
    df["ebitda_margin"] = safe_divide(df["ebitda"], df["revenue"])

    # ── Metric 3: FCF Margin ───────────────────────────────────────────────
    # Free Cash Flow / Revenue
    # The "cash translation" ratio — how much revenue actually turns into
    # spendable cash after all investments. A company with 30% EBITDA margin
    # but 5% FCF margin is spending heavily on capex. A company where FCF
    # margin ≈ EBITDA margin is asset-light and cash-generative.
    df["fcf_margin"] = safe_divide(df["free_cash_flow"], df["revenue"])

    # ── Metric 4: Debt / EBITDA ────────────────────────────────────────────
    # Total Debt / EBITDA
    # The primary leverage metric in M&A. Answers: "how many years of
    # EBITDA would it take to pay off all the debt?"
    # < 2x: very comfortable | 2–4x: normal | > 5x: highly leveraged
    #
    # NOTE: We already have net_debt_to_ebitda in master_dataset.
    # This is the GROSS debt version (ignores cash). Both are useful:
    # - gross debt/EBITDA = total leverage burden
    # - net debt/EBITDA   = leverage after using cash to pay down debt
    df["debt_to_ebitda"] = safe_divide(df["total_debt"], df["ebitda"])

    # ── Metric 5: Interest Coverage Ratio ─────────────────────────────────
    # EBIT / Interest Expense
    # "How many times can operating profit cover the interest bill?"
    # < 1.5x: danger zone (can barely pay interest)
    # 2–5x:   normal operating range
    # > 5x:   very comfortable, low financial risk
    #
    # We clip the floor at -20 so extreme negative values (distressed
    # companies with huge losses) don't dominate the percentile distribution.
    df["interest_expense"] = df["interest_expense"].fillna(0)

    # Companies with zero interest expense (no debt) get a high coverage
    # ratio by definition — we assign 999 as a sentinel for "not applicable,
    # but effectively infinite coverage." The scoring engine will treat this
    # as the top percentile, which is correct.
    no_interest_mask = df["interest_expense"] == 0
    df["interest_coverage"] = safe_divide(
        df["ebit"],
        df["interest_expense"]
    ).clip(lower=-20)

    # Assign high sentinel value for zero-debt companies
    df.loc[no_interest_mask & df["ebit"].notna(), "interest_coverage"] = 999

    print(f"  Added: gross_margin, ebitda_margin, fcf_margin, "
          f"debt_to_ebitda, interest_coverage, revenue_cagr_5yr")

    return df


# ── Display-friendly columns ───────────────────────────────────────────────────

def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds human-readable versions of large numbers and ratio metrics.
    These aren't used in scoring — they're for the dashboard display layer.
    
    Without these, a dashboard user sees "4200000000" instead of "$4.2B".
    """
    # Scale large monetary values to billions
    for col, out_col in [
        ("market_cap",        "market_cap_bn"),
        ("enterprise_value",  "ev_bn"),
        ("revenue",           "revenue_bn"),
        ("ebitda",            "ebitda_bn"),
        ("free_cash_flow",    "fcf_bn"),
        ("total_debt",        "total_debt_bn"),
        ("net_debt",          "net_debt_bn"),
    ]:
        if col in df.columns:
            df[out_col] = df[col] / 1e9

    # Scale ratio metrics to percentage for display
    for col, out_col in [
        ("gross_margin",      "gross_margin_pct"),
        ("ebitda_margin",     "ebitda_margin_pct"),
        ("fcf_margin",        "fcf_margin_pct"),
        ("revenue_cagr_5yr",  "revenue_cagr_pct"),
    ]:
        if col in df.columns:
            df[out_col] = df[col] * 100

    return df


# ── Outlier handling ───────────────────────────────────────────────────────────

def winsorize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values at the 5th and 95th percentile.
    
    We don't DELETE outlier companies — that would silently remove
    real businesses from the dataset. We WINSORIZE them — cap values
    at the 5th/95th percentile boundary so one extreme outlier doesn't
    compress everyone else into a narrow band at the bottom of a ranking.
    
    Example: if one company has EV/EBITDA of 800x (data error or tiny
    EBITDA), without winsorizing, 499 normal companies would all cluster
    near 0 on a 0–800 scale, making the ranking useless.
    """
    print("Winsorizing extreme values...")
    df = df.copy()

    metrics_to_cap = [
        "ev_to_ebitda",
        "ev_to_revenue",
        "debt_to_ebitda",
        "net_debt_to_ebitda",
        "pe_ratio",
        "interest_coverage",
        "revenue_cagr_5yr",
        "ebitda_margin",
        "fcf_margin",
        "gross_margin",
    ]

    for metric in metrics_to_cap:
        if metric not in df.columns:
            continue

        lo = df[metric].quantile(0.05)
        hi = df[metric].quantile(0.95)
        n_clipped = ((df[metric] < lo) | (df[metric] > hi)).sum()

        if n_clipped > 0:
            df[metric] = df[metric].clip(lower=lo, upper=hi)
            print(f"  {metric:<28} clipped {n_clipped} values to [{lo:.2f}, {hi:.2f}]")

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values using sector median imputation.
    
    Why impute instead of drop?
    If we drop every company missing a single metric, we'd lose ~15% of
    our dataset. Worse, the missing companies are often smaller, less
    covered firms — so we'd be inadvertently biasing the dataset towards
    large, well-covered companies. That's selection bias and it distorts
    every sector comparison we make downstream.
    
    Why sector median and not global median?
    A missing FCF margin for a Utilities company should be filled with
    the Utilities sector median (~8%), not the S&P 500 median (~12%)
    which is skewed by high-margin tech companies.
    
    We record which fields were imputed in an audit column so the
    dashboard can flag those companies with a "data estimated" label.
    """
    print("Imputing missing values with sector medians...")
    df = df.copy()
    df["imputed_fields"] = ""

    # These are the metrics used in scoring — we must have values for all of them
    scoring_metrics = [
        "ev_to_ebitda",
        "revenue_cagr_5yr",
        "ebitda_margin",
        "debt_to_ebitda",
        "fcf_margin",
        "interest_coverage",
    ]

    for metric in scoring_metrics:
        if metric not in df.columns:
            continue

        missing_mask = df[metric].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            continue

        # Compute sector medians using only non-missing, non-imputed values
        sector_medians = df.groupby("sector")[metric].transform("median")
        global_median  = df[metric].median()

        # Fill: sector median first, fall back to global median if sector
        # median is itself NaN (sector with too few companies)
        fill_values = sector_medians.fillna(global_median)
        df.loc[missing_mask, metric] = fill_values[missing_mask]

        # Track which fields were imputed for the audit trail
        df.loc[missing_mask, "imputed_fields"] += f"{metric},"
        print(f"  {metric:<28} imputed {n_missing} values with sector median")

    df["data_complete"] = df["imputed_fields"].str.len() == 0
    complete = df["data_complete"].sum()
    print(f"\n  Fully complete (no imputation): {complete}/{len(df)} companies")

    return df


# ── Main runner ────────────────────────────────────────────────────────────────

def run_compute_metrics():
    print("=" * 50)
    print("PHASE 2 STEP 1: COMPUTE METRICS")
    print("=" * 50)

    # Load Phase 1 output
    master_path = f"{PROCESSED_DIR}/master_dataset.parquet"
    print(f"\nLoading {master_path}...")
    df = pd.read_parquet(master_path)
    print(f"  {len(df)} companies × {len(df.columns)} columns")

    # Load supplementary data (run fetch_supplementary_data.py first)
    interest_df = pd.read_csv(f"{RAW_DIR}/interest_expense.csv")
    revenue_history = pd.read_csv(f"{RAW_DIR}/revenue_history.csv")

    # Compute Revenue CAGR from multi-year history
    cagr_df = compute_revenue_cagr(revenue_history, n_years=5)

    # Add the 6 missing metrics
    df = add_missing_metrics(df, interest_df, cagr_df)

    # Handle data quality
    df = winsorize_metrics(df)
    df = impute_missing(df)

    # Add display-friendly columns for the dashboard
    df = add_display_columns(df)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = f"{PROCESSED_DIR}/enriched_dataset.parquet"
    df.to_parquet(output_path, index=False)

    print(f"\nSaved enriched dataset → {output_path}")
    print(f"Shape: {df.shape[0]} companies × {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    df = run_compute_metrics()

    print("\nSample output — key metrics for 10 companies:")
    preview = [
        "ticker", "company_name", "sector",
        "ev_to_ebitda", "ebitda_margin_pct",
        "revenue_cagr_pct", "debt_to_ebitda", "interest_coverage"
    ]
    available = [c for c in preview if c in df.columns]
    print(df[available].dropna(subset=["ev_to_ebitda"]).head(10).to_string(index=False))