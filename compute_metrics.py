import pandas as pd
import numpy as np
import os
from config import RAW_DIR, PROCESSED_DIR


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:


    safe_denom = denominator.where(denominator > 0, other=np.nan)
    return numerator / safe_denom


def compute_cagr(start: float, end: float, years: int) -> float:


    if (pd.isna(start) or pd.isna(end) or
            start <= 0 or end <= 0 or years <= 0):
        return np.nan
    return (end / start) ** (1 / years) - 1


def compute_revenue_cagr(
        revenue_history: pd.DataFrame,
        n_years: int = 5
) -> pd.DataFrame:


    print(f"Computing {n_years}-year Revenue CAGR for all companies...")
    results = []

    for ticker, group in revenue_history.groupby("ticker"):
        group = group.sort_values("fiscal_year")


        if len(group) < 2:
            results.append({
                "ticker": ticker,
                "revenue_cagr_5yr": np.nan,
                "cagr_years_used": np.nan,
            })
            continue


        latest = group.iloc[-1]


        target_year = latest["fiscal_year"] - n_years
        earlier_candidates = group[group["fiscal_year"] <= target_year]

        if len(earlier_candidates) == 0:

            earlier = group.iloc[0]
        else:

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
            "cagr_years_used":  actual_years,
        })

    cagr_df = pd.DataFrame(results)
    valid = cagr_df["revenue_cagr_5yr"].notna().sum()
    print(f"  Computed CAGR for {valid}/{len(cagr_df)} companies")
    return cagr_df


def add_missing_metrics(
        df: pd.DataFrame,
        interest_df: pd.DataFrame,
        cagr_df: pd.DataFrame
) -> pd.DataFrame:


    print("Adding missing financial metrics...")
    df = df.copy()


    df = df.merge(interest_df, on="ticker", how="left")
    df = df.merge(cagr_df[["ticker", "revenue_cagr_5yr", "cagr_years_used"]],
                  on="ticker", how="left")


    df["gross_margin"] = safe_divide(df["gross_profit"], df["revenue"])


    df["ebitda_margin"] = safe_divide(df["ebitda"], df["revenue"])


    df["fcf_margin"] = safe_divide(df["free_cash_flow"], df["revenue"])


    df["debt_to_ebitda"] = safe_divide(df["total_debt"], df["ebitda"])


    df["interest_expense"] = df["interest_expense"].fillna(0)


    no_interest_mask = df["interest_expense"] == 0
    df["interest_coverage"] = safe_divide(
        df["ebit"],
        df["interest_expense"]
    ).clip(lower=-20)


    df.loc[no_interest_mask & df["ebit"].notna(), "interest_coverage"] = 999

    print(f"  Added: gross_margin, ebitda_margin, fcf_margin, "
          f"debt_to_ebitda, interest_coverage, revenue_cagr_5yr")

    return df


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:


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


    for col, out_col in [
        ("gross_margin",      "gross_margin_pct"),
        ("ebitda_margin",     "ebitda_margin_pct"),
        ("fcf_margin",        "fcf_margin_pct"),
        ("revenue_cagr_5yr",  "revenue_cagr_pct"),
    ]:
        if col in df.columns:
            df[out_col] = df[col] * 100

    return df


def winsorize_metrics(df: pd.DataFrame) -> pd.DataFrame:


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


    print("Imputing missing values with sector medians...")
    df = df.copy()
    df["imputed_fields"] = ""


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


        sector_medians = df.groupby("sector")[metric].transform("median")
        global_median  = df[metric].median()


        fill_values = sector_medians.fillna(global_median)
        df.loc[missing_mask, metric] = fill_values[missing_mask]


        df.loc[missing_mask, "imputed_fields"] += f"{metric},"
        print(f"  {metric:<28} imputed {n_missing} values with sector median")

    df["data_complete"] = df["imputed_fields"].str.len() == 0
    complete = df["data_complete"].sum()
    print(f"\n  Fully complete (no imputation): {complete}/{len(df)} companies")

    return df


def run_compute_metrics():
    print("=" * 50)
    print("PHASE 2 STEP 1: COMPUTE METRICS")
    print("=" * 50)


    master_path = f"{PROCESSED_DIR}/master_dataset.parquet"
    print(f"\nLoading {master_path}...")
    df = pd.read_parquet(master_path)
    print(f"  {len(df)} companies × {len(df.columns)} columns")


    interest_df = pd.read_csv(f"{RAW_DIR}/interest_expense.csv")
    revenue_history = pd.read_csv(f"{RAW_DIR}/revenue_history.csv")


    cagr_df = compute_revenue_cagr(revenue_history, n_years=5)


    df = add_missing_metrics(df, interest_df, cagr_df)


    df = winsorize_metrics(df)
    df = impute_missing(df)


    df = add_display_columns(df)


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
