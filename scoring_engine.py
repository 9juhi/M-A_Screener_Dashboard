import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR


WEIGHT_PRESETS = {
    "default": {
        "ev_to_ebitda_rank":      0.25,
        "revenue_cagr_5yr_rank":  0.20,
        "ebitda_margin_rank":     0.20,
        "debt_to_ebitda_rank":    0.15,
        "fcf_margin_rank":        0.10,
        "interest_coverage_rank": 0.10,
    },
    "strategic": {


        "ev_to_ebitda_rank":      0.20,
        "revenue_cagr_5yr_rank":  0.30,
        "ebitda_margin_rank":     0.25,
        "debt_to_ebitda_rank":    0.10,
        "fcf_margin_rank":        0.10,
        "interest_coverage_rank": 0.05,
    },
    "pe": {


        "ev_to_ebitda_rank":      0.35,
        "revenue_cagr_5yr_rank":  0.10,
        "ebitda_margin_rank":     0.15,
        "debt_to_ebitda_rank":    0.20,
        "fcf_margin_rank":        0.15,
        "interest_coverage_rank": 0.05,
    },
}


UNRELIABLE_SECTORS = {
    "Financials":  ["ev_to_ebitda_rank", "ebitda_margin_rank", "fcf_margin_rank"],
    "Real Estate": ["ev_to_ebitda_rank", "debt_to_ebitda_rank"],
}


def compute_acquirability_score(
        df: pd.DataFrame,
        weights: dict
) -> pd.DataFrame:


    df = df.copy()
    score_accumulator = pd.Series(0.0, index=df.index)

    for rank_col, weight in weights.items():
        if rank_col not in df.columns:
            print(f"  WARNING: {rank_col} not found — skipping this signal")
            continue


        contribution_col = rank_col.replace("_rank", "_contribution")
        df[contribution_col] = df[rank_col] * weight

        score_accumulator += df[contribution_col].fillna(0)

    df["acquirability_score"] = score_accumulator.round(2)

    return df


def add_score_metadata(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()


    def assign_tier(score: float) -> str:
        if pd.isna(score):   return "Unrated"
        if score >= 80:      return "S-tier"
        if score >= 65:      return "A-tier"
        if score >= 50:      return "B-tier"
        if score >= 35:      return "C-tier"
        return                      "D-tier"

    df["score_tier"] = df["acquirability_score"].apply(assign_tier)


    df["global_rank"] = (
        df["acquirability_score"]
          .rank(ascending=False, method="min")
          .astype("Int64")
    )


    df["sector_rank"] = (
        df.groupby("sector")["acquirability_score"]
          .rank(ascending=False, method="min")
          .astype("Int64")
    )


    df["sector_size"] = df.groupby("sector")["sector"].transform("count")


    def get_quality_warning(row):
        sector = row.get("sector", "")
        if sector in UNRELIABLE_SECTORS:
            bad_signals = UNRELIABLE_SECTORS[sector]
            readable = [s.replace("_rank", "").replace("_", "/") for s in bad_signals]
            return f"⚠ {sector}: {', '.join(readable)} metrics may be unreliable for this sector"
        return ""

    df["data_quality_warning"] = df.apply(get_quality_warning, axis=1)

    return df


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:


    display_cols = [

        "ticker", "company_name", "sector", "sub_industry",


        "acquirability_score", "score_tier", "global_rank",
        "sector_rank", "sector_size",


        "ev_to_ebitda_contribution",
        "revenue_cagr_5yr_contribution",
        "ebitda_margin_contribution",
        "debt_to_ebitda_contribution",
        "fcf_margin_contribution",
        "interest_coverage_contribution",


        "ev_to_ebitda", "revenue_cagr_pct", "ebitda_margin_pct",
        "debt_to_ebitda", "fcf_margin_pct", "interest_coverage",


        "ev_bn", "market_cap_bn", "revenue_bn", "price",


        "imputed_fields", "data_quality_warning",
    ]


    available = [c for c in display_cols if c in df.columns]
    leaderboard = df[available].sort_values(
        "acquirability_score", ascending=False
    ).reset_index(drop=True)

    return leaderboard


def print_score_summary(df: pd.DataFrame, weights: dict, preset_name: str):

    print(f"\n{'='*55}")
    print(f"ACQUIRABILITY SCORE SUMMARY — preset: '{preset_name}'")
    print(f"{'='*55}")

    print("\nWeight breakdown used:")
    for signal, weight in weights.items():
        readable = signal.replace("_rank", "").replace("_", " ")
        bar = "█" * int(weight * 40)
        print(f"  {readable:<25} {bar:<42} {weight*100:.0f}%")

    print(f"\nScore distribution across {len(df)} companies:")
    print(f"  Mean score:    {df['acquirability_score'].mean():.1f}")
    print(f"  Median score:  {df['acquirability_score'].median():.1f}")
    print(f"  Std deviation: {df['acquirability_score'].std():.1f}")
    print(f"  Min / Max:     {df['acquirability_score'].min():.1f} / "
          f"{df['acquirability_score'].max():.1f}")

    print(f"\nTier breakdown:")
    tier_counts = df["score_tier"].value_counts()
    for tier in ["S-tier", "A-tier", "B-tier", "C-tier", "D-tier", "Unrated"]:
        count = tier_counts.get(tier, 0)
        pct   = count / len(df) * 100
        print(f"  {tier:<10} {count:>3} companies  ({pct:.1f}%)")

    print(f"\nTop 20 acquisition targets (global ranking):")
    top20_cols = [
        "global_rank", "ticker", "company_name", "sector",
        "acquirability_score", "score_tier",
        "ev_to_ebitda", "revenue_cagr_pct", "ebitda_margin_pct"
    ]
    available = [c for c in top20_cols if c in df.columns]
    top20 = df.sort_values("acquirability_score", ascending=False).head(20)
    print(top20[available].to_string(index=False))

    print(f"\nBottom 5 (sanity check — should be distressed/overvalued companies):")
    bottom5_cols = ["ticker", "company_name", "sector",
                    "acquirability_score", "score_tier"]
    bottom5 = df.sort_values("acquirability_score").head(5)
    print(bottom5[bottom5_cols].to_string(index=False))

    print(f"\nTop 3 per sector:")
    for sector, group in df.groupby("sector"):
        top3 = group.nlargest(3, "acquirability_score")[
            ["ticker", "company_name", "acquirability_score", "score_tier"]
        ]
        print(f"\n  {sector}:")
        for _, row in top3.iterrows():
            print(f"    {row['ticker']:<6} {row['company_name']:<35} "
                  f"{row['acquirability_score']:.1f}  {row['score_tier']}")


def run_scoring_engine(preset: str = "default") -> pd.DataFrame:


    print("=" * 50)
    print(f"PHASE 3: ACQUIRABILITY SCORING ENGINE")
    print(f"Weight preset: '{preset}'")
    print("=" * 50)

    if preset not in WEIGHT_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Choose from: {list(WEIGHT_PRESETS.keys())}"
        )

    weights = WEIGHT_PRESETS[preset]


    scored_path = f"{PROCESSED_DIR}/scored_dataset.parquet"
    print(f"\nLoading {scored_path}...")
    df = pd.read_parquet(scored_path)
    print(f"  {len(df)} companies × {len(df.columns)} columns")


    print("\nComputing acquirability scores...")
    df = compute_acquirability_score(df, weights)


    df = add_score_metadata(df)


    leaderboard = build_leaderboard(df)


    os.makedirs(PROCESSED_DIR, exist_ok=True)


    full_output_path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    df.to_parquet(full_output_path, index=False)


    leaderboard_path = f"{PROCESSED_DIR}/leaderboard.parquet"
    leaderboard.to_parquet(leaderboard_path, index=False)

    print(f"\nSaved full scores  → {full_output_path}")
    print(f"Saved leaderboard  → {leaderboard_path}")


    print_score_summary(df, weights, preset)

    return leaderboard


if __name__ == "__main__":

    leaderboard = run_scoring_engine(preset="default")

    print(f"\n\nPhase 3 complete.")
    print(f"Leaderboard saved to data/processed/leaderboard.parquet")
    print(f"Shape: {leaderboard.shape}")
    print(f"\nReady for Phase 4: Comparable Company Analysis Engine")
