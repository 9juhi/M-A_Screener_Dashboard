# scoring_engine.py
# ─────────────────────────────────────────────────────────
# Phase 3: Acquirability Scoring Engine
#
# Takes the 6 percentile rank columns from Phase 2 and
# combines them into a single Acquirability Score (0–100).
#
# Formula:
#   Score = Σ (weight_i × rank_i)  for i in {6 signals}
#
# Because the rank columns are already 0–100 and the weights
# sum to 1.0, the final score is automatically on a 0–100 scale.
# A score of 80 means "this company's combined financial signals
# place it in the 80th percentile of acquisition attractiveness
# within its sector peer group."
#
# Two built-in weight presets:
#   "strategic"  — for a corporate acquirer who wants growth + fit
#   "pe"         — for a private equity firm who wants FCF + cheap entry
#
# Output: data/processed/acquirability_scores.parquet
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR


# ── Weight presets ─────────────────────────────────────────────────────────────
#
# Each preset maps a scoring signal to its weight (must sum to 1.0).
#
# STRATEGIC ACQUIRER logic:
#   Weights growth and margin heavily because a strategic buyer captures
#   revenue synergies — they're buying a growing business to integrate
#   into their own. Valuation matters but is secondary to strategic fit.
#
# PE FIRM logic:
#   A PE firm cannot capture revenue synergies — they make money by
#   buying cheap, cutting costs, servicing debt from FCF, and exiting.
#   So entry price (EV/EBITDA) and cash generation (FCF) dominate.
#   Leverage headroom (debt) is critical because they'll add more debt.
#
# DEFAULT is a balanced blend — suitable for a general-purpose screener
# that doesn't know the acquirer's specific mandate in advance.

WEIGHT_PRESETS = {
    "default": {
        "ev_to_ebitda_rank":      0.25,   # valuation attractiveness
        "revenue_cagr_5yr_rank":  0.20,   # growth profile
        "ebitda_margin_rank":     0.20,   # operational quality
        "debt_to_ebitda_rank":    0.15,   # leverage headroom
        "fcf_margin_rank":        0.10,   # cash generation
        "interest_coverage_rank": 0.10,   # financial safety
    },
    "strategic": {
        # Strategic buyers pay more attention to growth and margin
        # because they expect to extract value through integration.
        # Valuation still matters but they can justify paying more
        # for a genuinely growing, high-quality business.
        "ev_to_ebitda_rank":      0.20,
        "revenue_cagr_5yr_rank":  0.30,   # growth is most important
        "ebitda_margin_rank":     0.25,   # quality of earnings matters
        "debt_to_ebitda_rank":    0.10,
        "fcf_margin_rank":        0.10,
        "interest_coverage_rank": 0.05,
    },
    "pe": {
        # PE firms buy cheap and need FCF to service acquisition debt.
        # Growth is nice but not the primary thesis.
        # They CANNOT add more leverage to a target already at 4x+ debt.
        "ev_to_ebitda_rank":      0.35,   # entry price is everything in PE
        "revenue_cagr_5yr_rank":  0.10,
        "ebitda_margin_rank":     0.15,
        "debt_to_ebitda_rank":    0.20,   # must have room to lever up
        "fcf_margin_rank":        0.15,   # FCF services the deal debt
        "interest_coverage_rank": 0.05,
    },
}

# Signals where the underlying metric is unreliable for certain sectors.
# Companies in these sectors get a data quality warning flag on the dashboard.
# We don't exclude them from scoring — we just surface the caveat.
UNRELIABLE_SECTORS = {
    "Financials":  ["ev_to_ebitda_rank", "ebitda_margin_rank", "fcf_margin_rank"],
    "Real Estate": ["ev_to_ebitda_rank", "debt_to_ebitda_rank"],
}


# ── Core scoring function ──────────────────────────────────────────────────────

def compute_acquirability_score(
        df: pd.DataFrame,
        weights: dict
) -> pd.DataFrame:
    """
    Computes the weighted composite Acquirability Score for every company.

    The score is a weighted average of the 6 percentile rank columns.
    Because each rank is already 0–100 and the weights sum to 1.0,
    the result is a clean 0–100 score with no further scaling needed.

    We also compute a per-signal contribution (weight × rank) for each
    company. These contribution columns let the dashboard show a breakdown
    chart — "this company scores 72 overall; here's how each signal
    contributed to that number."
    """
    df = df.copy()
    score_accumulator = pd.Series(0.0, index=df.index)

    for rank_col, weight in weights.items():
        if rank_col not in df.columns:
            print(f"  WARNING: {rank_col} not found — skipping this signal")
            continue

        # Contribution of this single signal to the final score
        # e.g. if ev_to_ebitda_rank = 80 and weight = 0.25,
        # contribution = 20 out of a maximum possible 25
        contribution_col = rank_col.replace("_rank", "_contribution")
        df[contribution_col] = df[rank_col] * weight

        score_accumulator += df[contribution_col].fillna(0)

    df["acquirability_score"] = score_accumulator.round(2)

    return df


def add_score_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds human-readable metadata to each company's score row:

    1. Score tier label (S-tier, A-tier, etc.) — makes the dashboard
       immediately scannable without needing to read exact numbers.

    2. Sector rank — where does this company rank within its own sector?
       Useful for analysts who only care about targets in their sector.

    3. Global rank — where does this company rank across all 503?

    4. Data quality flag — warns the user when sector-specific
       metric reliability issues apply (e.g. Financials using EV/EBITDA).
    """
    df = df.copy()

    # ── Score tier labels ──────────────────────────────────────────────────
    # Tiers communicate quality at a glance without requiring the user
    # to interpret a raw number. A score of 78 is meaningless; "A-tier" is not.
    def assign_tier(score: float) -> str:
        if pd.isna(score):   return "Unrated"
        if score >= 80:      return "S-tier"    # top 20% — strong buy candidates
        if score >= 65:      return "A-tier"    # top 35% — worth investigating
        if score >= 50:      return "B-tier"    # average — not a priority
        if score >= 35:      return "C-tier"    # below average
        return                      "D-tier"    # bottom quartile — avoid

    df["score_tier"] = df["acquirability_score"].apply(assign_tier)

    # ── Global rank (1 = highest score across all 503 companies) ──────────
    df["global_rank"] = (
        df["acquirability_score"]
          .rank(ascending=False, method="min")
          .astype("Int64")
    )

    # ── Sector rank (1 = highest score within the company's sector) ───────
    df["sector_rank"] = (
        df.groupby("sector")["acquirability_score"]
          .rank(ascending=False, method="min")
          .astype("Int64")
    )

    # ── Sector size (useful context for sector_rank: "3 of 73" is better
    #    than "3 of 22") ────────────────────────────────────────────────────
    df["sector_size"] = df.groupby("sector")["sector"].transform("count")

    # ── Data quality warning for sectors with unreliable metrics ──────────
    def get_quality_warning(row):
        sector = row.get("sector", "")
        if sector in UNRELIABLE_SECTORS:
            bad_signals = UNRELIABLE_SECTORS[sector]
            readable = [s.replace("_rank", "").replace("_", "/") for s in bad_signals]
            return f"⚠ {sector}: {', '.join(readable)} metrics may be unreliable for this sector"
        return ""

    df["data_quality_warning"] = df.apply(get_quality_warning, axis=1)

    return df


# ── Output formatting ──────────────────────────────────────────────────────────

def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a clean, dashboard-ready leaderboard DataFrame.
    Contains only the columns needed for display — not the full 58-column
    enriched dataset. This is what the Streamlit dashboard will load as
    its primary data source.
    """
    display_cols = [
        # Identity
        "ticker", "company_name", "sector", "sub_industry",

        # Score
        "acquirability_score", "score_tier", "global_rank",
        "sector_rank", "sector_size",

        # Signal contributions (for the breakdown chart)
        "ev_to_ebitda_contribution",
        "revenue_cagr_5yr_contribution",
        "ebitda_margin_contribution",
        "debt_to_ebitda_contribution",
        "fcf_margin_contribution",
        "interest_coverage_contribution",

        # Raw metric values (for display alongside scores)
        "ev_to_ebitda", "revenue_cagr_pct", "ebitda_margin_pct",
        "debt_to_ebitda", "fcf_margin_pct", "interest_coverage",

        # Valuation context
        "ev_bn", "market_cap_bn", "revenue_bn", "price",

        # Audit
        "imputed_fields", "data_quality_warning",
    ]

    # Only keep columns that actually exist — guards against column name changes
    available = [c for c in display_cols if c in df.columns]
    leaderboard = df[available].sort_values(
        "acquirability_score", ascending=False
    ).reset_index(drop=True)

    return leaderboard


def print_score_summary(df: pd.DataFrame, weights: dict, preset_name: str):
    """Prints a readable summary to terminal for a sanity check."""
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


# ── Main runner ────────────────────────────────────────────────────────────────

def run_scoring_engine(preset: str = "default") -> pd.DataFrame:
    """
    Runs the full scoring pipeline for a given weight preset.
    Returns the complete leaderboard DataFrame.
    """
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

    # Load Phase 2 output
    scored_path = f"{PROCESSED_DIR}/scored_dataset.parquet"
    print(f"\nLoading {scored_path}...")
    df = pd.read_parquet(scored_path)
    print(f"  {len(df)} companies × {len(df.columns)} columns")

    # Compute composite score
    print("\nComputing acquirability scores...")
    df = compute_acquirability_score(df, weights)

    # Add ranks, tiers, and warnings
    df = add_score_metadata(df)

    # Build the leaderboard view
    leaderboard = build_leaderboard(df)

    # Save outputs
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Full dataset with scores (for Phase 4 comps engine)
    full_output_path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    df.to_parquet(full_output_path, index=False)

    # Leaderboard-only view (for the dashboard)
    leaderboard_path = f"{PROCESSED_DIR}/leaderboard.parquet"
    leaderboard.to_parquet(leaderboard_path, index=False)

    print(f"\nSaved full scores  → {full_output_path}")
    print(f"Saved leaderboard  → {leaderboard_path}")

    # Print summary to terminal
    print_score_summary(df, weights, preset)

    return leaderboard


if __name__ == "__main__":
    # Run with default weights
    leaderboard = run_scoring_engine(preset="default")

    print(f"\n\nPhase 3 complete.")
    print(f"Leaderboard saved to data/processed/leaderboard.parquet")
    print(f"Shape: {leaderboard.shape}")
    print(f"\nReady for Phase 4: Comparable Company Analysis Engine")