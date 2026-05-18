# comps_engine.py
# ─────────────────────────────────────────────────────────
# Phase 4: Comparable Company Analysis (Comps) Engine
#
# For any given target company (by ticker), this module:
#   1. Selects a peer group of comparable companies
#   2. Computes valuation multiples across the peer group
#   3. Applies peer median multiples to the target's
#      own financials to derive an implied valuation
#   4. Runs a scenario analysis (what-if EBITDA grows X%)
#
# Design philosophy:
#   This engine is written as a collection of pure functions
#   that each take a DataFrame and return a DataFrame.
#   That makes it easy to call from the Streamlit dashboard
#   interactively — the dashboard just calls run_comps(ticker)
#   and gets back everything it needs to render.
#
# Output:
#   data/processed/comps_results/  ← one JSON per company (demo)
#   The engine is also importable for use in the dashboard.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import json
from config import PROCESSED_DIR


# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum number of peers we want in a comps table.
# If we can't find this many with tight filters, we relax them.
MIN_PEERS = 5

# Maximum number of peers to include in the comps table.
# More than 12 peers starts to look like a full sector report
# rather than a focused comparable analysis.
MAX_PEERS = 12

# EV size band for peer selection.
# A peer's EV must be between (target_EV × LOW) and (target_EV × HIGH).
# 0.3x to 3x means a $10B EV company will consider peers from $3B to $30B.
# This is the standard "within one order of magnitude" rule of thumb.
SIZE_BAND_LOW  = 0.3
SIZE_BAND_HIGH = 3.0

# Relaxed size band used when MIN_PEERS can't be found with the tight band.
# Opens up to roughly 10x size difference.
SIZE_BAND_LOW_RELAXED  = 0.1
SIZE_BAND_HIGH_RELAXED = 10.0

# EBITDA growth scenarios for the what-if analysis.
# These represent post-acquisition improvement assumptions.
EBITDA_GROWTH_SCENARIOS = [-0.10, 0.00, 0.10, 0.20, 0.30]

# revised metric: Growth rate filter
# A peer's 5-year revenue CAGR must be within this many percentage points
# of the target's CAGR to be considered "comparable growth stage."
# Example: target CAGR = 14%, window = 15pp → peers must be between -1% and 29%
# This directly excludes hyper-growth companies (Datadog at 39%) from
# being compared to mature growers (Adobe at 14%).
GROWTH_CAGR_WINDOW_PP = 15.0   # percentage points, not decimal

# revised metric: EBITDA margin filter
# A peer's EBITDA margin must be within this many percentage points
# of the target's margin to be considered "same profitability stage."
# Example: target margin = 35%, window = 25pp → peers must be between 10% and 60%
# This excludes early-stage companies burning cash from mature company comps.
MARGIN_WINDOW_PP = 25.0        # percentage points

# revised metric: IQR multiplier for outlier removal on multiples.
# Standard box-plot definition: outlier > Q3 + 1.5×IQR
# We use 1.5 which is the most common statistical convention.
IQR_MULTIPLIER = 1.5


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the fully enriched and scored dataset from Phase 2/3.
    This is our single source of truth for all company metrics.
    """
    path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    df = pd.read_parquet(path)
    return df


# ── NEW: Multiple outlier removal ─────────────────────────────────────────────

def remove_multiple_outliers(
        peers: pd.DataFrame,
        multiple_col: str = "ev_to_ebitda",
        iqr_multiplier: float = IQR_MULTIPLIER
) -> tuple[pd.DataFrame, list]:
    """
    Removes peers whose EV/EBITDA is a statistical outlier using the IQR method.

    How it works:
      Q1 = 25th percentile of peer multiples
      Q3 = 75th percentile of peer multiples
      IQR = Q3 - Q1  (the "middle 50%" spread)
      Upper fence = Q3 + (iqr_multiplier × IQR)
      Any peer above the upper fence is an outlier and gets removed.

    Why only an upper fence and not a lower fence?
    In M&A valuation, a very LOW EV/EBITDA typically means the company
    is genuinely cheap — that's interesting information we want to keep.
    A very HIGH EV/EBITDA usually means the market is pricing in exceptional
    future growth that our target doesn't share — that's the distorting
    effect we want to remove.

    Returns the cleaned peer DataFrame and a list of tickers that were removed.
    """
    if multiple_col not in peers.columns or len(peers) < 4:
        return peers, []

    values = peers[multiple_col].dropna()
    if len(values) < 4:
        return peers, []

    q1  = values.quantile(0.25)
    q3  = values.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + iqr_multiplier * iqr

    outlier_mask = peers[multiple_col] > upper_fence
    removed_tickers = peers[outlier_mask]["ticker"].tolist()
    cleaned_peers   = peers[~outlier_mask]

    return cleaned_peers, removed_tickers

# ── Step 1: Peer Selection ─────────────────────────────────────────────────────

# ── Step 1: Peer Selection (updated) ──────────────────────────────────────────

def select_peers(
        df: pd.DataFrame,
        target_ticker: str,
        verbose: bool = True
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Finds the most comparable companies using a multi-dimensional filter.

    Dimension 1 — Sector/sub-industry: same GICS classification
    Dimension 2 — Size: EV within 0.3x–3x of target EV
    Dimension 3 — Growth stage: revenue CAGR within ±15 percentage points
    Dimension 4 — Profitability stage: EBITDA margin within ±25 percentage points

    The growth and margin filters are the key additions that fix the
    Adobe/Datadog problem. They encode the analyst's judgment that
    "comparable" means similar business trajectory, not just same label.

    Returns: (target_row, peers_df, selection_metadata_dict)
    The metadata dict records which filters were applied and how many
    peers each step found — useful for the dashboard's transparency layer.
    """
    target_mask = df["ticker"] == target_ticker
    if not target_mask.any():
        raise ValueError(f"Ticker '{target_ticker}' not found in dataset.")

    target = df[target_mask].iloc[0]
    target_ev           = target.get("enterprise_value", np.nan)
    target_sector       = target.get("sector", "")
    target_sub_industry = target.get("sub_industry", "")
    target_cagr         = target.get("revenue_cagr_5yr", np.nan)
    target_margin       = target.get("ebitda_margin", np.nan)

    if verbose:
        print(f"\nFinding peers for: {target['company_name']} ({target_ticker})")
        print(f"  Sector:        {target_sector}")
        print(f"  Sub-industry:  {target_sub_industry}")
        print(f"  EV:            ${target_ev/1e9:.2f}B" if pd.notna(target_ev) else "  EV: N/A")
        print(f"  Revenue CAGR:  {target_cagr*100:.1f}%" if pd.notna(target_cagr) else "  Revenue CAGR: N/A")
        print(f"  EBITDA margin: {target_margin*100:.1f}%" if pd.notna(target_margin) else "  EBITDA margin: N/A")

    # Remove the target itself from the universe we're searching
    universe = df[df["ticker"] != target_ticker].copy()
    metadata = {
        "target_ticker":       target_ticker,
        "filters_applied":     [],
        "peers_after_sector":  0,
        "peers_after_size":    0,
        "peers_after_growth":  0,
        "peers_after_margin":  0,
        "peers_after_outlier": 0,
        "match_level":         "",
        "outliers_removed":    [],
    }

    # ── Filter 1: Sector ───────────────────────────────────────────────────
    # We always start with sector matching. Cross-sector comparisons are
    # almost never valid in professional M&A analysis.
    sector_universe = universe[universe["sector"] == target_sector]
    metadata["peers_after_sector"] = len(sector_universe)

    # ── Filter 2: Size (EV band) ───────────────────────────────────────────
    def apply_size_filter(pool, low, high):
        if pd.notna(target_ev) and target_ev > 0:
            return pool[
                pool["enterprise_value"].between(target_ev * low, target_ev * high)
            ]
        return pool  # no EV data → skip size filter

    # Try tight size band first
    after_size_tight = apply_size_filter(sector_universe, SIZE_BAND_LOW, SIZE_BAND_HIGH)
    metadata["peers_after_size"] = len(after_size_tight)

    # Use tight if sufficient, otherwise relax
    after_size = after_size_tight if len(after_size_tight) >= MIN_PEERS else \
                 apply_size_filter(sector_universe, SIZE_BAND_LOW_RELAXED, SIZE_BAND_HIGH_RELAXED)

    # ── Filter 3: Growth rate similarity ──────────────────────────────────
    # This is the key fix for the Adobe/Datadog problem.
    #
    # Why ±15 percentage points as the window?
    # A 15pp window means: if Adobe grows at 14%, we accept peers growing
    # anywhere from -1% to 29%. This includes mature growers (Salesforce at 15%,
    # Autodesk at 13%) while excluding hyper-growth names (Datadog at 39%).
    # 15pp is wide enough to give us a reasonable peer set without being so
    # wide that it becomes meaningless.
    #
    # What if we don't have CAGR data for the target?
    # We skip this filter — it's better to have a broader peer set with
    # unknown growth similarity than to have no peers at all.
    def apply_growth_filter(pool):
        if pd.isna(target_cagr) or "revenue_cagr_5yr" not in pool.columns:
            return pool  # can't filter without data — skip gracefully

        window = GROWTH_CAGR_WINDOW_PP / 100.0  # convert pp to decimal
        cagr_low  = target_cagr - window
        cagr_high = target_cagr + window
        return pool[
            pool["revenue_cagr_5yr"].between(cagr_low, cagr_high) |
            pool["revenue_cagr_5yr"].isna()  # keep companies with missing CAGR
                                             # (missing ≠ bad, just unreported)
        ]

    after_growth = apply_growth_filter(after_size)
    metadata["peers_after_growth"] = len(after_growth)
    metadata["filters_applied"].append("growth_band")

    # ── Filter 4: EBITDA margin similarity ────────────────────────────────
    # Filters out companies at a fundamentally different profitability stage.
    # A company burning cash (2% margin) and a company printing cash (35%
    # margin) are in different phases of their lifecycle and should not
    # be valued using the same multiples framework.
    def apply_margin_filter(pool):
        if pd.isna(target_margin) or "ebitda_margin" not in pool.columns:
            return pool

        window = MARGIN_WINDOW_PP / 100.0
        margin_low  = target_margin - window
        margin_high = target_margin + window
        return pool[
            pool["ebitda_margin"].between(margin_low, margin_high) |
            pool["ebitda_margin"].isna()
        ]

    after_margin = apply_margin_filter(after_growth)
    metadata["peers_after_margin"] = len(after_margin)
    metadata["filters_applied"].append("margin_band")

    # ── Fallback logic if filters are too tight ────────────────────────────
    # If the combined filters leave us with too few peers, we relax
    # progressively — first drop margin filter, then drop growth filter.
    # We never drop sector — cross-sector comps are never acceptable.
    if len(after_margin) < MIN_PEERS:
        if verbose:
            print(f"  ⚠ Only {len(after_margin)} peers after growth+margin filters — "
                  f"relaxing margin filter...")
        after_margin = after_growth  # drop margin filter, keep growth filter

    if len(after_margin) < MIN_PEERS:
        if verbose:
            print(f"  ⚠ Still only {len(after_margin)} peers after relaxing margin — "
                  f"relaxing growth filter too...")
        after_margin = after_size  # drop both growth and margin, keep sector+size

    # ── Sub-industry preference ────────────────────────────────────────────
    # Within the filtered set, prefer companies in the same sub-industry.
    # If enough sub-industry peers exist after all filters, use only those.
    sub_industry_filtered = after_margin[
        after_margin["sub_industry"] == target_sub_industry
    ]
    if len(sub_industry_filtered) >= MIN_PEERS:
        working_peers = sub_industry_filtered
        metadata["match_level"] = "sub-industry + growth/margin filtered"
    else:
        working_peers = after_margin
        metadata["match_level"] = "sector + growth/margin filtered"

    # ── IQR outlier removal on EV/EBITDA multiples ────────────────────────
    # After all peer filters, do a final pass to remove any company whose
    # EV/EBITDA multiple is statistically extreme relative to the rest of
    # the peer group. This is the last-resort catch for data anomalies.
    working_peers, outliers_removed = remove_multiple_outliers(
        working_peers, multiple_col="ev_to_ebitda"
    )
    metadata["outliers_removed"]    = outliers_removed
    metadata["peers_after_outlier"] = len(working_peers)

    # ── Sort by EV proximity to target ────────────────────────────────────
    if pd.notna(target_ev) and "enterprise_value" in working_peers.columns:
        working_peers = working_peers.copy()
        working_peers["_ev_distance"] = (
            working_peers["enterprise_value"] - target_ev
        ).abs()
        working_peers = working_peers.sort_values("_ev_distance").drop(
            columns=["_ev_distance"]
        )

    peers = working_peers.head(MAX_PEERS)

    if verbose:
        print(f"\n  Peer selection pipeline:")
        print(f"    After sector filter:         {metadata['peers_after_sector']} companies")
        print(f"    After size filter:           {metadata['peers_after_size']} companies")
        print(f"    After growth filter (±{GROWTH_CAGR_WINDOW_PP:.0f}pp):  {metadata['peers_after_growth']} companies")
        print(f"    After margin filter (±{MARGIN_WINDOW_PP:.0f}pp):  {metadata['peers_after_margin']} companies")
        if outliers_removed:
            print(f"    After outlier removal:       {metadata['peers_after_outlier']} companies")
            print(f"    Outliers removed:            {', '.join(outliers_removed)}")
        print(f"    Final peer count:            {len(peers)} companies")
        print(f"    Match level:                 {metadata['match_level']}")

    return target, peers, metadata







# ── Step 2: Multiple computation ───────────────────────────────────────────────

def compute_peer_multiples(peers: pd.DataFrame) -> dict:
    """
    For the peer group, compute the distribution of each valuation multiple.

    We compute three multiples:
      EV/EBITDA  — the primary M&A multiple (use for profitable companies)
      EV/Revenue — the fallback for unprofitable or high-growth companies
      P/E        — equity-level multiple, for reference only

    For each multiple we report:
      median  — the central estimate (robust to outliers; preferred over mean)
      mean    — for reference
      pct_25  — lower bound of the "normal" range
      pct_75  — upper bound of the "normal" range
      min/max — the full observed range

    Why median over mean?
    If 9 peers trade at 15x EV/EBITDA and 1 peer trades at 150x
    (perhaps due to a recent earnings collapse), the mean is 28.5x —
    a number that doesn't represent any actual peer. The median is 15x,
    which correctly reflects the central tendency.
    """
    multiples = {}

    for multiple_col, label in [
        ("ev_to_ebitda",  "EV/EBITDA"),
        ("ev_to_revenue", "EV/Revenue"),
        ("pe_ratio",      "P/E"),
    ]:
        if multiple_col not in peers.columns:
            continue

        values = peers[multiple_col].dropna()

        # We need at least 3 data points for statistics to be meaningful.
        # With only 1–2 peers, the "median" is just one company's number.
        if len(values) < 3:
            multiples[multiple_col] = {
                "label":   label,
                "n":       len(values),
                "median":  values.median() if len(values) > 0 else np.nan,
                "mean":    np.nan,
                "pct_25":  np.nan,
                "pct_75":  np.nan,
                "min":     np.nan,
                "max":     np.nan,
                "reliable": False,  # flag: not enough data for confidence
            }
        else:
            multiples[multiple_col] = {
                "label":    label,
                "n":        len(values),
                "median":   values.median(),
                "mean":     values.mean(),
                "pct_25":   values.quantile(0.25),
                "pct_75":   values.quantile(0.75),
                "min":      values.min(),
                "max":      values.max(),
                "reliable": True,
            }

    return multiples


# ── Step 3: Implied Valuation ──────────────────────────────────────────────────

def compute_implied_valuation(
        target: pd.Series,
        multiples: dict,
        ebitda_growth: float = 0.0
) -> dict:
    """
    Apply peer multiples to the target's own financials to derive
    an implied enterprise value and implied share price.

    The ebitda_growth parameter is the scenario analysis lever.
    A value of 0.20 means "assume EBITDA is 20% higher post-acquisition"
    which models operational improvements an acquirer expects to achieve.

    Step-by-step calculation:
      1. Take the target's EBITDA and apply the growth assumption
         Scenario EBITDA = Actual EBITDA × (1 + ebitda_growth)

      2. Multiply by the peer median EV/EBITDA to get Implied EV
         Implied EV = Peer Median EV/EBITDA × Scenario EBITDA

      3. Subtract Net Debt to get Implied Equity Value
         Implied Equity = Implied EV - Net Debt
         (Net Debt = Total Debt - Cash)

      4. Divide by shares outstanding to get Implied Share Price
         Implied Price = Implied Equity / Shares Outstanding

      5. Compute upside/downside vs current price
         Upside = (Implied Price / Current Price - 1) × 100%

    Why do we subtract net debt?
    EV represents the total cost to acquire the WHOLE business (equity +
    debt - cash). But shareholders only own the equity portion. So to go
    from "what is the whole business worth" to "what is each share worth",
    we have to subtract the debt (which belongs to creditors, not equity
    holders) and add back the cash (which belongs to shareholders).
    """
    result = {
        "ebitda_growth_assumption": ebitda_growth,
        "scenario_label": f"{ebitda_growth:+.0%} EBITDA growth",
    }

    # ── Base financials ────────────────────────────────────────────────────
    actual_ebitda      = target.get("ebitda", np.nan)
    actual_revenue     = target.get("revenue", np.nan)
    net_debt           = target.get("net_debt", np.nan)
    shares_outstanding = target.get("shares_outstanding", np.nan)
    current_price      = target.get("price", np.nan)
    net_income         = target.get("net_income", np.nan)

    # Apply growth assumption to EBITDA
    scenario_ebitda = actual_ebitda * (1 + ebitda_growth) if pd.notna(actual_ebitda) else np.nan
    scenario_revenue = actual_revenue  # revenue stays the same (we only shock EBITDA)

    result["actual_ebitda"]    = actual_ebitda
    result["scenario_ebitda"]  = scenario_ebitda
    result["scenario_label"]   = (
        f"EBITDA {ebitda_growth:+.0%}" if ebitda_growth != 0 else "Base case"
    )

    # ── EV/EBITDA implied valuation ────────────────────────────────────────
    ev_ebitda_stats = multiples.get("ev_to_ebitda", {})
    peer_ev_ebitda_median = ev_ebitda_stats.get("median", np.nan)
    peer_ev_ebitda_p25    = ev_ebitda_stats.get("pct_25", np.nan)
    peer_ev_ebitda_p75    = ev_ebitda_stats.get("pct_75", np.nan)

    if pd.notna(peer_ev_ebitda_median) and pd.notna(scenario_ebitda) and scenario_ebitda > 0:
        implied_ev_median = peer_ev_ebitda_median * scenario_ebitda
        implied_ev_low    = peer_ev_ebitda_p25    * scenario_ebitda
        implied_ev_high   = peer_ev_ebitda_p75    * scenario_ebitda

        # Equity value = EV - Net Debt
        # If net_debt is negative (more cash than debt), this INCREASES equity value
        def ev_to_equity(ev):
            if pd.notna(ev) and pd.notna(net_debt):
                return ev - net_debt
            return np.nan

        implied_equity_median = ev_to_equity(implied_ev_median)
        implied_equity_low    = ev_to_equity(implied_ev_low)
        implied_equity_high   = ev_to_equity(implied_ev_high)

        # Per-share value
        def equity_to_price(equity):
            if pd.notna(equity) and pd.notna(shares_outstanding) and shares_outstanding > 0:
                return equity / shares_outstanding
            return np.nan

        implied_price_median = equity_to_price(implied_equity_median)
        implied_price_low    = equity_to_price(implied_equity_low)
        implied_price_high   = equity_to_price(implied_equity_high)

        # Upside/downside
        def compute_upside(implied_price):
            if pd.notna(implied_price) and pd.notna(current_price) and current_price > 0:
                return (implied_price / current_price - 1) * 100
            return np.nan

        result.update({
            "peer_ev_ebitda_median":    peer_ev_ebitda_median,
            "target_ev_ebitda":         target.get("ev_to_ebitda", np.nan),
            "implied_ev_median_bn":     implied_ev_median / 1e9 if pd.notna(implied_ev_median) else np.nan,
            "implied_ev_low_bn":        implied_ev_low    / 1e9 if pd.notna(implied_ev_low)    else np.nan,
            "implied_ev_high_bn":       implied_ev_high   / 1e9 if pd.notna(implied_ev_high)   else np.nan,
            "implied_price_median":     implied_price_median,
            "implied_price_low":        implied_price_low,
            "implied_price_high":       implied_price_high,
            "current_price":            current_price,
            "upside_median_pct":        compute_upside(implied_price_median),
            "upside_low_pct":           compute_upside(implied_price_low),
            "upside_high_pct":          compute_upside(implied_price_high),
            "ev_ebitda_reliable":       ev_ebitda_stats.get("reliable", False),
        })
    else:
        result.update({
            "peer_ev_ebitda_median":  np.nan,
            "implied_ev_median_bn":   np.nan,
            "implied_price_median":   np.nan,
            "upside_median_pct":      np.nan,
            "ev_ebitda_reliable":     False,
        })

    # ── EV/Revenue implied valuation (fallback / cross-check) ─────────────
    ev_revenue_stats = multiples.get("ev_to_revenue", {})
    peer_ev_rev_median = ev_revenue_stats.get("median", np.nan)

    if pd.notna(peer_ev_rev_median) and pd.notna(scenario_revenue) and scenario_revenue > 0:
        implied_ev_rev = peer_ev_rev_median * scenario_revenue
        implied_equity_rev = implied_ev_rev - net_debt if pd.notna(net_debt) else np.nan
        implied_price_rev = (
            implied_equity_rev / shares_outstanding
            if pd.notna(implied_equity_rev) and pd.notna(shares_outstanding) and shares_outstanding > 0
            else np.nan
        )
        result.update({
            "peer_ev_revenue_median":   peer_ev_rev_median,
            "target_ev_revenue":        target.get("ev_to_revenue", np.nan),
            "implied_ev_rev_bn":        implied_ev_rev / 1e9,
            "implied_price_rev":        implied_price_rev,
            "upside_rev_pct": (
                (implied_price_rev / current_price - 1) * 100
                if pd.notna(implied_price_rev) and pd.notna(current_price) and current_price > 0
                else np.nan
            ),
        })

    return result


# ── Step 4: Scenario Analysis Table ───────────────────────────────────────────

def build_scenario_table(
        target: pd.Series,
        multiples: dict
) -> pd.DataFrame:
    """
    Builds the full what-if scenario table across all EBITDA growth assumptions.

    This is the table that answers: "what is the implied share price if we
    buy this company and improve EBITDA by 0%, 10%, 20%, or 30%?"

    Each row is one scenario. The columns show the key valuation outputs
    for that scenario. This directly replicates the kind of sensitivity
    table that analysts build manually in Excel — but we generate it
    automatically for any company.
    """
    rows = []
    for growth in EBITDA_GROWTH_SCENARIOS:
        valuation = compute_implied_valuation(target, multiples, ebitda_growth=growth)
        rows.append({
            "scenario":              valuation["scenario_label"],
            "ebitda_growth":         growth,
            "scenario_ebitda_bn":    valuation.get("scenario_ebitda", np.nan) / 1e9
                                     if pd.notna(valuation.get("scenario_ebitda")) else np.nan,
            "implied_ev_low_bn":     valuation.get("implied_ev_low_bn",    np.nan),
            "implied_ev_median_bn":  valuation.get("implied_ev_median_bn", np.nan),
            "implied_ev_high_bn":    valuation.get("implied_ev_high_bn",   np.nan),
            "implied_price_low":     valuation.get("implied_price_low",    np.nan),
            "implied_price_median":  valuation.get("implied_price_median", np.nan),
            "implied_price_high":    valuation.get("implied_price_high",   np.nan),
            "upside_low_pct":        valuation.get("upside_low_pct",       np.nan),
            "upside_median_pct":     valuation.get("upside_median_pct",    np.nan),
            "upside_high_pct":       valuation.get("upside_high_pct",      np.nan),
        })

    return pd.DataFrame(rows)


# ── Full comps runner (the main callable) ──────────────────────────────────────

def run_comps(
        ticker: str,
        df: pd.DataFrame = None,
        verbose: bool = True
) -> dict:
    """
    The single entry point for the comps engine.
    Called by the Streamlit dashboard with a ticker string.
    Returns a dictionary containing everything needed to render
    the full comps analysis for that company.

    The return dict has these keys:
      "target"         → pd.Series  — the target company's full row
      "peers"          → pd.DataFrame — the peer group
      "metadata"       → dict — peer selection filter counts/details
      "multiples"      → dict — peer median multiples with full stats
      "base_valuation" → dict — implied valuation at 0% EBITDA growth
      "scenario_table" → pd.DataFrame — full what-if scenario table
      "peer_table"     → pd.DataFrame — clean comps table for display
    """
    if df is None:
        df = load_dataset()

    # Step 1: Find peers
    target, peers, metadata = select_peers(df, ticker, verbose=verbose)

    # Step 2: Compute multiples from peers
    multiples = compute_peer_multiples(peers)

    # Step 3: Base-case valuation (no EBITDA growth)
    base_valuation = compute_implied_valuation(target, multiples, ebitda_growth=0.0)

    # Step 4: Full scenario table
    scenario_table = build_scenario_table(target, multiples)

    # Step 5: Build a clean peer display table (just the columns we want to show)
    peer_display_cols = [
        "ticker", "company_name", "sub_industry",
        "ev_bn", "ev_to_ebitda", "ev_to_revenue",
        "ebitda_margin_pct", "revenue_cagr_pct",
        "debt_to_ebitda", "acquirability_score",
    ]
    available_peer_cols = [c for c in peer_display_cols if c in peers.columns]
    peer_table = peers[available_peer_cols].copy()

    return {
        "target":          target,
        "peers":           peers,
        "metadata":        metadata,
        "multiples":       multiples,
        "base_valuation":  base_valuation,
        "scenario_table":  scenario_table,
        "peer_table":      peer_table,
    }


# ── Pretty printer for terminal output ────────────────────────────────────────

def print_comps_report(ticker: str, results: dict):
    """
    Renders a human-readable comps report in the terminal.
    This is a debug/exploration tool — the actual dashboard
    will render this data visually in Streamlit.
    """
    target = results["target"]
    multiples = results["multiples"]
    base_val = results["base_valuation"]
    scenario_table = results["scenario_table"]
    peer_table = results["peer_table"]

    print(f"\n{'═'*65}")
    print(f"COMPS ANALYSIS: {target['company_name']} ({ticker})")
    print(f"{'═'*65}")

    print(f"\nTarget snapshot:")
    print(f"  Sector:          {target.get('sector', 'N/A')}")
    print(f"  EV:              ${target.get('ev_bn', np.nan):.2f}B")
    print(f"  Current price:   ${target.get('price', np.nan):.2f}")
    print(f"  EV/EBITDA:       {target.get('ev_to_ebitda', np.nan):.1f}x")
    print(f"  EBITDA margin:   {target.get('ebitda_margin_pct', np.nan):.1f}%")
    print(f"  Revenue CAGR:    {target.get('revenue_cagr_pct', np.nan):.1f}%")
    print(f"  Acq. Score:      {target.get('acquirability_score', np.nan):.1f} "
          f"({target.get('score_tier', 'N/A')})")

    print(f"\nPeer group ({len(results['peers'])} companies):")
    if not peer_table.empty:
        print(peer_table[
            ["ticker", "company_name", "ev_to_ebitda",
             "ebitda_margin_pct", "revenue_cagr_pct"]
        ].to_string(index=False))

    print(f"\nPeer multiple summary:")
    for key, stats in multiples.items():
        if pd.notna(stats.get("median")):
            print(f"  {stats['label']:<12} "
                  f"median={stats['median']:.1f}x   "
                  f"range=[{stats.get('pct_25', np.nan):.1f}x – "
                  f"{stats.get('pct_75', np.nan):.1f}x]   "
                  f"n={stats['n']}"
                  f"{'  ⚠ low confidence' if not stats.get('reliable') else ''}")

    print(f"\nBase-case implied valuation (0% EBITDA growth):")
    target_multiple   = base_val.get("target_ev_ebitda", np.nan)
    peer_multiple     = base_val.get("peer_ev_ebitda_median", np.nan)
    implied_price     = base_val.get("implied_price_median", np.nan)
    current_price     = base_val.get("current_price", np.nan)
    upside            = base_val.get("upside_median_pct", np.nan)
    implied_ev        = base_val.get("implied_ev_median_bn", np.nan)

    print(f"  Target EV/EBITDA:      {target_multiple:.1f}x")
    print(f"  Peer median EV/EBITDA: {peer_multiple:.1f}x")

    if pd.notna(target_multiple) and pd.notna(peer_multiple):
        discount = (target_multiple / peer_multiple - 1) * 100
        direction = "DISCOUNT to peers" if discount < 0 else "PREMIUM to peers"
        print(f"  Relative to peers:     {abs(discount):.1f}% {direction}")

    print(f"  Implied EV:            ${implied_ev:.2f}B" if pd.notna(implied_ev) else "  Implied EV:  N/A")
    print(f"  Implied share price:   ${implied_price:.2f}" if pd.notna(implied_price) else "  Implied price: N/A")
    print(f"  Current price:         ${current_price:.2f}" if pd.notna(current_price) else "  Current price: N/A")

    if pd.notna(upside):
        arrow = "▲" if upside > 0 else "▼"
        print(f"  Upside/downside:       {arrow} {abs(upside):.1f}% "
              f"({'undervalued' if upside > 0 else 'overvalued'} vs peers)")

    print(f"\nScenario analysis (what-if EBITDA growth post-acquisition):")
    print(f"  {'Scenario':<20} {'Impl. EV (median)':<22} "
          f"{'Impl. Price':<14} {'Upside vs today'}")
    print(f"  {'-'*20} {'-'*22} {'-'*14} {'-'*15}")
    for _, row in scenario_table.iterrows():
        impl_ev    = f"${row['implied_ev_median_bn']:.2f}B" if pd.notna(row['implied_ev_median_bn']) else "N/A"
        impl_price = f"${row['implied_price_median']:.2f}"  if pd.notna(row['implied_price_median']) else "N/A"
        upside_str = (f"{row['upside_median_pct']:+.1f}%"   if pd.notna(row['upside_median_pct'])   else "N/A")
        print(f"  {row['scenario']:<20} {impl_ev:<22} {impl_price:<14} {upside_str}")

    print(f"\n{'═'*65}\n")


# ── Batch runner (demonstrates the engine across multiple companies) ───────────

def run_batch_demo(
        tickers: list,
        df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Runs the comps engine for a list of tickers and collects
    the base-case implied valuation into a summary DataFrame.

    This is used to validate the engine across multiple companies
    before building the full dashboard. Run it on 10–20 companies
    to spot-check whether the implied valuations look reasonable.
    """
    if df is None:
        df = load_dataset()

    summary_rows = []

    for ticker in tickers:
        try:
            results = run_comps(ticker, df=df, verbose=False)
            target = results["target"]
            bv     = results["base_valuation"]
            peers  = results["peers"]

            summary_rows.append({
                "ticker":                ticker,
                "company_name":          target.get("company_name", ""),
                "sector":                target.get("sector", ""),
                "current_price":         bv.get("current_price", np.nan),
                "target_ev_ebitda":      bv.get("target_ev_ebitda", np.nan),
                "peer_median_ev_ebitda": bv.get("peer_ev_ebitda_median", np.nan),
                "implied_price":         bv.get("implied_price_median", np.nan),
                "upside_pct":            bv.get("upside_median_pct", np.nan),
                "n_peers":               len(peers),
                "acquirability_score":   target.get("acquirability_score", np.nan),
            })
            print(f"  ✓ {ticker:<6} {target.get('company_name',''):<35} "
                  f"upside: {bv.get('upside_median_pct', float('nan')):+.1f}%")

        except Exception as e:
            print(f"  ✗ {ticker:<6} FAILED: {e}")

    return pd.DataFrame(summary_rows)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load dataset once and reuse across all calls
    df = load_dataset()

    # --- Demo 1: single company deep-dive ---
    # Change this ticker to any S&P 500 company you want to analyse
    demo_ticker = "ADBE"
    results = run_comps(demo_ticker, df=df)
    print_comps_report(demo_ticker, results)

    # --- Demo 2: batch validation across a mix of sectors ---
    print("\nRunning batch validation across 10 companies...\n")
    demo_tickers = [
        "AAPL",   # Tech — large cap, trades at premium
        "ACN",    # Tech — consulting, should look cheap vs peers
        "JNJ",    # Healthcare — defensive, mature
        "PG",     # Consumer Staples — slow growth, high quality
        "XOM",    # Energy — cyclical, capital intensive
        "JPM",    # Financials — EV/EBITDA unreliable, expect warning
        "AMT",    # Real Estate — high EV multiples typical
        "CAT",    # Industrials — mid-cycle company
        "ADBE",   # Tech — high margin software
        "PEP",     # Consumer Staples — iconic brand
    ]
    summary = run_batch_demo(demo_tickers, df=df)

    print(f"\nBatch summary:")
    print(summary[[
        "ticker", "company_name", "target_ev_ebitda",
        "peer_median_ev_ebitda", "implied_price", "upside_pct", "n_peers"
    ]].to_string(index=False))

    # Save batch summary for reference
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summary.to_csv(f"{PROCESSED_DIR}/comps_batch_demo.csv", index=False)
    print(f"\nSaved batch summary → {PROCESSED_DIR}/comps_batch_demo.csv")
    print(f"\nPhase 4 complete. Ready for Phase 5: Streamlit Dashboard.")
