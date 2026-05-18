# app.py
# ─────────────────────────────────────────────────────────
# Root file of the Streamlit application.
# Responsibilities:
#   1. Global page configuration (title, layout, theme)
#   2. Shared data loading functions with caching
#      — every page imports from here so data loads once
#   3. Landing page content (overview and navigation guide)
#
# Why is data loading here and not in each page?
# Streamlit re-runs a file top-to-bottom on every interaction.
# If each page loaded its own data, you'd have three separate
# cache entries for the same 503-row DataFrame. Centralising
# the loaders here means all pages share one cached copy.
# ─────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add project root to path so we can import our pipeline modules
sys.path.append(os.path.dirname(__file__))
from config import PROCESSED_DIR

# ── Global page config ─────────────────────────────────────────────────────────
# This must be the FIRST Streamlit call in the entire app.
# wide layout gives us more horizontal space for tables and charts.
st.set_page_config(
    page_title="M&A Target Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared data loaders ────────────────────────────────────────────────────────
# @st.cache_data tells Streamlit: run this function once, store the result,
# return the cached copy on all subsequent calls.
# ttl=3600 means the cache expires after 1 hour — good for development.

@st.cache_data(ttl=3600)
def load_leaderboard() -> pd.DataFrame:
    """
    Loads the leaderboard Parquet produced by Phase 3.
    This is the primary data source for the Deal Universe page.
    Contains one row per company with scores, ranks, tiers, and key metrics.
    """
    path = f"{PROCESSED_DIR}/leaderboard.parquet"
    df = pd.read_parquet(path)
    # Round display columns to avoid messy floating point in tables
    for col in ["ev_to_ebitda", "debt_to_ebitda", "interest_coverage"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    for col in ["ebitda_margin_pct", "revenue_cagr_pct", "fcf_margin_pct",
                "acquirability_score"]:
        if col in df.columns:
            df[col] = df[col].round(1)
    return df


@st.cache_data(ttl=3600)
def load_full_dataset() -> pd.DataFrame:
    """
    Loads the full enriched + scored dataset produced by Phases 2 and 3.
    Used by the comps engine and the sector benchmarks page.
    Contains all 58 columns including rank columns.
    """
    path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_sector_benchmarks() -> pd.DataFrame:
    """
    Loads the sector benchmark statistics produced by Phase 2.
    Contains median, P25, P75 for each metric × sector combination.
    """
    path = f"{PROCESSED_DIR}/sector_benchmarks.parquet"
    return pd.read_parquet(path)


# ── Shared styling helpers ─────────────────────────────────────────────────────

TIER_COLORS = {
    "S-tier": "#10B981",   # green  — top acquisition candidates
    "A-tier": "#3B82F6",   # blue   — strong candidates
    "B-tier": "#F59E0B",   # amber  — average
    "C-tier": "#F97316",   # orange — below average
    "D-tier": "#EF4444",   # red    — avoid
    "Unrated": "#6B7280",  # gray
}

SECTOR_COLORS = {
    "Information Technology": "#3B82F6",
    "Health Care":            "#10B981",
    "Financials":             "#8B5CF6",
    "Consumer Discretionary": "#F59E0B",
    "Industrials":            "#F97316",
    "Consumer Staples":       "#EF4444",
    "Energy":                 "#06B6D4",
    "Real Estate":            "#EC4899",
    "Materials":              "#84CC16",
    "Communication Services": "#6366F1",
    "Utilities":              "#78716C",
}


def tier_badge(tier: str) -> str:
    """Returns an HTML badge string for a score tier."""
    color = TIER_COLORS.get(tier, "#6B7280")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{tier}</span>'


def metric_card(label: str, value: str, delta: str = None, color: str = "#3B82F6"):
    """Renders a single KPI card using Streamlit's metric component."""
    st.metric(label=label, value=value, delta=delta)


# ── Landing page ───────────────────────────────────────────────────────────────

def show_landing_page():
    st.title("📊 M&A Target Screener & Comparable Valuation Dashboard")
    st.markdown(
        "A systematic, data-driven tool for identifying and valuing "
        "acquisition targets across the S&P 500."
    )
    st.divider()

    # Load data to show live summary stats on the landing page
    try:
        df = load_leaderboard()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Companies Screened", f"{len(df):,}")
        with col2:
            st.metric("Sectors Covered", df["sector"].nunique())
        with col3:
            s_tier = (df["score_tier"] == "S-tier").sum()
            st.metric("S-tier Targets", s_tier)
        with col4:
            avg_score = df["acquirability_score"].mean()
            st.metric("Avg Acquirability Score", f"{avg_score:.1f}")
        with col5:
            complete = (df.get("imputed_fields", pd.Series([""])) == "").sum()
            pct = complete / len(df) * 100
            st.metric("Data Completeness", f"{pct:.0f}%")

        st.divider()

    except FileNotFoundError:
        st.error(
            "Data files not found. Please run the full pipeline first: "
            "`python main.py`"
        )
        return

    # Navigation guide
    st.subheader("How to use this dashboard")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            "**📋 Deal Universe**\n\n"
            "Screen all 503 companies by sector, size, and score tier. "
            "See the heatmap of where the most attractive targets are concentrated."
        )
    with col_b:
        st.info(
            "**🔍 Company Deep Dive**\n\n"
            "Select any company for a full comps analysis — peer group, "
            "implied valuation, scenario analysis, and scoring breakdown. "
            "Toggle between core and extended peer sets."
        )
    with col_c:
        st.info(
            "**📈 Sector Benchmarks**\n\n"
            "Understand what 'normal' looks like in each sector. "
            "Median multiples, growth rates, and margin profiles "
            "for all 11 GICS sectors."
        )

    st.divider()
    st.caption(
        "Data sourced from yfinance and SimFin API. "
        "Financial metrics based on most recent annual filings. "
        "This tool is for analytical and educational purposes only."
    )


if __name__ == "__main__" or True:
    show_landing_page()