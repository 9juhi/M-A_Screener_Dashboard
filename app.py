import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(__file__))
from config import PROCESSED_DIR


st.set_page_config(
    page_title="M&A Target Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def load_leaderboard() -> pd.DataFrame:


    path = f"{PROCESSED_DIR}/leaderboard.parquet"
    df = pd.read_parquet(path)

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


    path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_sector_benchmarks() -> pd.DataFrame:


    path = f"{PROCESSED_DIR}/sector_benchmarks.parquet"
    return pd.read_parquet(path)


TIER_COLORS = {
    "S-tier": "#10B981",
    "A-tier": "#3B82F6",
    "B-tier": "#F59E0B",
    "C-tier": "#F97316",
    "D-tier": "#EF4444",
    "Unrated": "#6B7280",
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

    color = TIER_COLORS.get(tier, "#6B7280")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">{tier}</span>'


def metric_card(label: str, value: str, delta: str = None, color: str = "#3B82F6"):

    st.metric(label=label, value=value, delta=delta)


def show_landing_page():
    st.title("📊 M&A Target Screener & Comparable Valuation Dashboard")
    st.markdown(
        "A systematic, data-driven tool for identifying and valuing "
        "acquisition targets across the S&P 500."
    )
    st.divider()


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
