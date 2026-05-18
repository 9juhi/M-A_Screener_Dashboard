# pages/1_Deal_Universe.py
# ─────────────────────────────────────────────────────────
# The Deal Universe page — the strategic overview.
#
# What an MD would use this page for:
#   "Show me the most attractive acquisition targets in
#    Healthcare with an EV under $50B."
#
# Layout:
#   Sidebar  → filters (sector, size, score tier, min score)
#   Top row  → 4 KPI cards reflecting filtered universe
#   Middle   → sector × tier heatmap
#   Bottom   → ranked leaderboard table
# ─────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import load_leaderboard, TIER_COLORS, SECTOR_COLORS

st.title("📋 Deal Universe")
st.markdown(
    "Screen the full S&P 500 universe and identify the most "
    "attractive acquisition targets based on the Acquirability Score."
)

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_leaderboard()

# ── Sidebar filters ────────────────────────────────────────────────────────────
# All filters live in the sidebar so they don't consume main-area space.
# Every filter widget returns a value that we use to slice the DataFrame.

st.sidebar.header("🔧 Filters")

# Sector filter — multiselect so you can compare 2-3 sectors at once
all_sectors = sorted(df["sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=all_sectors,
    default=all_sectors,
    help="Filter to specific GICS sectors"
)

# Score tier filter
all_tiers = ["S-tier", "A-tier", "B-tier", "C-tier", "D-tier"]
selected_tiers = st.sidebar.multiselect(
    "Score Tier",
    options=all_tiers,
    default=["S-tier", "A-tier", "B-tier"],
    help="S-tier = top 20% acquirability score"
)

# EV size filter — slider for the EV range in billions
# We clamp the slider range to the actual data range
ev_col = "ev_bn"
if ev_col in df.columns:
    ev_min = float(df[ev_col].dropna().quantile(0.05))
    ev_max = float(df[ev_col].dropna().quantile(0.95))
    ev_range = st.sidebar.slider(
        "Enterprise Value ($B)",
        min_value=round(ev_min, 1),
        max_value=round(ev_max, 1),
        value=(round(ev_min, 1), round(ev_max, 1)),
        help="Filter by company size (Enterprise Value in billions USD)"
    )
else:
    ev_range = (0, 9999)

# Minimum acquirability score
min_score = st.sidebar.slider(
    "Min Acquirability Score",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Only show companies scoring above this threshold"
)

# Data quality filter
show_imputed = st.sidebar.checkbox(
    "Include companies with imputed data",
    value=True,
    help="Uncheck to show only companies with fully complete financial data"
)

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = df.copy()

if selected_sectors:
    filtered = filtered[filtered["sector"].isin(selected_sectors)]

if selected_tiers:
    filtered = filtered[filtered["score_tier"].isin(selected_tiers)]

if ev_col in filtered.columns:
    filtered = filtered[
        filtered[ev_col].between(ev_range[0], ev_range[1]) |
        filtered[ev_col].isna()
    ]

filtered = filtered[filtered["acquirability_score"] >= min_score]

if not show_imputed and "imputed_fields" in filtered.columns:
    filtered = filtered[filtered["imputed_fields"] == ""]

# ── KPI summary row ────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Companies in View",
        f"{len(filtered):,}",
        delta=f"{len(filtered) - len(df):,} from total" if len(filtered) < len(df) else "All companies"
    )
with k2:
    s_count = (filtered["score_tier"] == "S-tier").sum()
    st.metric("S-tier Targets", s_count)
with k3:
    median_score = filtered["acquirability_score"].median()
    st.metric("Median Score", f"{median_score:.1f}")
with k4:
    if ev_col in filtered.columns:
        median_ev = filtered[ev_col].median()
        st.metric("Median EV", f"${median_ev:.1f}B")

st.divider()

# ── Sector × Tier heatmap ──────────────────────────────────────────────────────
# The heatmap shows, for each sector (rows) and tier (columns), how many
# companies fall in that cell. Dark colour = more targets concentrated there.
# An MD glancing at this immediately knows which sectors are richest in
# high-quality acquisition candidates.

st.subheader("Deal Universe Heatmap — Targets by Sector × Score Tier")
st.caption(
    "Each cell shows the number of companies in that sector with that score tier. "
    "Darker = more attractive targets concentrated in that sector."
)

tier_order = ["S-tier", "A-tier", "B-tier", "C-tier", "D-tier"]
heatmap_data = (
    filtered.groupby(["sector", "score_tier"])
    .size()
    .reset_index(name="count")
)
heatmap_pivot = heatmap_data.pivot(
    index="sector", columns="score_tier", values="count"
).reindex(columns=tier_order).fillna(0).astype(int)

fig_heatmap = px.imshow(
    heatmap_pivot,
    labels=dict(x="Score Tier", y="Sector", color="Company Count"),
    color_continuous_scale="Blues",
    aspect="auto",
    text_auto=True,
)
fig_heatmap.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    coloraxis_showscale=False,
    font=dict(size=12),
)
fig_heatmap.update_traces(textfont_size=13)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# ── Score distribution by sector ──────────────────────────────────────────────
st.subheader("Acquirability Score Distribution by Sector")
st.caption(
    "Box plot showing the spread of scores within each sector. "
    "A sector with many high-scoring companies is a rich hunting ground."
)

fig_box = px.box(
    filtered.dropna(subset=["acquirability_score"]),
    x="sector",
    y="acquirability_score",
    color="sector",
    color_discrete_map=SECTOR_COLORS,
    points=False,
    labels={"acquirability_score": "Acquirability Score", "sector": ""},
)
fig_box.update_layout(
    height=380,
    showlegend=False,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_tickangle=-30,
    yaxis=dict(range=[0, 100]),
)
st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Leaderboard table ──────────────────────────────────────────────────────────
st.subheader(f"Ranked Leaderboard — {len(filtered)} companies")
st.caption(
    "Sorted by Acquirability Score descending. "
    "Click column headers to re-sort. "
    "Use the filters in the sidebar to narrow the view."
)

# Select and rename columns for clean display
table_cols = {
    "global_rank":        "Rank",
    "ticker":             "Ticker",
    "company_name":       "Company",
    "sector":             "Sector",
    "acquirability_score": "Score",
    "score_tier":         "Tier",
    "ev_bn":              "EV ($B)",
    "ev_to_ebitda":       "EV/EBITDA",
    "ebitda_margin_pct":  "EBITDA Margin %",
    "revenue_cagr_pct":   "Rev CAGR %",
    "debt_to_ebitda":     "Debt/EBITDA",
}

available_cols = {k: v for k, v in table_cols.items() if k in filtered.columns}
display_df = (
    filtered[list(available_cols.keys())]
    .rename(columns=available_cols)
    .sort_values("Score", ascending=False)
    .reset_index(drop=True)
)

# Apply colour coding to the Tier column using Streamlit's column config
st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Score":          st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "EV ($B)":        st.column_config.NumberColumn("EV ($B)", format="$%.1fB"),
        "EV/EBITDA":      st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
        "EBITDA Margin %": st.column_config.NumberColumn("EBITDA Margin %", format="%.1f%%"),
        "Rev CAGR %":     st.column_config.NumberColumn("Rev CAGR %", format="%.1f%%"),
        "Debt/EBITDA":    st.column_config.NumberColumn("Debt/EBITDA", format="%.1fx"),
    },
    hide_index=True,
)

# Download button — analysts always want to export to Excel
csv = display_df.to_csv(index=False)
st.download_button(
    label="⬇ Download filtered results as CSV",
    data=csv,
    file_name="ma_screener_results.csv",
    mime="text/csv",
)