# pages/3_Sector_Benchmarks.py
# ─────────────────────────────────────────────────────────
# The Sector Benchmarks page — contextual reference layer.
#
# Answers: "what is a normal valuation for a company
# in this sector?" before you interpret the comps output.
#
# Shows: median multiples, margin and growth profiles,
# and the top targets within the selected sector.
# ─────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import load_leaderboard, load_sector_benchmarks, SECTOR_COLORS

st.title("📈 Sector Benchmarks")
st.markdown(
    "Reference statistics for each GICS sector. "
    "Use this page to understand what 'normal' looks like "
    "before interpreting any individual company's comps."
)

# ── Load data ──────────────────────────────────────────────────────────────────
leaderboard  = load_leaderboard()
benchmarks   = load_sector_benchmarks()

# ── Sector selector ────────────────────────────────────────────────────────────
all_sectors    = sorted(leaderboard["sector"].dropna().unique().tolist())
selected_sector = st.selectbox(
    "Select a sector",
    options=all_sectors,
    index=all_sectors.index("Information Technology") if "Information Technology" in all_sectors else 0,
)

sector_df    = leaderboard[leaderboard["sector"] == selected_sector]
sector_bench = benchmarks[benchmarks["sector"] == selected_sector]

st.divider()

# ── Sector summary metrics ─────────────────────────────────────────────────────
st.subheader(f"{selected_sector} — Sector Overview")

b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    st.metric("Companies", len(sector_df))
with b2:
    s_count = (sector_df["score_tier"] == "S-tier").sum()
    st.metric("S-tier Targets", s_count)
with b3:
    # Get median EV/EBITDA from benchmarks
    ev_row = sector_bench[sector_bench["metric"] == "ev_to_ebitda"]
    if not ev_row.empty:
        st.metric("Median EV/EBITDA", f"{ev_row.iloc[0]['median']:.1f}x")
    else:
        st.metric("Median EV/EBITDA", "N/A")
with b4:
    cagr_row = sector_bench[sector_bench["metric"] == "revenue_cagr_5yr"]
    if not cagr_row.empty:
        st.metric("Median Rev CAGR", f"{cagr_row.iloc[0]['median']*100:.1f}%")
    else:
        st.metric("Median Rev CAGR", "N/A")
with b5:
    margin_row = sector_bench[sector_bench["metric"] == "ebitda_margin"]
    if not margin_row.empty:
        st.metric("Median EBITDA Margin", f"{margin_row.iloc[0]['median']*100:.1f}%")
    else:
        st.metric("Median EBITDA Margin", "N/A")

st.divider()

# ── Cross-sector multiple comparison ──────────────────────────────────────────
st.subheader("EV/EBITDA Multiples Across All Sectors")
st.caption(
    "Median EV/EBITDA by sector with P25–P75 range shown as error bars. "
    "The selected sector is highlighted."
)

ev_bench = benchmarks[benchmarks["metric"] == "ev_to_ebitda"].copy()
ev_bench = ev_bench.sort_values("median", ascending=True)
ev_bench["highlight"] = ev_bench["sector"] == selected_sector

fig_cross = go.Figure()
for _, row in ev_bench.iterrows():
    color = SECTOR_COLORS.get(row["sector"], "#6B7280")
    opacity = 1.0 if row["highlight"] else 0.55
    fig_cross.add_trace(go.Bar(
        name=row["sector"],
        x=[row["sector"]],
        y=[row["median"]],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[row["pct_75"] - row["median"]],
            arrayminus=[row["median"] - row["pct_25"]],
        ),
        marker_color=color,
        opacity=opacity,
        showlegend=False,
        text=f"{row['median']:.1f}x",
        textposition="outside",
    ))

fig_cross.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    yaxis_title="EV/EBITDA (median)",
    xaxis_title="",
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_cross, use_container_width=True)

st.divider()

# ── Metric distributions within selected sector ────────────────────────────────
st.subheader(f"Metric Distributions — {selected_sector}")

metric_pairs = [
    ("ev_to_ebitda",      "EV/EBITDA",       "x"),
    ("ebitda_margin_pct", "EBITDA Margin",    "%"),
    ("revenue_cagr_pct",  "Revenue CAGR",     "%"),
    ("debt_to_ebitda",    "Debt/EBITDA",      "x"),
]

col1, col2 = st.columns(2)
charts = [col1, col2, col1, col2]

for i, (metric, label, unit) in enumerate(metric_pairs):
    if metric not in sector_df.columns:
        continue
    values = sector_df[metric].dropna()
    if len(values) < 3:
        continue

    fig_hist = px.histogram(
        values,
        nbins=20,
        labels={"value": f"{label} ({unit})"},
        color_discrete_sequence=[SECTOR_COLORS.get(selected_sector, "#3B82F6")],
    )
    fig_hist.update_layout(
        title=label,
        height=250,
        margin=dict(l=0, r=0, t=35, b=0),
        showlegend=False,
        yaxis_title="# companies",
    )
    with charts[i]:
        st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Top targets in sector ──────────────────────────────────────────────────────
st.subheader(f"Top Acquisition Targets — {selected_sector}")
st.caption("Companies in this sector ranked by Acquirability Score.")

top_sector = sector_df.sort_values("acquirability_score", ascending=False).head(15)

top_cols = {
    "sector_rank":        "Sector Rank",
    "ticker":             "Ticker",
    "company_name":       "Company",
    "score_tier":         "Tier",
    "acquirability_score": "Score",
    "ev_bn":              "EV ($B)",
    "ev_to_ebitda":       "EV/EBITDA",
    "ebitda_margin_pct":  "EBITDA Margin %",
    "revenue_cagr_pct":   "Rev CAGR %",
}
top_available = {k: v for k, v in top_cols.items() if k in top_sector.columns}
top_display = top_sector[list(top_available.keys())].rename(columns=top_available)

st.dataframe(
    top_display,
    use_container_width=True,
    column_config={
        "Score":          st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "EV ($B)":        st.column_config.NumberColumn(format="$%.1fB"),
        "EV/EBITDA":      st.column_config.NumberColumn(format="%.1fx"),
        "EBITDA Margin %": st.column_config.NumberColumn(format="%.1f%%"),
        "Rev CAGR %":     st.column_config.NumberColumn(format="%.1f%%"),
    },
    hide_index=True,
)