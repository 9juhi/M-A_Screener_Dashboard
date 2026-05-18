# pages/2_Company_Deep_Dive.py
# ─────────────────────────────────────────────────────────
# The analyst's working page.
# Given a ticker, shows:
#   1. Company snapshot card
#   2. Acquirability score breakdown (6-signal bar chart)
#   3. Peer comps table (with core / extended toggle)
#   4. Implied valuation — football field chart
#   5. Scenario analysis table (what-if EBITDA growth)
#
# The core/extended peer toggle is the key analytical
# feature — it lets the user see what changes when you
# include AI-premium, high-narrative companies in the comp set.
# ─────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import load_full_dataset, TIER_COLORS

# Import the extended-aware comps engine — this powers the Core/Extended toggle.
from comps_engine_2 import run_comps

st.title("🔍 Company Deep Dive")
st.markdown(
    "Select any S&P 500 company for a full comparable company analysis, "
    "implied valuation, and post-acquisition scenario modelling."
)

# ── Company selector ───────────────────────────────────────────────────────────
df = load_full_dataset()

# Build a display list: "AAPL — Apple Inc. (Information Technology)"
# Sorting alphabetically by ticker makes it easy to find companies
ticker_options = sorted(df["ticker"].dropna().unique().tolist())
company_names  = df.set_index("ticker")["company_name"].to_dict()

selected_ticker = st.selectbox(
    "Select a company",
    options=ticker_options,
    format_func=lambda t: f"{t} — {company_names.get(t, '')}",
    index=ticker_options.index("ADBE") if "ADBE" in ticker_options else 0,
    help="Type to search by ticker or company name"
)

# ── Core / Extended peer set toggle ───────────────────────────────────────────
# This is the key analytical feature explained in the project narrative.
# We display it prominently right below the company selector so users
# immediately understand it's a first-class analytical choice.

st.divider()
col_toggle, col_explainer = st.columns([1, 3])

with col_toggle:
    peer_mode = st.radio(
        "Peer set",
        options=["Core Peers", "Extended Peer Set"],
        index=0,
        help="Core: excludes companies trading at >2× sector median multiple. "
             "Extended: includes all sector peers regardless of multiple premium.",
    )

with col_explainer:
    if peer_mode == "Core Peers":
        st.info(
            "**Core Peers** excludes companies trading at an extreme premium to "
            "their sector median EV/EBITDA. This produces a conservative, "
            "defensible acquisition price anchor — what you would present to "
            "a CFO as the base-case offer range."
        )
    else:
        st.warning(
            "**Extended Peer Set** includes AI-premium and high-narrative peers. "
            "This shows where the market currently prices the most expensive "
            "companies with similar financials — useful for the bull case "
            "strategic narrative, but not a conservative acquisition price."
        )

use_extended = (peer_mode == "Extended Peer Set")

# ── Run the comps engine ───────────────────────────────────────────────────────
# We cache the comps result keyed on (ticker, peer_mode) so switching between
# core and extended doesn't re-run the engine unnecessarily.

@st.cache_data(ttl=3600, show_spinner="Running comps analysis...")
def get_comps(ticker: str, extended: bool) -> dict:
    """Cached wrapper around run_comps so the engine only re-runs when inputs change."""
    return run_comps(ticker, verbose=False, use_extended_peers=extended)

with st.spinner(f"Analysing {selected_ticker}..."):
    try:
        results       = get_comps(selected_ticker, use_extended)
        target        = results["target"]
        peer_table    = results["peer_table"]
        multiples     = results["multiples"]
        base_val      = results["base_valuation"]
        scenario_table = results["scenario_table"]
        metadata      = results["metadata"]
    except Exception as e:
        st.error(f"Could not run comps analysis for {selected_ticker}: {e}")
        st.stop()

st.divider()

# ── Target snapshot ────────────────────────────────────────────────────────────
tier        = target.get("score_tier", "Unrated")
tier_color  = TIER_COLORS.get(tier, "#6B7280")
score       = target.get("acquirability_score", np.nan)
global_rank = target.get("global_rank", np.nan)
sector_rank = target.get("sector_rank", np.nan)
sector_size = target.get("sector_size", np.nan)

st.subheader(f"{target.get('company_name', selected_ticker)} ({selected_ticker})")
st.markdown(
    f"**{target.get('sector', '')}** · {target.get('sub_industry', '')} · "
    f"Ranked **#{int(global_rank) if pd.notna(global_rank) else 'N/A'}** globally · "
    f"**#{int(sector_rank) if pd.notna(sector_rank) else 'N/A'} of "
    f"{int(sector_size) if pd.notna(sector_size) else 'N/A'}** in sector"
)

# Tier badge
st.markdown(
    f'<span style="background:{tier_color};color:white;padding:4px 12px;'
    f'border-radius:6px;font-size:14px;font-weight:700;">{tier} — {score:.1f}/100</span>',
    unsafe_allow_html=True,
)

st.markdown("")  # spacer

# Snapshot metrics in a 6-column row
s1, s2, s3, s4, s5, s6 = st.columns(6)
with s1:
    st.metric("EV", f"${target.get('ev_bn', np.nan):.1f}B")
with s2:
    st.metric("Current Price", f"${target.get('price', np.nan):.2f}")
with s3:
    st.metric("EV/EBITDA", f"{target.get('ev_to_ebitda', np.nan):.1f}x")
with s4:
    st.metric("EBITDA Margin", f"{target.get('ebitda_margin_pct', np.nan):.1f}%")
with s5:
    st.metric("Rev CAGR (5yr)", f"{target.get('revenue_cagr_pct', np.nan):.1f}%")
with s6:
    st.metric("Debt/EBITDA", f"{target.get('debt_to_ebitda', np.nan):.1f}x")

if target.get("data_quality_warning"):
    st.warning(target["data_quality_warning"])

st.divider()

# ── Two-column layout: score breakdown + comps table ──────────────────────────
left_col, right_col = st.columns([1, 1.6])

with left_col:
    st.subheader("Acquirability Score Breakdown")
    st.caption(
        "Each bar shows this signal's contribution to the total score. "
        "Length = weight × percentile rank within sector."
    )

    # Build the signal breakdown chart from the contribution columns
    signal_labels = {
        "ev_to_ebitda_contribution":      "Valuation (EV/EBITDA)",
        "revenue_cagr_5yr_contribution":  "Growth (Rev CAGR)",
        "ebitda_margin_contribution":     "Margin Quality",
        "debt_to_ebitda_contribution":    "Leverage Headroom",
        "fcf_margin_contribution":        "FCF Strength",
        "interest_coverage_contribution": "Financial Safety",
    }

    signal_data = []
    for col, label in signal_labels.items():
        val = target.get(col, np.nan)
        if pd.notna(val):
            signal_data.append({"Signal": label, "Contribution": val})

    if signal_data:
        signal_df = pd.DataFrame(signal_data).sort_values("Contribution")
        colors = [
            "#10B981" if v >= 15 else "#3B82F6" if v >= 10 else "#F59E0B" if v >= 5 else "#EF4444"
            for v in signal_df["Contribution"]
        ]
        fig_signals = go.Figure(go.Bar(
            x=signal_df["Contribution"],
            y=signal_df["Signal"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in signal_df["Contribution"]],
            textposition="outside",
        ))
        fig_signals.add_vline(
            x=score / 6,  # average contribution if perfectly balanced
            line_dash="dot",
            line_color="gray",
            annotation_text="avg",
            annotation_position="top",
        )
        fig_signals.update_layout(
            height=300,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(range=[0, 25], title="Points contributed (max 25)"),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_signals, use_container_width=True)
    else:
        st.caption("Score breakdown not available.")

with right_col:
    st.subheader(f"Peer Group ({len(peer_table)} companies)")

    # Show which peers were excluded by the sector cap filter
    if metadata.get("sector_cap_removed") and not use_extended:
        removed = ", ".join(metadata["sector_cap_removed"])
        st.caption(
            f"⚠ Excluded by sector multiple cap (>{ metadata.get('sector_cap_threshold', 0):.0f}x): "
            f"**{removed}** — switch to Extended Peer Set to include them."
        )

    if not peer_table.empty:
        peer_display = peer_table.copy()

        # Rename for display
        col_map = {
            "ticker":            "Ticker",
            "company_name":      "Company",
            "ev_bn":             "EV ($B)",
            "ev_to_ebitda":      "EV/EBITDA",
            "ebitda_margin_pct": "EBITDA Margin %",
            "revenue_cagr_pct":  "Rev CAGR %",
            "acquirability_score": "Score",
        }
        peer_display = peer_display.rename(columns={
            k: v for k, v in col_map.items() if k in peer_display.columns
        })

        st.dataframe(
            peer_display,
            use_container_width=True,
            height=280,
            column_config={
                "EV ($B)":        st.column_config.NumberColumn(format="$%.1fB"),
                "EV/EBITDA":      st.column_config.NumberColumn(format="%.1fx"),
                "EBITDA Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                "Rev CAGR %":     st.column_config.NumberColumn(format="%.1f%%"),
                "Score":          st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            },
            hide_index=True,
        )

        # Peer multiple summary beneath the table
        pm_cols = st.columns(3)
        for i, (key, stats) in enumerate(multiples.items()):
            if i >= 3:
                break
            with pm_cols[i]:
                if pd.notna(stats.get("median")):
                    reliable = "✓" if stats.get("reliable") else "⚠"
                    st.metric(
                        f"{reliable} Peer Median {stats['label']}",
                        f"{stats['median']:.1f}x",
                        delta=f"Range: {stats.get('pct_25', 0):.1f}x–{stats.get('pct_75', 0):.1f}x",
                    )
    else:
        st.caption("No peer data available.")

st.divider()

# ── Football field chart — implied valuation range ────────────────────────────
st.subheader("Implied Valuation — Football Field")
st.caption(
    "Each bar shows the implied share price range (P25 to P75 peer multiple) "
    "for a given valuation method. The dashed line is the current trading price."
)

current_price = base_val.get("current_price", np.nan)
football_data = []

# EV/EBITDA range
ev_low   = base_val.get("implied_price_low",   np.nan)
ev_med   = base_val.get("implied_price_median", np.nan)
ev_high  = base_val.get("implied_price_high",   np.nan)
if pd.notna(ev_low) and pd.notna(ev_high):
    football_data.append({
        "Method": "EV/EBITDA comps",
        "Low":    ev_low,
        "Median": ev_med,
        "High":   ev_high,
    })

# EV/Revenue range
rev_low  = base_val.get("implied_price_rev", np.nan)
if pd.notna(rev_low):
    football_data.append({
        "Method": "EV/Revenue comps",
        "Low":    rev_low * 0.85,   # approximate range using ±15% of point estimate
        "Median": rev_low,
        "High":   rev_low * 1.15,
    })

if football_data:
    ff_df = pd.DataFrame(football_data)

    fig_ff = go.Figure()
    for _, row in ff_df.iterrows():
        # Range bar (low to high)
        fig_ff.add_trace(go.Bar(
            name=row["Method"],
            y=[row["Method"]],
            x=[row["High"] - row["Low"]],
            base=[row["Low"]],
            orientation="h",
            marker_color="#3B82F6",
            opacity=0.4,
            showlegend=False,
        ))
        # Median point
        fig_ff.add_trace(go.Scatter(
            name=f"{row['Method']} median",
            y=[row["Method"]],
            x=[row["Median"]],
            mode="markers",
            marker=dict(symbol="line-ns", size=20, color="#1E40AF",
                        line=dict(width=3, color="#1E40AF")),
            showlegend=False,
        ))

    # Current price line
    if pd.notna(current_price):
        fig_ff.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="#EF4444",
            line_width=2,
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top",
            annotation_font_color="#EF4444",
        )

    fig_ff.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Implied Share Price ($)",
        yaxis_title="",
        barmode="overlay",
    )
    st.plotly_chart(fig_ff, use_container_width=True)
else:
    st.caption("Insufficient data to render football field chart.")

st.divider()

# ── Scenario analysis ──────────────────────────────────────────────────────────
st.subheader("Post-Acquisition Scenario Analysis")
st.caption(
    "What is this company worth if the acquirer improves EBITDA by X% "
    "through operational synergies? Each row is one scenario."
)

if not scenario_table.empty:
    # Scenario table
    scenario_display = scenario_table[[
        "scenario", "scenario_ebitda_bn",
        "implied_ev_low_bn", "implied_ev_median_bn", "implied_ev_high_bn",
        "implied_price_median", "upside_median_pct"
    ]].copy()

    scenario_display.columns = [
        "Scenario", "EBITDA ($B)",
        "Impl. EV Low ($B)", "Impl. EV Median ($B)", "Impl. EV High ($B)",
        "Impl. Price (Median)", "Upside vs Today %"
    ]

    st.dataframe(
        scenario_display,
        use_container_width=True,
        column_config={
            "EBITDA ($B)":          st.column_config.NumberColumn(format="$%.2fB"),
            "Impl. EV Low ($B)":    st.column_config.NumberColumn(format="$%.1fB"),
            "Impl. EV Median ($B)": st.column_config.NumberColumn(format="$%.1fB"),
            "Impl. EV High ($B)":   st.column_config.NumberColumn(format="$%.1fB"),
            "Impl. Price (Median)": st.column_config.NumberColumn(format="$%.2f"),
            "Upside vs Today %":    st.column_config.NumberColumn(format="%+.1f%%"),
        },
        hide_index=True,
    )

    # Scenario bar chart — implied price by scenario
    fig_scenario = px.bar(
        scenario_table,
        x="scenario",
        y="implied_price_median",
        color="upside_median_pct",
        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
        labels={
            "scenario":             "Scenario",
            "implied_price_median": "Implied Share Price ($)",
            "upside_median_pct":    "Upside %"
        },
        text_auto=".0f",
    )
    if pd.notna(current_price):
        fig_scenario.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#6B7280",
            annotation_text=f"Current price: ${current_price:.2f}",
            annotation_position="top left",
        )
    fig_scenario.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    fig_scenario.update_traces(textposition="outside")
    st.plotly_chart(fig_scenario, use_container_width=True)
