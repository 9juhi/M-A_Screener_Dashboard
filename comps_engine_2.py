import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR


MIN_PEERS = 5
MAX_PEERS = 12


SIZE_BAND_LOW  = 0.3
SIZE_BAND_HIGH = 3.0
SIZE_BAND_LOW_RELAXED  = 0.1
SIZE_BAND_HIGH_RELAXED = 10.0


SECTOR_MEDIAN_MULTIPLE_CAP = 2.0


GROWTH_CAGR_WINDOW_PP = 15.0


MARGIN_WINDOW_PP = 25.0


IQR_MULTIPLIER = 1.5

EBITDA_GROWTH_SCENARIOS = [-0.10, 0.00, 0.10, 0.20, 0.30]


def load_dataset() -> pd.DataFrame:
    path = f"{PROCESSED_DIR}/acquirability_scores.parquet"
    return pd.read_parquet(path)


def remove_multiple_outliers(
        peers: pd.DataFrame,
        multiple_col: str = "ev_to_ebitda",
        iqr_multiplier: float = IQR_MULTIPLIER
) -> tuple[pd.DataFrame, list]:


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


def apply_sector_multiple_cap(
        peers: pd.DataFrame,
        sector: str,
        benchmarks: pd.DataFrame,
        cap_multiplier: float = SECTOR_MEDIAN_MULTIPLE_CAP,
        multiple_col: str = "ev_to_ebitda"
) -> tuple[pd.DataFrame, list]:


    if multiple_col not in peers.columns or benchmarks is None:
        return peers, []


    sector_row = benchmarks[
        (benchmarks["sector"] == sector) &
        (benchmarks["metric"] == multiple_col)
    ]
    if sector_row.empty:
        return peers, []

    sector_median = sector_row.iloc[0]["median"]
    if pd.isna(sector_median) or sector_median <= 0:
        return peers, []

    upper_cap = sector_median * cap_multiplier

    outlier_mask    = peers[multiple_col] > upper_cap
    removed_tickers = peers[outlier_mask]["ticker"].tolist()
    cleaned_peers   = peers[~outlier_mask]

    return cleaned_peers, removed_tickers, upper_cap


def select_peers(
        df: pd.DataFrame,
        target_ticker: str,
        verbose: bool = True,
        use_extended_peers: bool = False
) -> tuple[pd.Series, pd.DataFrame, dict]:


    target_mask = df["ticker"] == target_ticker
    if not target_mask.any():
        raise ValueError(f"Ticker '{target_ticker}' not found in dataset.")

    target = df[target_mask].iloc[0]
    target_ev           = target.get("enterprise_value", np.nan)
    target_sector       = target.get("sector", "")
    target_sub_industry = target.get("sub_industry", "")
    target_cagr         = target.get("revenue_cagr_5yr", np.nan)
    target_margin       = target.get("ebitda_margin", np.nan)


    try:
        benchmarks = pd.read_parquet(f"{PROCESSED_DIR}/sector_benchmarks.parquet")
    except FileNotFoundError:
        benchmarks = None
        if verbose:
            print("  ⚠ sector_benchmarks.parquet not found — "
                  "sector multiple cap filter will be skipped")

    if verbose:
        print(f"\nFinding peers for: {target['company_name']} ({target_ticker})")
        print(f"  Sector:        {target_sector}")
        print(f"  Sub-industry:  {target_sub_industry}")
        print(f"  EV:            ${target_ev/1e9:.2f}B" if pd.notna(target_ev) else "  EV: N/A")
        print(f"  Revenue CAGR:  {target_cagr*100:.1f}%" if pd.notna(target_cagr) else "  CAGR: N/A")
        print(f"  EBITDA margin: {target_margin*100:.1f}%" if pd.notna(target_margin) else "  Margin: N/A")

    universe = df[df["ticker"] != target_ticker].copy()
    metadata = {
        "target_ticker":          target_ticker,
        "filters_applied":        [],
        "peers_after_sector":     0,
        "peers_after_size":       0,
        "peers_after_growth":     0,
        "peers_after_margin":     0,
        "peers_after_sector_cap": 0,
        "peers_after_outlier":    0,
        "match_level":            "",
        "outliers_removed":       [],
        "sector_cap_removed":     [],
        "sector_cap_threshold":   None,
    }


    sector_universe = universe[universe["sector"] == target_sector]
    metadata["peers_after_sector"] = len(sector_universe)


    def apply_size_filter(pool, low, high):
        if pd.notna(target_ev) and target_ev > 0:
            return pool[pool["enterprise_value"].between(target_ev * low, target_ev * high)]
        return pool

    after_size = apply_size_filter(sector_universe, SIZE_BAND_LOW, SIZE_BAND_HIGH)
    if len(after_size) < MIN_PEERS:
        after_size = apply_size_filter(sector_universe, SIZE_BAND_LOW_RELAXED, SIZE_BAND_HIGH_RELAXED)
    metadata["peers_after_size"] = len(after_size)


    def apply_growth_filter(pool):
        if pd.isna(target_cagr) or "revenue_cagr_5yr" not in pool.columns:
            return pool
        window = GROWTH_CAGR_WINDOW_PP / 100.0
        return pool[
            pool["revenue_cagr_5yr"].between(target_cagr - window, target_cagr + window) |
            pool["revenue_cagr_5yr"].isna()
        ]

    after_growth = apply_growth_filter(after_size)
    metadata["peers_after_growth"] = len(after_growth)


    def apply_margin_filter(pool):
        if pd.isna(target_margin) or "ebitda_margin" not in pool.columns:
            return pool
        window = MARGIN_WINDOW_PP / 100.0
        return pool[
            pool["ebitda_margin"].between(target_margin - window, target_margin + window) |
            pool["ebitda_margin"].isna()
        ]

    after_margin = apply_margin_filter(after_growth)
    if len(after_margin) < MIN_PEERS:
        after_margin = after_growth
    if len(after_margin) < MIN_PEERS:
        after_margin = after_size

    metadata["peers_after_margin"] = len(after_margin)


    if benchmarks is not None and not use_extended_peers:
        result = apply_sector_multiple_cap(
            after_margin, target_sector, benchmarks
        )

        after_sector_cap, sector_cap_removed, cap_threshold = result
        metadata["sector_cap_removed"]   = sector_cap_removed
        metadata["sector_cap_threshold"] = cap_threshold


        if len(after_sector_cap) >= MIN_PEERS:
            after_margin = after_sector_cap
        else:
            if verbose:
                print(f"  ⚠ Sector multiple cap would leave only "
                      f"{len(after_sector_cap)} peers — skipping cap filter "
                      f"to preserve minimum peer count")
            sector_cap_removed = []

        metadata["peers_after_sector_cap"] = len(after_margin)
    else:

        metadata["peers_after_sector_cap"] = len(after_margin)
        metadata["sector_cap_removed"]     = []
        if use_extended_peers and verbose:
            print("  ℹ Extended peer set mode — sector multiple cap bypassed")


    sub_industry_filtered = after_margin[
        after_margin["sub_industry"] == target_sub_industry
    ]
    if len(sub_industry_filtered) >= MIN_PEERS:
        working_peers = sub_industry_filtered
        metadata["match_level"] = "sub-industry + all filters"
    else:
        working_peers = after_margin
        metadata["match_level"] = "sector + all filters"


    working_peers, iqr_removed = remove_multiple_outliers(
        working_peers, multiple_col="ev_to_ebitda"
    )
    metadata["outliers_removed"]    = iqr_removed
    metadata["peers_after_outlier"] = len(working_peers)


    if pd.notna(target_ev) and "enterprise_value" in working_peers.columns:
        working_peers = working_peers.copy()
        working_peers["_ev_dist"] = (working_peers["enterprise_value"] - target_ev).abs()
        working_peers = working_peers.sort_values("_ev_dist").drop(columns=["_ev_dist"])

    peers = working_peers.head(MAX_PEERS)

    if verbose:
        print(f"\n  Peer selection pipeline:")
        print(f"    After sector filter:         {metadata['peers_after_sector']} companies")
        print(f"    After size filter:           {metadata['peers_after_size']} companies")
        print(f"    After growth filter (±{GROWTH_CAGR_WINDOW_PP:.0f}pp):  {metadata['peers_after_growth']} companies")
        print(f"    After margin filter (±{MARGIN_WINDOW_PP:.0f}pp):  {metadata['peers_after_margin']} companies")
        if metadata.get("sector_cap_threshold"):
            print(f"    After sector cap (>{metadata['sector_cap_threshold']:.1f}x):   "
                  f"{metadata['peers_after_sector_cap']} companies")
            if metadata["sector_cap_removed"]:
                print(f"    Sector cap removed:          {', '.join(metadata['sector_cap_removed'])}")
        if metadata["outliers_removed"]:
            print(f"    After IQR outlier removal:   {metadata['peers_after_outlier']} companies")
            print(f"    IQR outliers removed:        {', '.join(metadata['outliers_removed'])}")
        print(f"    Final peer count:            {len(peers)} companies")
        print(f"    Match level:                 {metadata['match_level']}")

    return target, peers, metadata


def compute_implied_valuation(
        target: pd.Series,
        multiples: dict,
        ebitda_growth: float = 0.0
) -> dict:


    result = {
        "ebitda_growth_assumption": ebitda_growth,
        "scenario_label": f"EBITDA {ebitda_growth:+.0%}" if ebitda_growth != 0 else "Base case",
    }

    actual_ebitda      = target.get("ebitda", np.nan)
    actual_revenue     = target.get("revenue", np.nan)
    net_debt           = target.get("net_debt", np.nan)
    shares_outstanding = target.get("shares_outstanding", np.nan)
    current_price      = target.get("price", np.nan)

    scenario_ebitda  = actual_ebitda  * (1 + ebitda_growth) if pd.notna(actual_ebitda)  else np.nan
    scenario_revenue = actual_revenue

    result["actual_ebitda"]   = actual_ebitda
    result["scenario_ebitda"] = scenario_ebitda


    ev_stats = multiples.get("ev_to_ebitda", {})
    p_median = ev_stats.get("median", np.nan)
    p_p25    = ev_stats.get("pct_25",  np.nan)
    p_p75    = ev_stats.get("pct_75",  np.nan)

    if pd.notna(p_median) and pd.notna(scenario_ebitda) and scenario_ebitda > 0:
        def to_price(ev):
            if not pd.notna(ev): return np.nan
            equity = ev - net_debt if pd.notna(net_debt) else np.nan
            if pd.notna(equity) and pd.notna(shares_outstanding) and shares_outstanding > 0:
                return equity / shares_outstanding
            return np.nan

        def upside(price):
            if pd.notna(price) and pd.notna(current_price) and current_price > 0:
                return (price / current_price - 1) * 100
            return np.nan

        implied_ev_median = p_median * scenario_ebitda
        implied_ev_low    = p_p25    * scenario_ebitda
        implied_ev_high   = p_p75    * scenario_ebitda

        result.update({
            "peer_ev_ebitda_median": p_median,
            "target_ev_ebitda":      target.get("ev_to_ebitda", np.nan),
            "implied_ev_median_bn":  implied_ev_median / 1e9,
            "implied_ev_low_bn":     implied_ev_low    / 1e9,
            "implied_ev_high_bn":    implied_ev_high   / 1e9,
            "implied_price_median":  to_price(implied_ev_median),
            "implied_price_low":     to_price(implied_ev_low),
            "implied_price_high":    to_price(implied_ev_high),
            "current_price":         current_price,
            "upside_median_pct":     upside(to_price(implied_ev_median)),
            "upside_low_pct":        upside(to_price(implied_ev_low)),
            "upside_high_pct":       upside(to_price(implied_ev_high)),
            "ev_ebitda_reliable":    ev_stats.get("reliable", False),
        })
    else:
        result.update({
            "peer_ev_ebitda_median": np.nan, "implied_ev_median_bn": np.nan,
            "implied_price_median":  np.nan, "upside_median_pct":    np.nan,
            "ev_ebitda_reliable":    False,
        })


    ev_rev_stats   = multiples.get("ev_to_revenue", {})
    p_rev_median   = ev_rev_stats.get("median", np.nan)

    if pd.notna(p_rev_median) and pd.notna(scenario_revenue) and scenario_revenue > 0:
        implied_ev_rev   = p_rev_median * scenario_revenue
        implied_eq_rev   = implied_ev_rev - net_debt if pd.notna(net_debt) else np.nan
        implied_price_rev = (
            implied_eq_rev / shares_outstanding
            if pd.notna(implied_eq_rev) and pd.notna(shares_outstanding) and shares_outstanding > 0
            else np.nan
        )
        result.update({
            "peer_ev_revenue_median": p_rev_median,
            "target_ev_revenue":      target.get("ev_to_revenue", np.nan),
            "implied_ev_rev_bn":      implied_ev_rev / 1e9,
            "implied_price_rev":      implied_price_rev,
            "upside_rev_pct": (
                (implied_price_rev / current_price - 1) * 100
                if pd.notna(implied_price_rev) and pd.notna(current_price) and current_price > 0
                else np.nan
            ),
        })

    return result


def compute_peer_multiples(peers: pd.DataFrame) -> dict:


    multiples = {}

    for multiple_col, label in [
        ("ev_to_ebitda",  "EV/EBITDA"),
        ("ev_to_revenue", "EV/Revenue"),
        ("pe_ratio",      "P/E"),
    ]:
        if multiple_col not in peers.columns:
            continue

        values = peers[multiple_col].dropna()

        if len(values) < 3:
            multiples[multiple_col] = {
                "label":    label,
                "n":        len(values),
                "median":   values.median() if len(values) > 0 else np.nan,
                "mean":     np.nan,
                "pct_25":   np.nan,
                "pct_75":   np.nan,
                "min":      np.nan,
                "max":      np.nan,
                "reliable": False,
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


def build_scenario_table(target: pd.Series, multiples: dict) -> pd.DataFrame:
    rows = []
    for growth in EBITDA_GROWTH_SCENARIOS:
        v = compute_implied_valuation(target, multiples, ebitda_growth=growth)
        rows.append({
            "scenario":             v["scenario_label"],
            "ebitda_growth":        growth,
            "scenario_ebitda_bn":   v.get("scenario_ebitda", np.nan) / 1e9
                                    if pd.notna(v.get("scenario_ebitda")) else np.nan,
            "implied_ev_low_bn":    v.get("implied_ev_low_bn",    np.nan),
            "implied_ev_median_bn": v.get("implied_ev_median_bn", np.nan),
            "implied_ev_high_bn":   v.get("implied_ev_high_bn",   np.nan),
            "implied_price_low":    v.get("implied_price_low",    np.nan),
            "implied_price_median": v.get("implied_price_median", np.nan),
            "implied_price_high":   v.get("implied_price_high",   np.nan),
            "upside_low_pct":       v.get("upside_low_pct",       np.nan),
            "upside_median_pct":    v.get("upside_median_pct",    np.nan),
            "upside_high_pct":      v.get("upside_high_pct",      np.nan),
        })
    return pd.DataFrame(rows)


def run_comps(
        ticker: str,
        df: pd.DataFrame = None,
        verbose: bool = True,
        use_extended_peers: bool = False
) -> dict:


    if df is None:
        df = load_dataset()

    target, peers, metadata = select_peers(df, ticker, verbose=verbose, use_extended_peers = use_extended_peers)
    multiples = compute_peer_multiples(peers)
    base_valuation = compute_implied_valuation(target, multiples, ebitda_growth=0.0)
    scenario_table = build_scenario_table(target, multiples)

    peer_display_cols = [
        "ticker", "company_name", "sub_industry",
        "ev_bn", "ev_to_ebitda", "ev_to_revenue",
        "ebitda_margin_pct", "revenue_cagr_pct",
        "debt_to_ebitda", "acquirability_score",
    ]
    available = [c for c in peer_display_cols if c in peers.columns]
    peer_table = peers[available].copy()

    return {
        "target":          target,
        "peers":           peers,
        "peer_table":      peer_table,
        "multiples":       multiples,
        "base_valuation":  base_valuation,
        "scenario_table":  scenario_table,
        "metadata":        metadata,
        "extended_peers": use_extended_peers,
    }


def print_comps_report(ticker: str, results: dict):
    target         = results["target"]
    multiples      = results["multiples"]
    base_val       = results["base_valuation"]
    scenario_table = results["scenario_table"]
    peer_table     = results["peer_table"]
    metadata       = results.get("metadata", {})

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

    if metadata.get("outliers_removed"):
        print(f"\n  ⚠ Outliers excluded from peer group: "
              f"{', '.join(metadata['outliers_removed'])}")
        print(f"    (EV/EBITDA multiples were statistical outliers "
              f"vs the peer group — IQR method)")

    print(f"\nPeer group ({len(results['peers'])} companies):")
    if not peer_table.empty:
        display_cols = ["ticker", "company_name", "ev_to_ebitda",
                        "ebitda_margin_pct", "revenue_cagr_pct"]
        available = [c for c in display_cols if c in peer_table.columns]
        print(peer_table[available].to_string(index=False))

    print(f"\nPeer multiple summary:")
    for key, stats in multiples.items():
        if pd.notna(stats.get("median")):
            reliable_flag = "" if stats.get("reliable") else "  ⚠ low confidence"
            print(f"  {stats['label']:<12} "
                  f"median={stats['median']:.1f}x   "
                  f"range=[{stats.get('pct_25', np.nan):.1f}x – "
                  f"{stats.get('pct_75', np.nan):.1f}x]   "
                  f"n={stats['n']}{reliable_flag}")

    print(f"\nBase-case implied valuation (0% EBITDA growth):")
    target_mult  = base_val.get("target_ev_ebitda", np.nan)
    peer_mult    = base_val.get("peer_ev_ebitda_median", np.nan)
    impl_price   = base_val.get("implied_price_median", np.nan)
    curr_price   = base_val.get("current_price", np.nan)
    upside       = base_val.get("upside_median_pct", np.nan)
    impl_ev      = base_val.get("implied_ev_median_bn", np.nan)

    print(f"  Target EV/EBITDA:      {target_mult:.1f}x")
    print(f"  Peer median EV/EBITDA: {peer_mult:.1f}x")
    if pd.notna(target_mult) and pd.notna(peer_mult):
        discount   = (target_mult / peer_mult - 1) * 100
        direction  = "DISCOUNT to peers" if discount < 0 else "PREMIUM to peers"
        print(f"  Relative to peers:     {abs(discount):.1f}% {direction}")

    print(f"  Implied EV:            ${impl_ev:.2f}B"    if pd.notna(impl_ev)    else "  Implied EV: N/A")
    print(f"  Implied share price:   ${impl_price:.2f}"  if pd.notna(impl_price) else "  Implied price: N/A")
    print(f"  Current price:         ${curr_price:.2f}"  if pd.notna(curr_price) else "  Current price: N/A")

    if pd.notna(upside):
        arrow = "▲" if upside > 0 else "▼"
        label = "undervalued" if upside > 0 else "overvalued"
        print(f"  Upside/downside:       {arrow} {abs(upside):.1f}% ({label} vs peers)")

    print(f"\nScenario analysis (what-if EBITDA growth post-acquisition):")
    print(f"  {'Scenario':<20} {'Impl. EV (median)':<22} "
          f"{'Impl. Price':<14} {'Upside vs today'}")
    print(f"  {'-'*20} {'-'*22} {'-'*14} {'-'*15}")
    for _, row in scenario_table.iterrows():
        ev_str    = f"${row['implied_ev_median_bn']:.2f}B" if pd.notna(row['implied_ev_median_bn']) else "N/A"
        price_str = f"${row['implied_price_median']:.2f}"  if pd.notna(row['implied_price_median']) else "N/A"
        up_str    = f"{row['upside_median_pct']:+.1f}%"    if pd.notna(row['upside_median_pct'])    else "N/A"
        print(f"  {row['scenario']:<20} {ev_str:<22} {price_str:<14} {up_str}")

    print(f"\n{'═'*65}\n")


def run_batch_demo(tickers: list, df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = load_dataset()
    rows = []
    for ticker in tickers:
        try:
            results = run_comps(ticker, df=df, verbose=False)
            target  = results["target"]
            bv      = results["base_valuation"]
            rows.append({
                "ticker":                ticker,
                "company_name":          target.get("company_name", ""),
                "sector":                target.get("sector", ""),
                "current_price":         bv.get("current_price", np.nan),
                "target_ev_ebitda":      bv.get("target_ev_ebitda", np.nan),
                "peer_median_ev_ebitda": bv.get("peer_ev_ebitda_median", np.nan),
                "implied_price":         bv.get("implied_price_median", np.nan),
                "upside_pct":            bv.get("upside_median_pct", np.nan),
                "n_peers":               len(results["peers"]),
                "acquirability_score":   target.get("acquirability_score", np.nan),
            })
            up = bv.get("upside_median_pct", float("nan"))
            print(f"  ✓ {ticker:<6} {target.get('company_name',''):<35} "
                  f"upside: {up:+.1f}%")
        except Exception as e:
            print(f"  ✗ {ticker:<6} FAILED: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()

    results = run_comps("AAPL", df=df)
    print_comps_report("AAPL", results)

    print("\nRunning batch validation...\n")
    demo_tickers = ["AAPL", "ACN", "JNJ", "PG", "XOM",
                    "JPM", "AMT", "CAT", "ADBE", "KO"]
    summary = run_batch_demo(demo_tickers, df=df)

    print(f"\nBatch summary:")
    print(summary[[
        "ticker", "company_name", "target_ev_ebitda",
        "peer_median_ev_ebitda", "implied_price", "upside_pct", "n_peers"
    ]].to_string(index=False))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summary.to_csv(f"{PROCESSED_DIR}/comps_batch_demo.csv", index=False)
    print(f"\nSaved batch summary → {PROCESSED_DIR}/comps_batch_demo.csv")
    print(f"\nPhase 4 complete. Ready for Phase 5: Streamlit Dashboard.")
