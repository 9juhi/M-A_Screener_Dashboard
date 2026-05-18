# M&A Screener Dashboard

A Streamlit dashboard for screening S&P 500 companies as potential acquisition targets using valuation, growth, profitability, leverage, cash flow, and sector benchmark signals.

The project combines market data, financial statement data, engineered metrics, an acquirability scoring model, and comparable company analysis into an interactive M&A target screening workflow.

## Features

- Deal universe view with sector, score tier, enterprise value, and data quality filters
- Acquirability score for each company based on sector-relative financial signals
- Company deep dive with peer selection, comparable valuation, scenario analysis, and score breakdown
- Sector benchmark page showing valuation and operating metric distributions
- Data pipeline for fetching, processing, scoring, and exporting analysis-ready datasets

## Project Structure

```text
.
├── app.py                         # Streamlit app entry point and shared loaders
├── main.py                        # Pipeline runner
├── fetch_sp500.py                 # S&P 500 universe fetch
├── fetch_market_data.py           # Market data collection
├── fetch_financials.py            # SimFin statement data collection
├── build_dataset.py               # Raw data merge into master dataset
├── compute_metrics.py             # Financial metric engineering
├── compute_sector_benchmarks.py   # Sector benchmark generation
├── scoring_engine.py              # Acquirability scoring model
├── comps_engine_2.py              # Comparable company analysis engine
├── pages/
│   ├── 1_Deal_Landscape.py
│   ├── 2_Company_Deep_Dive.py
│   └── 3_Sector_Benchmarks.py
├── data/
│   ├── raw/
│   └── processed/
└── requirements.txt
```


## Running the Dashboard

If the processed data files are already available in `data/processed`, start the Streamlit app:

```bash
streamlit run app.py
```

The app includes:

- `Deal Landscape`: filter and rank the full screened company universe
- `Company Deep Dive`: run comparable company valuation for a selected ticker
- `Sector Benchmarks`: compare sector-level valuation and operating metrics

## Rebuilding the Data

The pipeline is organized into phases:

```python
run_phase1()  # Fetch company universe, market data, and financials
run_phase2()  # Compute metrics and sector benchmarks
run_phase3()  # Generate acquirability scores and leaderboard
```

At the moment, `main.py` runs Phase 3 by default. To rebuild everything from scratch, update the `if __name__ == "__main__"` block in `main.py` to run the phases you need, then run:

```bash
python main.py
```

## Scoring Methodology

The acquirability score ranks companies from 0 to 100 using sector-relative signals:

- EV/EBITDA valuation attractiveness
- Revenue CAGR
- EBITDA margin
- Debt/EBITDA
- Free cash flow margin
- Interest coverage

The final score is mapped into tiers from `S-tier` to `D-tier` for easier screening.

## Notes

- Data is sourced from yfinance and SimFin.
- The dashboard is intended for analytical and educational use only.
- Outputs depend on data availability, API limits, and the freshness of local processed files.
- Do not commit API keys or private credentials to the repository.
