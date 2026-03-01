# Natural Gas Weather Signals

**Multi-phase research system that models U.S. natural gas storage using weather data, detects consensus surprises, and generates trading signals on NG futures.**

EIA weekly storage reports move natural gas prices — but the market reaction depends on whether the reported number beats or misses expectations. This project builds a weather-driven model to predict storage changes, measures deviations from consensus, and backtests whether those surprises create tradeable signals.

---

## Project Phases

### Phase 1 — Data Collection (`nat_gas_weather_collector.py`)
Pulls three data streams into a unified SQLite database:
- **EIA API v2**: Weekly natural gas storage by region (Lower 48 + 5 sub-regions, 1993–present)
- **NOAA CPC FTP**: Daily population-weighted heating/cooling degree days by state and region (1981–present)
- **yfinance**: Henry Hub front-month futures prices (NG=F)

Merges everything into a `master_weekly` table aligned by EIA reporting week.

### Phase 2 — Weather-Storage Relationship (`phase2_weather_storage_analysis.py`)
Analyzes how HDD/CDD degree days predict weekly storage changes:
- Regression models: HDD → storage withdrawals, CDD → storage injections
- Seasonal decomposition and residual analysis
- Measures predictive power of weather vs. historical averages

### Phase 2B — Consensus Proxy Model (`phase2b_consensus_enhanced.py`)
Builds a "consensus replacement" using the 5-year storage average:
- Backtests surprise signals (actual vs. 5-year average) against forward price moves
- Enhanced feature engineering to beat simple consensus
- Framework for incorporating real analyst consensus data

### Phase 2C — Storage Deviation Deep Dive (`phase2c_storage_deviation.py`)
The core strategy research — 8 tests to validate the signal:
1. Price mean-reversion baseline (is NG price alone predictive?)
2. Storage deviation after controlling for price level
3. Storage deviation velocity (worsening vs. improving)
4. Seasonal timing (which months matter most?)
5. Regime analysis (deficit vs. surplus as separate signals)
6. Combined signal: storage deviation + weather momentum
7. Walk-forward out-of-sample trading strategy
8. Drawdown and risk analysis

### Utilities
- `ng_consensus_accumulator.py` — Accumulates and tracks consensus estimates
- `eia_diagnostic.py` — Validates EIA data integrity
- `eia_storage_fix.py` — Repairs gaps in historical storage data

---

## Data Sources

| Source | Data | Frequency | History |
|--------|------|-----------|---------|
| EIA API v2 | Natural gas storage (Bcf) | Weekly | 1993–present |
| NOAA CPC | Population-weighted HDD/CDD | Daily | 1981–present |
| yfinance | NG=F futures prices | Daily | 1990–present |

---

## Setup

```bash
git clone https://github.com/KPH3802/natural-gas-weather-signals.git
cd natural-gas-weather-signals

pip install requests yfinance

# Phase 1: Collect data (requires free EIA API key)
# Get one at: https://www.eia.gov/opendata/register.php
python3 nat_gas_weather_collector.py YOUR_EIA_API_KEY

# Phase 2+: Run analysis (requires Phase 1 database)
python3 phase2_weather_storage_analysis.py
python3 phase2b_consensus_enhanced.py
python3 phase2c_storage_deviation.py
```

---

## Architecture

```
nat_gas_weather_collector.py      # Phase 1: EIA + NOAA + futures data pipeline
phase2_weather_storage_analysis.py # Phase 2: Weather → storage regression
phase2b_consensus_enhanced.py      # Phase 2B: Consensus proxy and enhanced model
phase2c_storage_deviation.py       # Phase 2C: Signal validation and OOS backtest
ng_consensus_accumulator.py        # Consensus tracking utility
eia_diagnostic.py                  # Data validation
eia_storage_fix.py                 # Historical data repair
data/                              # SQLite DB + CSV exports (not committed)
results/                           # Backtest outputs (not committed)
```

---

## Disclaimer

This project is for **educational and research purposes only**. Natural gas futures are highly leveraged instruments. Nothing here constitutes financial advice.

---

## Related Projects

This is part of a suite of quantitative research tools:

- [congress-trade-tracker](https://github.com/KPH3802/congress-trade-tracker) — Automated congressional stock trade tracking with 10 detection algorithms and 46K+ backtested signals
- [form4-insider-scanner](https://github.com/KPH3802/form4-insider-scanner) — SEC Form 4 insider transaction detection with cluster scoring and cross-signal enrichment
- [options-volume-scanner](https://github.com/KPH3802/options-volume-scanner) — Unusual options volume detection across S&P 500 stocks
- [volatility-scanner](https://github.com/KPH3802/volatility-scanner) — IV rank, HV patterns, and term structure tracking across 500+ instruments
- [trading-utilities](https://github.com/KPH3802/trading-utilities) — Shared data pipeline: 13F filings, FRED data, price history, dividends, earnings, short interest

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kevin--heaney-blue?logo=linkedin)](https://www.linkedin.com/in/kevin-heaney/)
[![Medium](https://img.shields.io/badge/Medium-@KPH3802-black?logo=medium)](https://medium.com/@KPH3802)
---

## License

MIT
