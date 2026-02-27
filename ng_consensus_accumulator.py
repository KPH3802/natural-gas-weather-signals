#!/usr/bin/env python3
"""
NAT GAS CONSENSUS DATA ACCUMULATOR
Captures weekly:
  1. Market consensus forecast (from FMP economic calendar)
  2. Actual EIA storage report (from EIA API)
  3. Our model's prediction (from weather regression)
  4. NG price at report time

Schedule on PythonAnywhere: Thursday 17:00 UTC (12pm CT, after 10:30am ET release)
This captures both the consensus AND actual in one shot after the report drops.

Also run: Wednesday 22:00 UTC (5pm CT) to capture pre-release consensus
(in case it differs from post-release snapshot)

Usage:
  python3 ng_consensus_accumulator.py           # Normal run
  python3 ng_consensus_accumulator.py --backfill # Try to pull historical consensus from FMP

Storage: data/nat_gas_weather.db -> consensus_tracking table
"""

import os
import sys
import json
import sqlite3
import math
import urllib.request
import urllib.error
from datetime import datetime, timedelta, date

# ============================================================
# CONFIGURATION
# ============================================================
DB_PATH = os.path.expanduser("~/Desktop/Claude_Programs/Trading_Programs/data/nat_gas_weather.db")

# PythonAnywhere path fallback
if not os.path.exists(os.path.dirname(DB_PATH)):
    DB_PATH = os.path.expanduser("~/nat_gas_weather/data/nat_gas_weather.db")

# API Keys
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "EwoNkAuyC5zfd4Spg1ji0NjQBiM8eAnRB9VpBwQw")

# If running locally, try to read from a config file
if not FMP_API_KEY and os.path.exists("config_keys.py"):
    try:
        exec(open("config_keys.py").read())
        FMP_API_KEY = locals().get("FMP_API_KEY", "")
    except:
        pass

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def api_get(url, headers=None):
    """Simple HTTP GET with error handling"""
    if headers is None:
        headers = {"User-Agent": USER_AGENT}
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url[:80]}...")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def iso_week(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.isocalendar()[1]

def multi_reg_predict(xs_list, y, x_new):
    """Fit and predict in one shot for model prediction"""
    n_vars = len(xs_list)
    complete = []
    for i in range(len(y)):
        if y[i] is None:
            continue
        all_good = True
        for xs in xs_list:
            if i >= len(xs) or xs[i] is None:
                all_good = False
                break
        if all_good:
            row = [xs[i] for xs in xs_list] + [y[i]]
            complete.append(row)
    
    if len(complete) < n_vars + 2:
        return None
    
    n = len(complete)
    x_cols = [[row[j] for row in complete] for j in range(n_vars)]
    y_col = [row[n_vars] for row in complete]
    
    means_x = [sum(col)/len(col) for col in x_cols]
    mean_y = sum(y_col)/len(y_col)
    
    S = [[0]*n_vars for _ in range(n_vars)]
    Sy = [0]*n_vars
    for i in range(n_vars):
        for j in range(n_vars):
            S[i][j] = sum((x_cols[i][k] - means_x[i]) * (x_cols[j][k] - means_x[j]) for k in range(n))
        Sy[i] = sum((x_cols[i][k] - means_x[i]) * (y_col[k] - mean_y) for k in range(n))
    
    aug = [S[i][:] + [Sy[i]] for i in range(n_vars)]
    for i in range(n_vars):
        max_row = max(range(i, n_vars), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        if abs(aug[i][i]) < 1e-12:
            return None
        for j in range(i+1, n_vars):
            factor = aug[j][i] / aug[i][i]
            for k in range(i, n_vars + 1):
                aug[j][k] -= factor * aug[i][k]
    
    betas = [0]*n_vars
    for i in range(n_vars - 1, -1, -1):
        betas[i] = aug[i][n_vars]
        for j in range(i+1, n_vars):
            betas[i] -= aug[i][j] * betas[j]
        betas[i] /= aug[i][i]
    
    intercept = mean_y - sum(betas[i] * means_x[i] for i in range(n_vars))
    return intercept + sum(betas[i] * x_new[i] for i in range(n_vars))


# ============================================================
# DATABASE SETUP
# ============================================================
def setup_db(conn):
    """Create consensus_tracking table if not exists"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consensus_tracking (
            report_date TEXT PRIMARY KEY,
            week_ending TEXT,
            
            -- Consensus data
            fmp_consensus REAL,
            fmp_actual REAL,
            fmp_previous REAL,
            consensus_source TEXT,
            
            -- Our model predictions
            model_basic_pred REAL,
            model_enhanced_pred REAL,
            model_5yr_avg REAL,
            
            -- EIA actual
            eia_actual_change REAL,
            eia_storage_level REAL,
            
            -- Market context
            ng_price_pre REAL,
            ng_price_post REAL,
            ng_return_1d REAL,
            
            -- Storage context
            storage_vs_5yr_pct REAL,
            
            -- Surprises (computed)
            consensus_surprise REAL,
            model_basic_surprise REAL,
            model_enhanced_surprise REAL,
            
            -- Metadata
            captured_at TEXT,
            capture_type TEXT
        )
    """)
    conn.commit()


# ============================================================
# DATA SOURCES
# ============================================================
def fetch_fmp_consensus(target_date=None):
    """
    Fetch consensus forecast from FMP economic calendar.
    Returns dict with {forecast, actual, previous} or None.
    """
    if not FMP_API_KEY:
        print("  WARNING: FMP_API_KEY not set, skipping FMP consensus")
        return None
    
    # Search around the target date for NG storage events
    if target_date is None:
        target_date = date.today()
    
    start = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
    
    url = (f"https://financialmodelingprep.com/stable/economic-calendar"
           f"?from={start}&to={end}&apikey={FMP_API_KEY}")
    
    print(f"  Fetching FMP economic calendar {start} to {end}...")
    data = api_get(url)
    
    if not data:
        print("  FMP returned no data")
        return None
    
    # Find natural gas storage entry
    ng_events = []
    for event in data:
        event_name = event.get("event", "").lower()
        if "natural gas" in event_name and "storage" in event_name:
            ng_events.append(event)
        elif "natural gas" in event_name and "inventories" in event_name:
            ng_events.append(event)
        elif "natural gas" in event_name and "stock" in event_name:
            ng_events.append(event)
    
    if not ng_events:
        # Broader search
        for event in data:
            event_name = event.get("event", "").lower()
            if "natural gas" in event_name:
                ng_events.append(event)
    
    if not ng_events:
        print(f"  No NG storage event found in FMP calendar for {start} to {end}")
        # Print what events ARE there for debugging
        us_events = [e for e in data if e.get("country", "").lower() in ["us", "united states", "usa"]]
        if us_events:
            print(f"  Found {len(us_events)} US events, sample names:")
            for e in us_events[:5]:
                print(f"    - {e.get('event', 'N/A')}")
        return None
    
    event = ng_events[0]
    result = {
        "forecast": None,
        "actual": None,
        "previous": None,
        "event_date": event.get("date", ""),
        "event_name": event.get("event", ""),
    }
    
    # Parse values — FMP may return strings like "-249B" or numbers
    for field in ["estimate", "forecast"]:
        val = event.get(field)
        if val is not None:
            result["forecast"] = _parse_bcf(val)
            break
    
    val = event.get("actual")
    if val is not None:
        result["actual"] = _parse_bcf(val)
    
    val = event.get("previous")
    if val is not None:
        result["previous"] = _parse_bcf(val)
    
    print(f"  FMP event: {result['event_name']}")
    print(f"  Forecast: {result['forecast']}, Actual: {result['actual']}, Previous: {result['previous']}")
    
    return result

def _parse_bcf(val):
    """Parse a BCF value from various formats"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip().upper()
    val_str = val_str.replace(",", "").replace("B", "").replace("BCF", "")
    
    try:
        return float(val_str)
    except ValueError:
        return None


def fetch_eia_latest():
    """
    Fetch the latest EIA storage report from the API.
    Returns dict with {report_date, week_ending, change_bcf, level_bcf} or None.
    """
    url = (f"https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
           f"?api_key={EIA_API_KEY}"
           f"&frequency=weekly"
           f"&data[0]=value"
           f"&facets[process][]=SWO"
           f"&facets[duoarea][]=R48"
           f"&sort[0][column]=period"
           f"&sort[0][direction]=desc"
           f"&length=2")
    
    print("  Fetching latest EIA storage data...")
    data = api_get(url)
    
    if not data or "response" not in data or "data" not in data["response"]:
        print("  EIA API returned no data")
        return None
    
    records = data["response"]["data"]
    if len(records) < 1:
        print("  No EIA records returned")
        return None
    
    latest = records[0]
    level = float(latest["value"])
    
    change = None
    if len(records) >= 2:
        prev_level = float(records[1]["value"])
        change = level - prev_level
    
    # Report date is the Friday after the period end
    week_ending = latest["period"]
    # EIA reports come out Thursday, covering the week ending Friday before
    we = datetime.strptime(week_ending, "%Y-%m-%d")
    report_date = we + timedelta(days=6)  # Following Thursday
    
    result = {
        "report_date": report_date.strftime("%Y-%m-%d"),
        "week_ending": week_ending,
        "change_bcf": change,
        "level_bcf": level,
    }
    
    print(f"  EIA: week ending {week_ending}, level={level} BCF, change={change} BCF")
    return result


def compute_model_predictions(conn, report_date):
    """
    Compute our model's predictions for the given report date.
    Uses historical data from master_weekly table.
    Returns dict with {basic_pred, enhanced_pred, avg5yr} or None.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM master_weekly 
        WHERE report_date <= ? 
        ORDER BY report_date
    """, (report_date,))
    
    rows = cursor.fetchall()
    if len(rows) < 156:  # Need 3+ years
        print("  Not enough history for model predictions")
        return None
    
    data = [dict(row) for row in rows]
    latest = data[-1]
    
    # Determine season
    month = int(latest["report_date"][5:7])
    season = "WITHDRAWAL" if month in [11, 12, 1, 2, 3] else "INJECTION"
    
    # 5-year average
    yr = int(latest["report_date"][:4])
    wk = iso_week(latest["report_date"])
    
    year_week_change = {}
    for d in data:
        if d["storage_change_bcf"] is None:
            continue
        y = int(d["report_date"][:4])
        w = iso_week(d["report_date"])
        year_week_change[(y, w)] = d["storage_change_bcf"]
    
    prior_changes = []
    for y_offset in range(1, 6):
        for wk_offset in [0, -1, 1]:
            key = (yr - y_offset, wk + wk_offset)
            if key in year_week_change:
                prior_changes.append(year_week_change[key])
                break
    
    avg5yr = mean(prior_changes) if len(prior_changes) >= 3 else None
    
    # Basic model: HDD + CDD -> storage change (same season)
    same_season = [d for d in data[:-1] 
                   if d["storage_change_bcf"] is not None
                   and d["national_hdd"] is not None
                   and d.get("national_cdd") is not None
                   and (("WITHDRAWAL" if int(d["report_date"][5:7]) in [11,12,1,2,3] else "INJECTION") == season)]
    
    if len(same_season) < 30 or latest.get("national_hdd") is None:
        return {"basic_pred": None, "enhanced_pred": None, "avg5yr": avg5yr}
    
    x1 = [d["national_hdd"] for d in same_season]
    x2 = [d["national_cdd"] or 0 for d in same_season]
    y = [d["storage_change_bcf"] for d in same_season]
    
    basic_pred = multi_reg_predict([x1, x2], y, [latest["national_hdd"], latest.get("national_cdd", 0) or 0])
    
    # Enhanced model: HDD + CDD + prior_change + storage_vs_5yr
    # Need storage_vs_5yr for training data too
    year_week_level = {}
    for d in data:
        if d.get("storage_level_bcf") is None:
            continue
        y2 = int(d["report_date"][:4])
        w2 = iso_week(d["report_date"])
        year_week_level[(y2, w2)] = d["storage_level_bcf"]
    
    # Compute storage_vs_5yr for each training row
    enhanced_train = []
    for i in range(1, len(same_season)):
        d = same_season[i]
        prev_d = same_season[i-1]
        
        y3 = int(d["report_date"][:4])
        w3 = iso_week(d["report_date"])
        
        pl = []
        for yo in range(1, 6):
            for wo in [0, -1, 1]:
                key = (y3 - yo, w3 + wo)
                if key in year_week_level:
                    pl.append(year_week_level[key])
                    break
        
        if len(pl) >= 3 and d.get("storage_level_bcf") and prev_d.get("storage_change_bcf") is not None:
            avg5_lvl = mean(pl)
            sv5 = (d["storage_level_bcf"] / avg5_lvl - 1) * 100 if avg5_lvl else None
            if sv5 is not None:
                enhanced_train.append({
                    "hdd": d["national_hdd"],
                    "cdd": d.get("national_cdd", 0) or 0,
                    "prior_change": prev_d["storage_change_bcf"],
                    "sv5": sv5,
                    "change": d["storage_change_bcf"],
                })
    
    enhanced_pred = None
    if len(enhanced_train) > 50 and len(data) >= 2:
        prev_change = data[-2].get("storage_change_bcf")
        
        # Compute current sv5
        pl_now = []
        for yo in range(1, 6):
            for wo in [0, -1, 1]:
                key = (yr - yo, wk + wo)
                if key in year_week_level:
                    pl_now.append(year_week_level[key])
                    break
        
        sv5_now = None
        if len(pl_now) >= 3 and latest.get("storage_level_bcf"):
            avg5_now = mean(pl_now)
            sv5_now = (latest["storage_level_bcf"] / avg5_now - 1) * 100 if avg5_now else None
        
        if prev_change is not None and sv5_now is not None:
            x1e = [t["hdd"] for t in enhanced_train]
            x2e = [t["cdd"] for t in enhanced_train]
            x3e = [t["prior_change"] for t in enhanced_train]
            x4e = [t["sv5"] for t in enhanced_train]
            ye = [t["change"] for t in enhanced_train]
            
            enhanced_pred = multi_reg_predict(
                [x1e, x2e, x3e, x4e], ye,
                [latest["national_hdd"], latest.get("national_cdd", 0) or 0, prev_change, sv5_now]
            )
    
    result = {
        "basic_pred": basic_pred,
        "enhanced_pred": enhanced_pred,
        "avg5yr": avg5yr,
    }
    
    basic_str = f"{basic_pred:.1f}" if basic_pred is not None else "N/A"
    enhanced_str = f"{enhanced_pred:.1f}" if enhanced_pred is not None else "N/A"
    avg5yr_str = f"{avg5yr:.1f}" if avg5yr is not None else "N/A"
    print(f"  Model predictions: basic={basic_str}, enhanced={enhanced_str}, 5yr_avg={avg5yr_str}")
    return result


def fetch_ng_price():
    """Fetch current NG price from yfinance-style source"""
    # Use FMP for NG price if available
    if FMP_API_KEY:
        url = f"https://financialmodelingprep.com/stable/quote/NGUSD?apikey={FMP_API_KEY}"
        data = api_get(url)
        if data and len(data) > 0:
            price = data[0].get("price") or data[0].get("previousClose")
            if price:
                print(f"  NG price (FMP): ${price}")
                return float(price)
    
    # Fallback: try a simple quote endpoint
    try:
        import yfinance as yf
        ng = yf.Ticker("NG=F")
        hist = ng.history(period="5d")
        if not hist.empty:
            price = hist["Close"].iloc[-1]
            print(f"  NG price (yfinance): ${price:.2f}")
            return float(price)
    except:
        pass
    
    print("  WARNING: Could not fetch NG price")
    return None


# ============================================================
# MAIN ACCUMULATION LOGIC
# ============================================================
def accumulate(backfill=False):
    """Main accumulation run"""
    print("=" * 70)
    print("NAT GAS CONSENSUS DATA ACCUMULATOR")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        sys.exit(1)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    setup_db(conn)
    
    if backfill:
        run_backfill(conn)
    else:
        run_weekly(conn)
    
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def run_weekly(conn):
    """Normal weekly run — capture this week's data"""
    
    # 1. Get latest EIA data
    print("\n--- STEP 1: EIA Actual Data ---")
    eia = fetch_eia_latest()
    
    if not eia:
        print("  Could not fetch EIA data. Will retry next run.")
        return
    
    report_date = eia["report_date"]
    
    # Check if we already have this week
    existing = conn.execute(
        "SELECT report_date FROM consensus_tracking WHERE report_date = ?",
        (report_date,)
    ).fetchone()
    
    if existing:
        print(f"\n  Already have data for {report_date}, updating...")
    
    # 2. Get FMP consensus
    print("\n--- STEP 2: FMP Consensus Forecast ---")
    target = datetime.strptime(report_date, "%Y-%m-%d").date()
    fmp = fetch_fmp_consensus(target)
    
    # 3. Get our model predictions
    print("\n--- STEP 3: Our Model Predictions ---")
    models = compute_model_predictions(conn, report_date)
    
    # 4. Get NG price
    print("\n--- STEP 4: NG Price ---")
    ng_price = fetch_ng_price()
    
    # 5. Compute storage vs 5yr
    cursor = conn.cursor()
    cursor.execute("""
        SELECT storage_level_bcf FROM master_weekly WHERE report_date = ?
    """, (report_date,))
    row = cursor.fetchone()
    storage_level = row["storage_level_bcf"] if row else eia.get("level_bcf")
    
    # Get 5yr avg level
    yr = int(report_date[:4])
    wk = iso_week(report_date)
    cursor.execute("""
        SELECT storage_level_bcf, report_date FROM master_weekly
        WHERE report_date < ?
        ORDER BY report_date
    """, (report_date,))
    all_rows = cursor.fetchall()
    
    year_week_level = {}
    for r in all_rows:
        y = int(r["report_date"][:4])
        w = iso_week(r["report_date"])
        year_week_level[(y, w)] = r["storage_level_bcf"]
    
    prior_levels = []
    for yo in range(1, 6):
        for wo in [0, -1, 1]:
            key = (yr - yo, wk + wo)
            if key in year_week_level:
                prior_levels.append(year_week_level[key])
                break
    
    sv5_pct = None
    if len(prior_levels) >= 3 and storage_level:
        avg5_lvl = mean(prior_levels)
        sv5_pct = (storage_level / avg5_lvl - 1) * 100 if avg5_lvl else None
    
    # 6. Compute surprises
    actual_change = eia.get("change_bcf")
    
    consensus_surprise = None
    if actual_change is not None and fmp and fmp.get("forecast") is not None:
        consensus_surprise = actual_change - fmp["forecast"]
    
    basic_surprise = None
    if actual_change is not None and models and models.get("basic_pred") is not None:
        basic_surprise = actual_change - models["basic_pred"]
    
    enhanced_surprise = None
    if actual_change is not None and models and models.get("enhanced_pred") is not None:
        enhanced_surprise = actual_change - models["enhanced_pred"]
    
    # 7. Insert/Update
    print("\n--- STEP 5: Saving to Database ---")
    
    conn.execute("""
        INSERT OR REPLACE INTO consensus_tracking (
            report_date, week_ending,
            fmp_consensus, fmp_actual, fmp_previous, consensus_source,
            model_basic_pred, model_enhanced_pred, model_5yr_avg,
            eia_actual_change, eia_storage_level,
            ng_price_pre, storage_vs_5yr_pct,
            consensus_surprise, model_basic_surprise, model_enhanced_surprise,
            captured_at, capture_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_date, eia["week_ending"],
        fmp["forecast"] if fmp else None,
        fmp["actual"] if fmp else None,
        fmp["previous"] if fmp else None,
        "FMP" if fmp else None,
        models["basic_pred"] if models else None,
        models["enhanced_pred"] if models else None,
        models["avg5yr"] if models else None,
        actual_change,
        eia["level_bcf"],
        ng_price,
        sv5_pct,
        consensus_surprise,
        basic_surprise,
        enhanced_surprise,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weekly_auto",
    ))
    conn.commit()
    
    # 8. Summary
    print(f"\n{'=' * 50}")
    print(f"  SAVED: {report_date}")
    print(f"  Week ending: {eia['week_ending']}")
    print(f"  EIA actual: {actual_change:.0f} BCF" if actual_change else "  EIA actual: N/A")
    print(f"  FMP consensus: {fmp['forecast']:.0f} BCF" if fmp and fmp.get('forecast') else "  FMP consensus: N/A")
    print(f"  Our basic model: {models['basic_pred']:.0f} BCF" if models and models.get('basic_pred') else "  Our basic model: N/A")
    print(f"  Our enhanced model: {models['enhanced_pred']:.0f} BCF" if models and models.get('enhanced_pred') else "  Our enhanced model: N/A")
    print(f"  5yr avg: {models['avg5yr']:.0f} BCF" if models and models.get('avg5yr') else "  5yr avg: N/A")
    print(f"  Consensus surprise: {consensus_surprise:.0f} BCF" if consensus_surprise is not None else "  Consensus surprise: N/A")
    print(f"  Storage vs 5yr: {sv5_pct:.1f}%" if sv5_pct is not None else "  Storage vs 5yr: N/A")
    print(f"  NG price: ${ng_price:.2f}" if ng_price else "  NG price: N/A")
    print(f"{'=' * 50}")
    
    # Show tracking table status
    count = conn.execute("SELECT COUNT(*) as n FROM consensus_tracking").fetchone()["n"]
    print(f"\n  Total weeks in consensus_tracking: {count}")


def run_backfill(conn):
    """Try to backfill historical consensus from FMP"""
    print("\n--- BACKFILL MODE ---")
    print("  Attempting to pull historical consensus from FMP economic calendar...")
    
    if not FMP_API_KEY:
        print("  ERROR: FMP_API_KEY required for backfill")
        return
    
    # Try pulling calendar data for past dates
    # FMP may have historical economic calendar with actual/forecast
    today = date.today()
    
    weeks_found = 0
    weeks_total = 0
    
    # Go back week by week
    for weeks_back in range(1, 53):  # Try up to 1 year
        target = today - timedelta(weeks=weeks_back)
        
        # Find the Thursday of that week
        days_since_thursday = (target.weekday() - 3) % 7
        thursday = target - timedelta(days=days_since_thursday)
        
        # Check if we already have this
        report_date_approx = thursday.strftime("%Y-%m-%d")
        existing = conn.execute(
            "SELECT report_date FROM consensus_tracking WHERE report_date BETWEEN ? AND ?",
            ((thursday - timedelta(days=3)).strftime("%Y-%m-%d"),
             (thursday + timedelta(days=3)).strftime("%Y-%m-%d"))
        ).fetchone()
        
        if existing:
            continue
        
        weeks_total += 1
        
        start = (thursday - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (thursday + timedelta(days=1)).strftime("%Y-%m-%d")
        
        url = (f"https://financialmodelingprep.com/stable/economic-calendar"
               f"?from={start}&to={end}&apikey={FMP_API_KEY}")
        
        data = api_get(url)
        if not data:
            continue
        
        for event in data:
            event_name = event.get("event", "").lower()
            if "natural gas" in event_name and ("storage" in event_name or "inventories" in event_name or "stock" in event_name):
                forecast = _parse_bcf(event.get("estimate") or event.get("forecast"))
                actual = _parse_bcf(event.get("actual"))
                previous = _parse_bcf(event.get("previous"))
                event_date = event.get("date", thursday.strftime("%Y-%m-%d"))[:10]
                
                if actual is not None:
                    surprise = (actual - forecast) if forecast is not None else None
                    
                    conn.execute("""
                        INSERT OR IGNORE INTO consensus_tracking (
                            report_date, fmp_consensus, fmp_actual, fmp_previous,
                            consensus_source, consensus_surprise,
                            captured_at, capture_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event_date, forecast, actual, previous,
                        "FMP_backfill", surprise,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "backfill",
                    ))
                    conn.commit()
                    weeks_found += 1
                    
                    f_str = f"{forecast:.0f}" if forecast is not None else "N/A"
                    prev_str = f"{previous:.0f}" if previous else "N/A"
                    print(f"  {event_date}: actual={actual:.0f}, forecast={f_str}, previous={prev_str}")
                break
    
    print(f"\n  Backfill complete: found {weeks_found} / {weeks_total} weeks checked")
    
    count = conn.execute("SELECT COUNT(*) as n FROM consensus_tracking").fetchone()["n"]
    print(f"  Total weeks in consensus_tracking: {count}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    backfill = "--backfill" in sys.argv
    accumulate(backfill=backfill)
