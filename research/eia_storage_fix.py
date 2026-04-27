#!/usr/bin/env python3
"""
EIA STORAGE FIX - Repulls Section 1 with correct API v2 parameters
and rebuilds the master weekly table.

Fixes:
  - Correct duoarea facet codes: R48, R31, R32, R33, R34, R35
  - Correct process codes: SWO (total), SSO (salt), SNO (non-salt)
  - Added data[0]=value parameter
  - Proper pagination with offset

Usage: python3 eia_storage_fix.py YOUR_EIA_KEY
  Run from: ~/Desktop/Claude_Programs/Trading_Programs/
  Expects: data/nat_gas_weather.db already exists (from collector)
"""

import sys
import os
import json
import time
import sqlite3
import urllib.request
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
if not API_KEY:
    print("Usage: python3 eia_storage_fix.py YOUR_EIA_KEY")
    sys.exit(1)

DB_PATH = "data/nat_gas_weather.db"
SLEEP = 3  # seconds between API calls
PAGE_SIZE = 5000  # EIA max per call

# Correct region mapping from diagnostic
REGIONS = {
    "L48":           {"duoarea": "R48", "process": "SWO"},
    "EAST":          {"duoarea": "R31", "process": "SWO"},
    "MIDWEST":       {"duoarea": "R32", "process": "SWO"},
    "SOUTH_CENTRAL": {"duoarea": "R33", "process": "SWO"},
    "MOUNTAIN":      {"duoarea": "R34", "process": "SWO"},
    "PACIFIC":       {"duoarea": "R35", "process": "SWO"},
    # Bonus: Salt/Non-salt for South Central
    "SC_SALT":       {"duoarea": "R33", "process": "SSO"},
    "SC_NONSALT":    {"duoarea": "R33", "process": "SNO"},
}

BASE_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data"

print("=" * 60)
print("EIA STORAGE FIX")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Database: {DB_PATH}")
print(f"Sleep between calls: {SLEEP}s")
print("=" * 60)

if not os.path.exists(DB_PATH):
    print(f"\nERROR: Database not found at {DB_PATH}")
    print("Run this from ~/Desktop/Claude_Programs/Trading_Programs/")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ============================================================
# SECTION 1: RE-PULL EIA STORAGE WITH CORRECT PARAMETERS
# ============================================================
print(f"\n{'=' * 60}")
print("SECTION 1: RE-PULLING EIA WEEKLY STORAGE (CORRECTED)")
print(f"{'=' * 60}")

# Drop and recreate the storage table
cursor.execute("DROP TABLE IF EXISTS eia_storage")
cursor.execute("""
    CREATE TABLE eia_storage (
        period TEXT,
        region TEXT,
        duoarea TEXT,
        process TEXT,
        process_name TEXT,
        series_id TEXT,
        value REAL,
        units TEXT,
        week_change REAL,
        PRIMARY KEY (period, region)
    )
""")
conn.commit()

total_stored = 0

for region_name, params in REGIONS.items():
    duoarea = params["duoarea"]
    process = params["process"]
    
    print(f"\n  Fetching region: {region_name} (duoarea={duoarea}, process={process})...")
    
    offset = 0
    region_rows = 0
    
    while True:
        url = (
            f"{BASE_URL}?"
            f"api_key={API_KEY}"
            f"&frequency=weekly"
            f"&data[0]=value"
            f"&facets[duoarea][]={duoarea}"
            f"&facets[process][]={process}"
            f"&sort[0][column]=period"
            f"&sort[0][direction]=asc"
            f"&offset={offset}"
            f"&length={PAGE_SIZE}"
        )
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    ERROR at offset {offset}: {e}")
            break
        
        response = data.get("response", {})
        rows = response.get("data", [])
        total_available = response.get("total", 0)
        
        if not rows:
            break
        
        for row in rows:
            period = row.get("period")
            value_str = row.get("value")
            
            # Skip null values
            if value_str is None or value_str == "":
                continue
            
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            
            cursor.execute("""
                INSERT OR REPLACE INTO eia_storage 
                (period, region, duoarea, process, process_name, series_id, value, units)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                period,
                region_name,
                duoarea,
                process,
                row.get("process-name", ""),
                row.get("series", ""),
                value,
                row.get("units", "BCF"),
            ))
            region_rows += 1
        
        offset += len(rows)
        
        if offset >= int(total_available):
            break
        
        print(f"    Page: offset={offset}, total={total_available}")
        time.sleep(SLEEP)
    
    total_stored += region_rows
    print(f"    Region {region_name}: {region_rows} rows stored")
    conn.commit()
    time.sleep(SLEEP)

# ============================================================
# CALCULATE WEEK-OVER-WEEK CHANGES
# ============================================================
print(f"\n  Calculating week-over-week storage changes...")

for region_name in REGIONS.keys():
    cursor.execute("""
        SELECT period, value FROM eia_storage 
        WHERE region = ? 
        ORDER BY period ASC
    """, (region_name,))
    rows = cursor.fetchall()
    
    prev_value = None
    for period, value in rows:
        if prev_value is not None and value is not None:
            change = value - prev_value
            cursor.execute("""
                UPDATE eia_storage SET week_change = ?
                WHERE period = ? AND region = ?
            """, (change, period, region_name))
        prev_value = value

conn.commit()

# Print summary
cursor.execute("SELECT region, COUNT(*), MIN(period), MAX(period) FROM eia_storage GROUP BY region ORDER BY region")
print(f"\n  EIA Storage Summary:")
print(f"  {'Region':<20s} {'Rows':>6s}  {'Start':>12s}  {'End':>12s}")
print(f"  {'-'*55}")
for row in cursor.fetchall():
    print(f"  {row[0]:<20s} {row[1]:>6d}  {row[2]:>12s}  {row[3]:>12s}")

print(f"\n  Total EIA rows: {total_stored}")

# Export to CSV
print(f"  Exporting to data/eia_storage.csv...")
cursor.execute("SELECT * FROM eia_storage ORDER BY period, region")
rows = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]
with open("data/eia_storage.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for row in rows:
        f.write(",".join(str(v) if v is not None else "" for v in row) + "\n")
print(f"  Exported {len(rows)} rows")

# ============================================================
# SECTION 2: REBUILD MASTER WEEKLY TABLE
# ============================================================
print(f"\n{'=' * 60}")
print("SECTION 2: REBUILDING MASTER WEEKLY TABLE")
print(f"{'=' * 60}")

cursor.execute("DROP TABLE IF EXISTS master_weekly")
cursor.execute("""
    CREATE TABLE master_weekly (
        report_date TEXT PRIMARY KEY,
        
        -- EIA Storage (L48 = national)
        storage_level_bcf REAL,
        storage_change_bcf REAL,
        
        -- Regional storage levels
        east_storage REAL,
        midwest_storage REAL,
        south_central_storage REAL,
        mountain_storage REAL,
        pacific_storage REAL,
        
        -- Regional changes
        east_change REAL,
        midwest_change REAL,
        south_central_change REAL,
        mountain_change REAL,
        pacific_change REAL,
        
        -- South Central salt/non-salt
        sc_salt_storage REAL,
        sc_nonsalt_storage REAL,
        
        -- NOAA Degree Days (national = population-weighted sum)
        national_hdd REAL,
        national_cdd REAL,
        
        -- Regional degree days
        east_hdd REAL,
        east_cdd REAL,
        midwest_hdd REAL,
        midwest_cdd REAL,
        south_central_hdd REAL,
        south_central_cdd REAL,
        mountain_hdd REAL,
        mountain_cdd REAL,
        pacific_hdd REAL,
        pacific_cdd REAL,
        
        -- NG Futures (Thursday close = report day)
        ng_close_report_day REAL,
        ng_close_day_before REAL,
        ng_close_day_after REAL,
        ng_close_5d_after REAL,
        ng_volume_report_day REAL,
        
        -- Derived
        ng_return_1d REAL,
        ng_return_5d REAL,
        
        -- Seasonal tag
        season TEXT
    )
""")

# Get all EIA report dates (from L48)
cursor.execute("""
    SELECT DISTINCT period FROM eia_storage 
    WHERE region = 'L48' 
    ORDER BY period
""")
report_dates = [row[0] for row in cursor.fetchall()]
print(f"  EIA report dates: {len(report_dates)} weeks ({report_dates[0]} to {report_dates[-1]})")

# Check NOAA data availability
cursor.execute("SELECT COUNT(*) FROM noaa_weekly_dd")
noaa_count = cursor.fetchone()[0]
print(f"  NOAA weekly records: {noaa_count}")

# Check NG futures data
cursor.execute("SELECT COUNT(*) FROM ng_futures")
ng_count = cursor.fetchone()[0]
print(f"  NG futures daily records: {ng_count}")

master_count = 0

for report_date in report_dates:
    # -- EIA Storage --
    storage_data = {}
    region_map = {
        "L48": ("storage_level_bcf", "storage_change_bcf"),
        "EAST": ("east_storage", "east_change"),
        "MIDWEST": ("midwest_storage", "midwest_change"),
        "SOUTH_CENTRAL": ("south_central_storage", "south_central_change"),
        "MOUNTAIN": ("mountain_storage", "mountain_change"),
        "PACIFIC": ("pacific_storage", "pacific_change"),
        "SC_SALT": ("sc_salt_storage", None),
        "SC_NONSALT": ("sc_nonsalt_storage", None),
    }
    
    for region, (level_col, change_col) in region_map.items():
        cursor.execute("""
            SELECT value, week_change FROM eia_storage
            WHERE period = ? AND region = ?
        """, (report_date, region))
        row = cursor.fetchone()
        if row:
            storage_data[level_col] = row[0]
            if change_col:
                storage_data[change_col] = row[1]
    
    # -- NOAA Degree Days --
    # Find the NOAA week that best matches this EIA report date
    # EIA reports on Thursday for the week ending Friday before
    # NOAA weeks are Sun-Sat. We want the week ending closest to the report date.
    noaa_data = {}
    
    # Try exact week match first, then +/- 1 week
    cursor.execute("""
        SELECT week_ending, region, hdd_total, cdd_total, hdd_national, cdd_national 
        FROM noaa_weekly_dd
        WHERE week_ending BETWEEN date(?, '-7 days') AND date(?, '+3 days')
    """, (report_date, report_date))
    
    noaa_rows = cursor.fetchall()
    
    # Map NOAA regions to our columns
    noaa_region_map = {
        "NATIONAL": ("national_hdd", "national_cdd"),
        "EAST": ("east_hdd", "east_cdd"),
        "MIDWEST": ("midwest_hdd", "midwest_cdd"),
        "SOUTH_CENTRAL": ("south_central_hdd", "south_central_cdd"),
        "MOUNTAIN": ("mountain_hdd", "mountain_cdd"),
        "PACIFIC": ("pacific_hdd", "pacific_cdd"),
    }
    
    # Group by region, pick closest week_ending to report_date
    from collections import defaultdict
    region_weeks = defaultdict(list)
    for week_ending, region, hdd_total, cdd_total, hdd_national, cdd_national in noaa_rows:
        region_weeks[region].append((week_ending, hdd_total, cdd_total, hdd_national, cdd_national))
    
    # National totals from any region row (they all have the same national values)
    for region, weeks in region_weeks.items():
        best = min(weeks, 
                  key=lambda x: abs((datetime.strptime(x[0], "%Y-%m-%d") - 
                                    datetime.strptime(report_date, "%Y-%m-%d")).days))
        # Set national from any row
        if best[3] is not None:
            noaa_data["national_hdd"] = best[3]
            noaa_data["national_cdd"] = best[4]
        break  # Only need one region for national
    
    for region, (hdd_col, cdd_col) in noaa_region_map.items():
        if region == "NATIONAL":
            continue  # Already handled above
        if region in region_weeks:
            best = min(region_weeks[region], 
                      key=lambda x: abs((datetime.strptime(x[0], "%Y-%m-%d") - 
                                        datetime.strptime(report_date, "%Y-%m-%d")).days))
            noaa_data[hdd_col] = best[1]  # hdd_total (regional)
            noaa_data[cdd_col] = best[2]  # cdd_total (regional)
    
    # -- NG Futures Prices --
    ng_data = {}
    
    # Report day close (or nearest trading day)
    cursor.execute("""
        SELECT date, close, volume FROM ng_futures
        WHERE date BETWEEN date(?, '-3 days') AND ?
        ORDER BY date DESC LIMIT 1
    """, (report_date, report_date))
    row = cursor.fetchone()
    if row:
        ng_data["ng_close_report_day"] = row[1]
        ng_data["ng_volume_report_day"] = row[2]
    
    # Day before
    cursor.execute("""
        SELECT close FROM ng_futures
        WHERE date < ? ORDER BY date DESC LIMIT 1
    """, (report_date,))
    row = cursor.fetchone()
    if row:
        ng_data["ng_close_day_before"] = row[0]
    
    # Day after
    cursor.execute("""
        SELECT close FROM ng_futures
        WHERE date > ? ORDER BY date ASC LIMIT 1
    """, (report_date,))
    row = cursor.fetchone()
    if row:
        ng_data["ng_close_day_after"] = row[0]
    
    # 5 trading days after
    cursor.execute("""
        SELECT close FROM ng_futures
        WHERE date > ? ORDER BY date ASC LIMIT 5
    """, (report_date,))
    rows_5d = cursor.fetchall()
    if len(rows_5d) == 5:
        ng_data["ng_close_5d_after"] = rows_5d[-1][0]
    
    # Returns
    if ng_data.get("ng_close_report_day") and ng_data.get("ng_close_day_after"):
        ng_data["ng_return_1d"] = (ng_data["ng_close_day_after"] / ng_data["ng_close_report_day"] - 1) * 100
    
    if ng_data.get("ng_close_report_day") and ng_data.get("ng_close_5d_after"):
        ng_data["ng_return_5d"] = (ng_data["ng_close_5d_after"] / ng_data["ng_close_report_day"] - 1) * 100
    
    # Season tag
    month = int(report_date[5:7])
    if month in [11, 12, 1, 2, 3]:
        season = "WITHDRAWAL"
    elif month in [4, 5, 6, 7, 8, 9, 10]:
        season = "INJECTION"
    else:
        season = "SHOULDER"
    
    # -- INSERT --
    all_data = {**storage_data, **noaa_data, **ng_data, "season": season}
    
    cols = ["report_date"] + list(all_data.keys())
    vals = [report_date] + list(all_data.values())
    placeholders = ",".join(["?"] * len(vals))
    col_str = ",".join(cols)
    
    cursor.execute(f"INSERT OR REPLACE INTO master_weekly ({col_str}) VALUES ({placeholders})", vals)
    master_count += 1

conn.commit()

# ============================================================
# SUMMARY
# ============================================================
print(f"\n  Master table built: {master_count} weeks")

# Check data completeness
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN storage_level_bcf IS NOT NULL THEN 1 ELSE 0 END) as has_storage,
        SUM(CASE WHEN national_hdd IS NOT NULL THEN 1 ELSE 0 END) as has_weather,
        SUM(CASE WHEN ng_close_report_day IS NOT NULL THEN 1 ELSE 0 END) as has_price,
        SUM(CASE WHEN storage_level_bcf IS NOT NULL 
                  AND national_hdd IS NOT NULL 
                  AND ng_close_report_day IS NOT NULL THEN 1 ELSE 0 END) as complete_rows,
        MIN(report_date) as first_date,
        MAX(report_date) as last_date
    FROM master_weekly
""")
row = cursor.fetchone()
print(f"\n  Data Completeness:")
print(f"    Total weeks:        {row[0]}")
print(f"    Has storage:        {row[1]}")
print(f"    Has weather:        {row[2]}")
print(f"    Has price:          {row[3]}")
print(f"    Complete (all 3):   {row[4]}")
print(f"    Date range:         {row[5]} to {row[6]}")

# Season breakdown
cursor.execute("""
    SELECT season, COUNT(*), 
           AVG(storage_change_bcf),
           AVG(national_hdd),
           AVG(national_cdd)
    FROM master_weekly
    WHERE storage_change_bcf IS NOT NULL
    GROUP BY season
""")
print(f"\n  Season Breakdown:")
print(f"  {'Season':<15s} {'Weeks':>6s}  {'Avg Change':>12s}  {'Avg HDD':>10s}  {'Avg CDD':>10s}")
print(f"  {'-'*60}")
for row in cursor.fetchall():
    chg = f"{row[2]:.1f}" if row[2] else "N/A"
    hdd = f"{row[3]:.1f}" if row[3] else "N/A"
    cdd = f"{row[4]:.1f}" if row[4] else "N/A"
    print(f"  {row[0]:<15s} {row[1]:>6d}  {chg:>12s}  {hdd:>10s}  {cdd:>10s}")

# Sample of most recent data
cursor.execute("""
    SELECT report_date, storage_level_bcf, storage_change_bcf, 
           national_hdd, national_cdd, ng_close_report_day, ng_return_1d
    FROM master_weekly
    ORDER BY report_date DESC LIMIT 5
""")
print(f"\n  Most Recent 5 Weeks:")
print(f"  {'Date':<12s} {'Level':>8s} {'Change':>8s} {'HDD':>8s} {'CDD':>8s} {'NG Close':>10s} {'1d Ret%':>8s}")
print(f"  {'-'*70}")
for row in cursor.fetchall():
    vals = []
    for v in row:
        if v is None:
            vals.append("N/A")
        elif isinstance(v, float):
            vals.append(f"{v:.2f}")
        else:
            vals.append(str(v))
    print(f"  {vals[0]:<12s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s} {vals[4]:>8s} {vals[5]:>10s} {vals[6]:>8s}")

# Export master to CSV
print(f"\n  Exporting master table to data/master_weekly.csv...")
cursor.execute("SELECT * FROM master_weekly ORDER BY report_date")
rows = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]
with open("data/master_weekly.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for row in rows:
        f.write(",".join(str(v) if v is not None else "" for v in row) + "\n")
print(f"  Exported {len(rows)} rows")

# ============================================================
# DATABASE SIZE
# ============================================================
conn.close()
db_size = os.path.getsize(DB_PATH) / 1024
print(f"\n  Database size: {db_size:.1f} KB")

print(f"\n{'=' * 60}")
print(f"FIX COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}")
