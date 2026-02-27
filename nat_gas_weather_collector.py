#!/usr/bin/env python3
"""
Natural Gas + Weather Data Collector (Phase 1)
Pulls EIA weekly storage, NOAA HDD/CDD degree days, and NG futures prices
into a unified SQLite database for backtesting.

Usage:
    python3 nat_gas_weather_collector.py <EIA_API_KEY>

Data Sources:
    - EIA API v2: Weekly natural gas storage (1993-present)
    - NOAA CPC FTP: Daily population-weighted HDD/CDD by state/region (1981-present)
    - yfinance: NG=F Henry Hub front-month futures

Output:
    - data/nat_gas_weather.db (SQLite)
    - data/master_weekly.csv (merged dataset)
    - data/eia_storage.csv
    - data/noaa_degree_days.csv
    - data/ng_futures_prices.csv
"""

import sys
import os
import time
import json
import sqlite3
import requests
import io
import re
from datetime import datetime, timedelta
from collections import defaultdict

# === CONFIGURATION ===
SLEEP_BETWEEN_CALLS = 3  # seconds between API calls
EIA_BASE_URL = "https://api.eia.gov/v2"
NOAA_FTP_BASE = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "nat_gas_weather.db")

# EIA storage regions - Lower 48 total + 5 regions
EIA_STORAGE_SERIES = {
    "L48": "NUS",   # Lower 48 total
    "EAST": "NE",   # East region (old: renamed in some contexts)
    "MIDWEST": "NMW", # Midwest
    "MOUNTAIN": "NMT", # Mountain
    "PACIFIC": "NPC",  # Pacific
    "SOUTH_CENTRAL": "NSC",  # South Central
}

# NOAA CPC regions that map to EIA storage regions (approximate)
# We'll pull all states and aggregate ourselves
NOAA_REGIONS = {
    "EAST": ["CT", "DE", "DC", "ME", "MD", "MA", "NH", "NJ", "NY", "PA", "RI", "VT", "VA", "WV"],
    "MIDWEST": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "SOUTH_CENTRAL": ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "OK", "SC", "TN", "TX"],
    "MOUNTAIN": ["CO", "ID", "MT", "NM", "UT", "WY"],
    "PACIFIC": ["AZ", "CA", "NV", "OR", "WA"],
}

# State name to abbreviation mapping for NOAA data parsing
STATE_ABBREV = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRCT COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME",
    "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}


def setup_database():
    """Create SQLite database and tables."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # EIA weekly storage
    c.execute("""
        CREATE TABLE IF NOT EXISTS eia_storage (
            report_date TEXT,
            region TEXT,
            value_bcf REAL,
            net_change_bcf REAL,
            PRIMARY KEY (report_date, region)
        )
    """)

    # NOAA daily degree days by state
    c.execute("""
        CREATE TABLE IF NOT EXISTS noaa_daily_dd (
            date TEXT,
            state TEXT,
            hdd REAL,
            cdd REAL,
            PRIMARY KEY (date, state)
        )
    """)

    # NOAA weekly aggregated degree days by region
    c.execute("""
        CREATE TABLE IF NOT EXISTS noaa_weekly_dd (
            week_ending TEXT,
            region TEXT,
            hdd_total REAL,
            cdd_total REAL,
            hdd_national REAL,
            cdd_national REAL,
            PRIMARY KEY (week_ending, region)
        )
    """)

    # NG futures daily prices
    c.execute("""
        CREATE TABLE IF NOT EXISTS ng_futures (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            adj_close REAL
        )
    """)

    # Master weekly merged table
    c.execute("""
        CREATE TABLE IF NOT EXISTS master_weekly (
            week_ending TEXT PRIMARY KEY,
            storage_level_bcf REAL,
            storage_change_bcf REAL,
            hdd_national REAL,
            cdd_national REAL,
            hdd_east REAL,
            hdd_midwest REAL,
            hdd_south_central REAL,
            hdd_mountain REAL,
            hdd_pacific REAL,
            cdd_east REAL,
            cdd_midwest REAL,
            cdd_south_central REAL,
            cdd_mountain REAL,
            cdd_pacific REAL,
            ng_close_before REAL,
            ng_close_after REAL,
            ng_return_1d REAL,
            ng_return_5d REAL,
            season TEXT
        )
    """)

    conn.commit()
    return conn


# =========================================================
# SECTION 1: EIA STORAGE DATA
# =========================================================

def fetch_eia_storage(api_key, conn):
    """Pull weekly natural gas storage from EIA API v2."""
    print("\n" + "="*60)
    print("SECTION 1: EIA WEEKLY NATURAL GAS STORAGE")
    print("="*60)

    c = conn.cursor()
    total_rows = 0

    for region_name, region_code in EIA_STORAGE_SERIES.items():
        print(f"\n  Fetching region: {region_name} (code: {region_code})...")

        offset = 0
        region_rows = 0

        while True:
            url = (
                f"{EIA_BASE_URL}/natural-gas/stor/wkly/data/"
                f"?api_key={api_key}"
                f"&frequency=weekly"
                f"&data[0]=value"
                f"&facets[process][]=SAY"
                f"&facets[duoarea][]={region_code}"
                f"&sort[0][column]=period"
                f"&sort[0][direction]=asc"
                f"&offset={offset}"
                f"&length=5000"
            )

            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    ERROR fetching {region_name} offset {offset}: {e}")
                break

            if "response" not in data or "data" not in data["response"]:
                print(f"    No data in response for {region_name} offset {offset}")
                # Print what we got for debugging
                if "error" in data:
                    print(f"    API Error: {data['error']}")
                break

            rows = data["response"]["data"]
            if not rows:
                break

            for row in rows:
                period = row.get("period", "")
                value = row.get("value")
                if period and value is not None:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue

                    c.execute("""
                        INSERT OR REPLACE INTO eia_storage (report_date, region, value_bcf, net_change_bcf)
                        VALUES (?, ?, ?, NULL)
                    """, (period, region_name, value))
                    region_rows += 1

            print(f"    Fetched {len(rows)} rows (offset {offset})")
            offset += 5000

            if len(rows) < 5000:
                break

            time.sleep(SLEEP_BETWEEN_CALLS)

        total_rows += region_rows
        print(f"    Region {region_name}: {region_rows} total rows stored")
        time.sleep(SLEEP_BETWEEN_CALLS)

    conn.commit()

    # Calculate net changes (week over week) for each region
    print("\n  Calculating week-over-week storage changes...")
    for region_name in EIA_STORAGE_SERIES:
        c.execute("""
            SELECT report_date, value_bcf FROM eia_storage
            WHERE region = ? ORDER BY report_date
        """, (region_name,))
        rows = c.fetchall()

        for i in range(1, len(rows)):
            change = rows[i][1] - rows[i-1][1]
            c.execute("""
                UPDATE eia_storage SET net_change_bcf = ?
                WHERE report_date = ? AND region = ?
            """, (change, rows[i][0], region_name))

    conn.commit()

    # Summary
    c.execute("SELECT COUNT(*) FROM eia_storage")
    total = c.fetchone()[0]
    c.execute("SELECT MIN(report_date), MAX(report_date) FROM eia_storage WHERE region='L48'")
    date_range = c.fetchone()
    print(f"\n  EIA Storage Complete: {total} total rows")
    print(f"  Date range (L48): {date_range[0]} to {date_range[1]}")

    # Export CSV
    c.execute("SELECT * FROM eia_storage ORDER BY report_date, region")
    rows = c.fetchall()
    csv_path = os.path.join(DATA_DIR, "eia_storage.csv")
    with open(csv_path, 'w') as f:
        f.write("report_date,region,value_bcf,net_change_bcf\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3] if row[3] is not None else ''}\n")
    print(f"  Exported to {csv_path}")


# =========================================================
# SECTION 2: NOAA DEGREE DAY DATA
# =========================================================

def parse_noaa_daily_file(text, year):
    """Parse a NOAA CPC daily degree day file (StatesCONUS.txt format).
    
    Format is fixed-width: each line has state abbreviation and daily values.
    Files contain daily HDD and CDD by state for the entire year.
    """
    records = []
    lines = text.strip().split('\n')

    # The daily data files have a specific format:
    # First few lines are headers, then data rows
    # Each row: STATE  DD values for each day
    # We need to identify the date range from the file header

    return records


def parse_noaa_weekly_summary(text, end_date):
    """Parse NOAA CPC weekly summary file (population-weighted state averages).
    
    Returns dict of {state_abbrev: weekly_total_hdd_or_cdd}
    """
    records = {}
    lines = text.strip().split('\n')

    for line in lines:
        # Try to match state lines - they start with a state name
        # Format: STATE_NAME  WEEK_TOTAL  DEV_FROM_NORM  DEV_FROM_LY  CUM  CUM_DEV  ...
        parts = line.split()
        if not parts:
            continue

        # Try to match state name (may be two words like NEW YORK)
        state_name = ""
        val_start_idx = 0

        # Check single word state
        if parts[0] in STATE_ABBREV:
            state_name = parts[0]
            val_start_idx = 1
        # Check two word state
        elif len(parts) > 1 and f"{parts[0]} {parts[1]}" in STATE_ABBREV:
            state_name = f"{parts[0]} {parts[1]}"
            val_start_idx = 2
        # Check DISTRCT COLUMBIA
        elif len(parts) > 1 and parts[0] == "DISTRCT":
            state_name = "DISTRCT COLUMBIA"
            val_start_idx = 2
        else:
            continue

        abbrev = STATE_ABBREV.get(state_name)
        if not abbrev:
            continue

        # First numeric value after state name is the weekly total
        remaining = parts[val_start_idx:]
        if remaining:
            try:
                weekly_val = int(remaining[0])
                if weekly_val == -999 or weekly_val == -9999:
                    weekly_val = 0
                records[abbrev] = weekly_val
            except (ValueError, IndexError):
                continue

    return records


def fetch_noaa_degree_days_via_legacy(conn):
    """Fetch NOAA weekly HDD/CDD from legacy weekly summary files on CPC FTP.
    
    URL pattern:
        https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/legacy_files/
        heating/statesCONUS/{year}/weekly-{YYYYMMDD}.txt
        cooling/statesCONUS/{year}/weekly-{YYYYMMDD}.txt
    
    These files have weekly population-weighted HDD/CDD by state.
    Available from ~2005 onward in this format.
    """
    print("\n" + "="*60)
    print("SECTION 2: NOAA DEGREE DAY DATA (LEGACY WEEKLY FILES)")
    print("="*60)

    c = conn.cursor()
    total_weeks = 0
    errors = 0

    # Try years from 2005 to current
    current_year = datetime.now().year
    start_year = 2005

    for year in range(start_year, current_year + 1):
        print(f"\n  Processing year {year}...")
        year_weeks = 0

        # Generate weekly dates (every Saturday, which is NOAA's week-ending day)
        # Start from first Saturday of the year
        jan1 = datetime(year, 1, 1)
        # Find first Saturday
        days_until_sat = (5 - jan1.weekday()) % 7
        if days_until_sat == 0:
            days_until_sat = 7
        first_sat = jan1 + timedelta(days=days_until_sat)

        # But NOAA may not use Saturday consistently - they use the last day of the collection period
        # Let's try every 7 days from early January through December
        current_date = first_sat
        if current_date.month != 1 or current_date.day > 10:
            current_date = jan1 + timedelta(days=(5 - jan1.weekday()) % 7)

        while current_date.year == year:
            date_str = current_date.strftime("%Y%m%d")

            # Fetch HDD
            hdd_url = f"https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/legacy_files/heating/statesCONUS/{year}/weekly-{date_str}.txt"
            cdd_url = f"https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/legacy_files/cooling/statesCONUS/{year}/weekly-{date_str}.txt"

            hdd_data = {}
            cdd_data = {}

            try:
                resp = requests.get(hdd_url, timeout=15)
                if resp.status_code == 200:
                    hdd_data = parse_noaa_weekly_summary(resp.text, date_str)
                time.sleep(1)  # Short sleep between HDD and CDD
            except Exception as e:
                pass  # File may not exist for this date

            try:
                resp = requests.get(cdd_url, timeout=15)
                if resp.status_code == 200:
                    cdd_data = parse_noaa_weekly_summary(resp.text, date_str)
                time.sleep(1)
            except Exception as e:
                pass

            if hdd_data or cdd_data:
                iso_date = current_date.strftime("%Y-%m-%d")
                all_states = set(list(hdd_data.keys()) + list(cdd_data.keys()))

                for state in all_states:
                    hdd_val = hdd_data.get(state, 0)
                    cdd_val = cdd_data.get(state, 0)

                    c.execute("""
                        INSERT OR REPLACE INTO noaa_daily_dd (date, state, hdd, cdd)
                        VALUES (?, ?, ?, ?)
                    """, (iso_date, state, hdd_val, cdd_val))

                year_weeks += 1

                if year_weeks % 10 == 0:
                    print(f"    {year_weeks} weeks processed ({date_str})...")

            else:
                errors += 1

            current_date += timedelta(days=7)
            time.sleep(SLEEP_BETWEEN_CALLS)

        conn.commit()
        total_weeks += year_weeks
        print(f"    Year {year}: {year_weeks} weeks collected")

    print(f"\n  NOAA Legacy Weekly Complete: {total_weeks} total weeks, {errors} missing")

    # Now aggregate by region
    print("\n  Aggregating by EIA storage region...")
    aggregate_noaa_by_region(conn)


def fetch_noaa_degree_days_via_daily(conn):
    """Fetch NOAA daily degree day data from CPC FTP.
    
    URL pattern:
        https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{year}/
        Files: StatesCONUS.Heating.txt and StatesCONUS.Cooling.txt
    
    Daily data available from 1981 onward.
    """
    print("\n" + "="*60)
    print("SECTION 2B: NOAA DEGREE DAY DATA (DAILY FILES)")
    print("="*60)

    c = conn.cursor()
    total_days = 0
    current_year = datetime.now().year

    # Daily files go back to 1981 - but for nat gas storage (1993+), start at 1993
    start_year = 1993

    for year in range(start_year, current_year + 1):
        print(f"\n  Processing year {year}...")

        hdd_url = f"{NOAA_FTP_BASE}/{year}/StatesCONUS.Heating.txt"
        cdd_url = f"{NOAA_FTP_BASE}/{year}/StatesCONUS.Cooling.txt"

        hdd_text = None
        cdd_text = None

        try:
            resp = requests.get(hdd_url, timeout=30)
            if resp.status_code == 200:
                hdd_text = resp.text
                print(f"    HDD file: {len(resp.text)} bytes")
            else:
                print(f"    HDD file not found ({resp.status_code})")
        except Exception as e:
            print(f"    HDD fetch error: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        try:
            resp = requests.get(cdd_url, timeout=30)
            if resp.status_code == 200:
                cdd_text = resp.text
                print(f"    CDD file: {len(resp.text)} bytes")
            else:
                print(f"    CDD file not found ({resp.status_code})")
        except Exception as e:
            print(f"    CDD fetch error: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # Parse daily files
        # Format: Each file has header rows, then state rows
        # Each state row: STATE_ABBREV  val1 val2 val3 ... (one per day of year)
        year_days = 0

        if hdd_text:
            hdd_by_state = parse_daily_dd_file(hdd_text, year, "HDD")
            if hdd_by_state:
                for state, daily_vals in hdd_by_state.items():
                    for date_str, val in daily_vals.items():
                        c.execute("""
                            INSERT OR IGNORE INTO noaa_daily_dd (date, state, hdd, cdd)
                            VALUES (?, ?, ?, 0)
                        """, (date_str, state, val))
                        year_days += 1

        if cdd_text:
            cdd_by_state = parse_daily_dd_file(cdd_text, year, "CDD")
            if cdd_by_state:
                for state, daily_vals in cdd_by_state.items():
                    for date_str, val in daily_vals.items():
                        # Update existing row or insert
                        c.execute("""
                            UPDATE noaa_daily_dd SET cdd = ? WHERE date = ? AND state = ?
                        """, (val, date_str, state))
                        if c.rowcount == 0:
                            c.execute("""
                                INSERT OR IGNORE INTO noaa_daily_dd (date, state, hdd, cdd)
                                VALUES (?, ?, 0, ?)
                            """, (date_str, state, val))

        conn.commit()
        total_days += year_days
        print(f"    Year {year}: ~{year_days} state-day records stored")

    print(f"\n  NOAA Daily Complete: ~{total_days} total state-day records")

    # Aggregate to weekly by region
    print("\n  Aggregating daily data to weekly by EIA region...")
    aggregate_noaa_by_region(conn)


def parse_daily_dd_file(text, year, dd_type):
    """Parse NOAA CPC daily degree day file.
    
    These files have a specific fixed-width format:
    - Header lines
    - Then one row per state, with daily values across columns
    - State abbreviation in first column, then 365/366 daily values
    """
    result = {}
    lines = text.strip().split('\n')

    # Find the data section - look for lines starting with state abbreviations
    for line in lines:
        parts = line.split('|')
        if len(parts) < 2:
            # Try space-delimited
            parts = line.split()
            if len(parts) < 10:
                continue

            # Check if first element is a state abbreviation (2 chars, uppercase)
            state = parts[0].strip()
            if len(state) != 2 or not state.isalpha() or state not in STATE_ABBREV.values():
                continue

            # Remaining values are daily degree days
            daily_vals = {}
            day_num = 1
            for val_str in parts[1:]:
                try:
                    val = float(val_str.strip())
                    if val < 0:
                        val = 0  # -9999 or similar missing indicator
                    # Convert day number to date
                    try:
                        date = datetime(year, 1, 1) + timedelta(days=day_num - 1)
                        date_str = date.strftime("%Y-%m-%d")
                        daily_vals[date_str] = val
                    except ValueError:
                        pass  # Day beyond year end
                    day_num += 1
                except ValueError:
                    day_num += 1
                    continue

            if daily_vals:
                result[state] = daily_vals

        else:
            # Pipe-delimited format
            state = parts[0].strip()
            if len(state) != 2 or not state.isalpha() or state not in STATE_ABBREV.values():
                continue

            daily_vals = {}
            day_num = 1
            for val_str in parts[1:]:
                try:
                    val = float(val_str.strip())
                    if val < 0:
                        val = 0
                    try:
                        date = datetime(year, 1, 1) + timedelta(days=day_num - 1)
                        date_str = date.strftime("%Y-%m-%d")
                        daily_vals[date_str] = val
                    except ValueError:
                        pass
                    day_num += 1
                except ValueError:
                    day_num += 1
                    continue

            if daily_vals:
                result[state] = daily_vals

    return result


def aggregate_noaa_by_region(conn):
    """Aggregate daily state-level degree days into weekly regional totals.
    
    Aligns to EIA report weeks (Friday endings).
    """
    c = conn.cursor()

    # Get all dates
    c.execute("SELECT DISTINCT date FROM noaa_daily_dd ORDER BY date")
    all_dates = [row[0] for row in c.fetchall()]

    if not all_dates:
        print("    No daily data to aggregate!")
        return

    print(f"    Daily data range: {all_dates[0]} to {all_dates[-1]}")
    print(f"    Total unique dates: {len(all_dates)}")

    # Group dates into weeks ending on Friday (EIA report week)
    # Each EIA report covers the week ending the previous Friday
    weeks = defaultdict(list)
    for date_str in all_dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Find the Friday this date belongs to (current or next Friday)
        days_until_friday = (4 - dt.weekday()) % 7
        if days_until_friday == 0 and dt.weekday() == 4:
            friday = dt  # It IS Friday
        else:
            friday = dt + timedelta(days=days_until_friday)
        week_key = friday.strftime("%Y-%m-%d")
        weeks[week_key].append(date_str)

    print(f"    Aggregating {len(weeks)} weeks...")

    week_count = 0
    for week_ending, dates in sorted(weeks.items()):
        # Get all state data for this week's dates
        placeholders = ','.join(['?' for _ in dates])
        c.execute(f"""
            SELECT state, SUM(hdd), SUM(cdd) FROM noaa_daily_dd
            WHERE date IN ({placeholders})
            GROUP BY state
        """, dates)

        state_data = {}
        for row in c.fetchall():
            state_data[row[0]] = {"hdd": row[1] or 0, "cdd": row[2] or 0}

        if not state_data:
            continue

        # Calculate national totals (simple sum across all states)
        national_hdd = sum(v["hdd"] for v in state_data.values())
        national_cdd = sum(v["cdd"] for v in state_data.values())

        # Calculate regional totals
        for region_name, states in NOAA_REGIONS.items():
            region_hdd = sum(state_data.get(st, {}).get("hdd", 0) for st in states)
            region_cdd = sum(state_data.get(st, {}).get("cdd", 0) for st in states)

            c.execute("""
                INSERT OR REPLACE INTO noaa_weekly_dd
                (week_ending, region, hdd_total, cdd_total, hdd_national, cdd_national)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (week_ending, region_name, region_hdd, region_cdd, national_hdd, national_cdd))

        # Also store national as its own "region"
        c.execute("""
            INSERT OR REPLACE INTO noaa_weekly_dd
            (week_ending, region, hdd_total, cdd_total, hdd_national, cdd_national)
            VALUES (?, 'NATIONAL', ?, ?, ?, ?)
        """, (week_ending, national_hdd, national_cdd, national_hdd, national_cdd))

        week_count += 1

    conn.commit()
    print(f"    Regional aggregation complete: {week_count} weeks")

    # Export CSV
    c.execute("SELECT * FROM noaa_weekly_dd ORDER BY week_ending, region")
    rows = c.fetchall()
    csv_path = os.path.join(DATA_DIR, "noaa_degree_days.csv")
    with open(csv_path, 'w') as f:
        f.write("week_ending,region,hdd_total,cdd_total,hdd_national,cdd_national\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
    print(f"    Exported to {csv_path}")


# =========================================================
# SECTION 3: NATURAL GAS FUTURES PRICES
# =========================================================

def fetch_ng_futures(conn):
    """Pull NG=F (Henry Hub front-month) from yfinance."""
    print("\n" + "="*60)
    print("SECTION 3: NATURAL GAS FUTURES PRICES (yfinance)")
    print("="*60)

    try:
        import yfinance as yf
    except ImportError:
        print("  Installing yfinance...")
        os.system("pip install yfinance --break-system-packages -q")
        import yfinance as yf

    c = conn.cursor()

    ticker = yf.Ticker("NG=F")
    print("  Fetching NG=F max history...")

    try:
        df = ticker.history(period="max")
    except Exception as e:
        print(f"  Error fetching NG=F: {e}")
        print("  Trying with start date...")
        df = ticker.history(start="1993-01-01")

    if df.empty:
        print("  WARNING: No data returned for NG=F!")
        return

    print(f"  Got {len(df)} daily records")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    rows_stored = 0
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        c.execute("""
            INSERT OR REPLACE INTO ng_futures (date, open, high, low, close, volume, adj_close)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (date_str, row.get('Open'), row.get('High'), row.get('Low'),
              row.get('Close'), row.get('Volume'), row.get('Close')))
        rows_stored += 1

    conn.commit()
    print(f"  Stored {rows_stored} daily price records")

    # Also fetch UNG (ETF) for supplementary data
    print("\n  Fetching UNG (Natural Gas ETF) for supplementary data...")
    try:
        ung = yf.Ticker("UNG")
        df_ung = ung.history(period="max")
        if not df_ung.empty:
            print(f"  UNG: {len(df_ung)} records ({df_ung.index[0].strftime('%Y-%m-%d')} to {df_ung.index[-1].strftime('%Y-%m-%d')})")
            # Store in separate table
            c.execute("""
                CREATE TABLE IF NOT EXISTS ung_etf (
                    date TEXT PRIMARY KEY, open REAL, high REAL, low REAL,
                    close REAL, volume REAL
                )
            """)
            for idx, row in df_ung.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                c.execute("INSERT OR REPLACE INTO ung_etf VALUES (?,?,?,?,?,?)",
                          (date_str, row.get('Open'), row.get('High'), row.get('Low'),
                           row.get('Close'), row.get('Volume')))
            conn.commit()
            print(f"  UNG stored: {len(df_ung)} records")
    except Exception as e:
        print(f"  UNG fetch failed (non-critical): {e}")

    # Export CSV
    c.execute("SELECT * FROM ng_futures ORDER BY date")
    rows = c.fetchall()
    csv_path = os.path.join(DATA_DIR, "ng_futures_prices.csv")
    with open(csv_path, 'w') as f:
        f.write("date,open,high,low,close,volume,adj_close\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
    print(f"  Exported to {csv_path}")


# =========================================================
# SECTION 4: MERGE INTO MASTER WEEKLY DATASET
# =========================================================

def build_master_weekly(conn):
    """Merge EIA storage, NOAA degree days, and NG prices into master weekly table."""
    print("\n" + "="*60)
    print("SECTION 4: BUILDING MASTER WEEKLY DATASET")
    print("="*60)

    c = conn.cursor()

    # Get all EIA report dates (these define our weeks)
    c.execute("""
        SELECT report_date, value_bcf, net_change_bcf FROM eia_storage
        WHERE region = 'L48'
        ORDER BY report_date
    """)
    eia_rows = c.fetchall()
    print(f"  EIA L48 storage weeks: {len(eia_rows)}")

    if not eia_rows:
        print("  No EIA data — cannot build master table!")
        return

    master_count = 0
    for report_date, storage_level, storage_change in eia_rows:
        # Determine season
        month = int(report_date[5:7])
        if month in [11, 12, 1, 2, 3]:
            season = "WITHDRAWAL"
        elif month in [4, 5, 6, 7, 8, 9, 10]:
            season = "INJECTION"
        else:
            season = "SHOULDER"

        # Get NOAA data for this week
        # EIA reports on Thursday for the week ending Friday before
        # Find closest NOAA week
        c.execute("""
            SELECT region, hdd_total, cdd_total, hdd_national, cdd_national
            FROM noaa_weekly_dd
            WHERE week_ending BETWEEN date(?, '-7 days') AND date(?, '+7 days')
            ORDER BY ABS(julianday(week_ending) - julianday(?))
        """, (report_date, report_date, report_date))

        noaa_rows = c.fetchall()
        noaa_by_region = {}
        hdd_national = None
        cdd_national = None
        for nr in noaa_rows:
            noaa_by_region[nr[0]] = {"hdd": nr[1], "cdd": nr[2]}
            if nr[3] is not None:
                hdd_national = nr[3]
            if nr[4] is not None:
                cdd_national = nr[4]

        # Get NG futures price BEFORE report (Wednesday close) and AFTER (Thursday close)
        # EIA reports Thursday at 10:30 ET
        c.execute("""
            SELECT date, close FROM ng_futures
            WHERE date <= ? ORDER BY date DESC LIMIT 5
        """, (report_date,))
        pre_prices = c.fetchall()

        c.execute("""
            SELECT date, close FROM ng_futures
            WHERE date >= ? ORDER BY date ASC LIMIT 6
        """, (report_date,))
        post_prices = c.fetchall()

        ng_close_before = pre_prices[1][1] if len(pre_prices) > 1 else None  # Day before report
        ng_close_after = post_prices[0][1] if post_prices else None  # Report day close

        # Calculate returns
        ng_return_1d = None
        ng_return_5d = None
        if ng_close_before and ng_close_after and ng_close_before > 0:
            ng_return_1d = (ng_close_after - ng_close_before) / ng_close_before
        if ng_close_before and len(post_prices) >= 5 and ng_close_before > 0:
            ng_return_5d = (post_prices[4][1] - ng_close_before) / ng_close_before

        c.execute("""
            INSERT OR REPLACE INTO master_weekly
            (week_ending, storage_level_bcf, storage_change_bcf,
             hdd_national, cdd_national,
             hdd_east, hdd_midwest, hdd_south_central, hdd_mountain, hdd_pacific,
             cdd_east, cdd_midwest, cdd_south_central, cdd_mountain, cdd_pacific,
             ng_close_before, ng_close_after, ng_return_1d, ng_return_5d, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_date, storage_level, storage_change,
            hdd_national, cdd_national,
            noaa_by_region.get("EAST", {}).get("hdd"),
            noaa_by_region.get("MIDWEST", {}).get("hdd"),
            noaa_by_region.get("SOUTH_CENTRAL", {}).get("hdd"),
            noaa_by_region.get("MOUNTAIN", {}).get("hdd"),
            noaa_by_region.get("PACIFIC", {}).get("hdd"),
            noaa_by_region.get("EAST", {}).get("cdd"),
            noaa_by_region.get("MIDWEST", {}).get("cdd"),
            noaa_by_region.get("SOUTH_CENTRAL", {}).get("cdd"),
            noaa_by_region.get("MOUNTAIN", {}).get("cdd"),
            noaa_by_region.get("PACIFIC", {}).get("cdd"),
            ng_close_before, ng_close_after,
            ng_return_1d, ng_return_5d,
            season
        ))
        master_count += 1

    conn.commit()

    # Summary stats
    c.execute("SELECT COUNT(*) FROM master_weekly")
    total = c.fetchone()[0]
    c.execute("SELECT MIN(week_ending), MAX(week_ending) FROM master_weekly")
    dr = c.fetchone()
    c.execute("SELECT COUNT(*) FROM master_weekly WHERE hdd_national IS NOT NULL")
    with_weather = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM master_weekly WHERE ng_close_before IS NOT NULL")
    with_prices = c.fetchone()[0]

    print(f"\n  Master Weekly Dataset Built:")
    print(f"    Total weeks: {total}")
    print(f"    Date range: {dr[0]} to {dr[1]}")
    print(f"    Weeks with weather data: {with_weather}")
    print(f"    Weeks with NG price data: {with_prices}")
    print(f"    Weeks with BOTH: ~{min(with_weather, with_prices)}")

    # Season breakdown
    c.execute("SELECT season, COUNT(*) FROM master_weekly GROUP BY season")
    for row in c.fetchall():
        print(f"    {row[0]}: {row[1]} weeks")

    # Export master CSV
    c.execute("SELECT * FROM master_weekly ORDER BY week_ending")
    rows = c.fetchall()
    cols = [desc[0] for desc in c.description]
    csv_path = os.path.join(DATA_DIR, "master_weekly.csv")
    with open(csv_path, 'w') as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(v) if v is not None else "" for v in row) + "\n")
    print(f"  Exported to {csv_path}")


# =========================================================
# SECTION 5: DATABASE SUMMARY
# =========================================================

def print_db_summary(conn):
    """Print final database summary."""
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)

    c = conn.cursor()

    tables = ["eia_storage", "noaa_daily_dd", "noaa_weekly_dd", "ng_futures", "master_weekly"]
    for table in tables:
        try:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            count = c.fetchone()[0]
            print(f"  {table}: {count:,} rows")
        except:
            print(f"  {table}: NOT CREATED")

    # Check for UNG
    try:
        c.execute("SELECT COUNT(*) FROM ung_etf")
        count = c.fetchone()[0]
        print(f"  ung_etf: {count:,} rows")
    except:
        pass

    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\n  Database size: {db_size:.1f} MB")
    print(f"  Location: {os.path.abspath(DB_PATH)}")

    # List all output files
    print(f"\n  Output files:")
    for f in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"    {f}: {size:.1f} KB")


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 nat_gas_weather_collector.py <EIA_API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]
    print("="*60)
    print("NATURAL GAS + WEATHER DATA COLLECTOR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sleep between calls: {SLEEP_BETWEEN_CALLS}s")
    print("="*60)

    conn = setup_database()

    try:
        # Section 1: EIA Storage
        fetch_eia_storage(api_key, conn)

        # Section 2: NOAA Degree Days
        # Try daily files first (more granular, goes back further)
        fetch_noaa_degree_days_via_daily(conn)

        # If daily files didn't work well, fall back to legacy weekly
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM noaa_daily_dd")
        dd_count = c.fetchone()[0]
        if dd_count < 1000:
            print("\n  Daily files yielded few records — trying legacy weekly files...")
            fetch_noaa_degree_days_via_legacy(conn)

        # Section 3: NG Futures Prices
        fetch_ng_futures(conn)

        # Section 4: Build Master Weekly
        build_master_weekly(conn)

        # Section 5: Summary
        print_db_summary(conn)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        conn.commit()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        conn.commit()
    finally:
        conn.close()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
