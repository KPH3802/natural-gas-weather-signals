#!/usr/bin/env python3
"""
EIA API v2 Diagnostic - Natural Gas Storage
Probes the API to find correct routes, facets, and data fields
for weekly natural gas storage data.

Usage: python3 eia_diagnostic.py YOUR_EIA_KEY
"""

import sys
import json
import time
import urllib.request

API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
if not API_KEY:
    print("Usage: python3 eia_diagnostic.py YOUR_EIA_KEY")
    sys.exit(1)

BASE = "https://api.eia.gov/v2"

def fetch_json(url):
    """Fetch JSON from EIA API"""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def probe(label, path):
    """Probe a route and print what we find"""
    print(f"\n{'='*60}")
    print(f"PROBE: {label}")
    print(f"URL: {BASE}/{path}?api_key=***")
    print(f"{'='*60}")
    
    url = f"{BASE}/{path}?api_key={API_KEY}"
    data = fetch_json(url)
    
    if not data:
        print("  No response")
        return data
    
    resp = data.get("response", {})
    
    # Print routes (child datasets)
    routes = resp.get("routes", [])
    if routes:
        print(f"\n  CHILD ROUTES ({len(routes)}):")
        for r in routes:
            print(f"    {r.get('id', '?'):30s} - {r.get('name', '?')}")
    
    # Print frequency options
    freq = resp.get("frequency", [])
    if freq:
        print(f"\n  FREQUENCIES:")
        for f in freq:
            if isinstance(f, dict):
                print(f"    {f.get('id', '?'):15s} - {f.get('description', '?')}")
            else:
                print(f"    {f}")
    
    # Print facets
    facets = resp.get("facets", [])
    if facets:
        print(f"\n  FACETS:")
        for f in facets:
            if isinstance(f, dict):
                print(f"    {f.get('id', '?'):15s} - {f.get('description', '?')}")
            else:
                print(f"    {f}")
    
    # Print data columns
    data_cols = resp.get("data", {})
    if data_cols and isinstance(data_cols, dict):
        print(f"\n  DATA COLUMNS:")
        for k, v in data_cols.items():
            if isinstance(v, dict):
                print(f"    {k:20s} - units: {v.get('units', '?')}, agg: {v.get('aggregation-method', '?')}")
            else:
                print(f"    {k}: {v}")
    
    # Print start/end period
    start = resp.get("startPeriod")
    end = resp.get("endPeriod")
    if start or end:
        print(f"\n  DATE RANGE: {start} to {end}")
    
    # Print description
    desc = resp.get("description")
    if desc:
        print(f"\n  DESCRIPTION: {desc[:200]}")
    
    time.sleep(1)
    return data


def probe_facet_values(route, facet_id, label):
    """Get all valid values for a facet"""
    print(f"\n{'='*60}")
    print(f"FACET VALUES: {label} ({facet_id})")
    print(f"{'='*60}")
    
    url = f"{BASE}/{route}/facet/{facet_id}?api_key={API_KEY}"
    data = fetch_json(url)
    
    if not data:
        print("  No response")
        return
    
    resp = data.get("response", {})
    facets = resp.get("facets", [])
    
    print(f"  Total values: {resp.get('totalFacets', '?')}")
    for f in facets[:30]:  # Limit to 30
        if isinstance(f, dict):
            print(f"    id={f.get('id', '?'):15s}  name={f.get('name', '?')}")
        else:
            print(f"    {f}")
    
    if len(facets) > 30:
        print(f"    ... and {len(facets) - 30} more")
    
    time.sleep(1)


def probe_sample_data(route, label, params=""):
    """Pull a small sample of actual data"""
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA: {label}")
    print(f"{'='*60}")
    
    url = f"{BASE}/{route}/data?api_key={API_KEY}&length=5&sort[0][column]=period&sort[0][direction]=desc{params}"
    print(f"  URL: ...{route}/data?...length=5&sort=desc{params}")
    data = fetch_json(url)
    
    if not data:
        print("  No response")
        return
    
    resp = data.get("response", {})
    total = resp.get("total")
    print(f"  Total matching rows: {total}")
    
    rows = resp.get("data", [])
    if rows:
        print(f"  Sample ({len(rows)} rows):")
        for row in rows:
            print(f"    {json.dumps(row, indent=None)}")
    else:
        print("  NO DATA RETURNED")
        # Print full response for debugging
        print(f"  Full response keys: {list(resp.keys())}")
        print(f"  Full response: {json.dumps(resp, indent=2)[:500]}")
    
    time.sleep(1)


# ============================================================
# START PROBING
# ============================================================
print("EIA API v2 DIAGNOSTIC - Natural Gas Storage")
print(f"API Key: {API_KEY[:6]}...{API_KEY[-4:]}")
print(f"{'='*60}")

# Step 1: Natural gas top level
probe("Natural Gas - Top Level", "natural-gas")

# Step 2: Storage sub-routes
probe("Natural Gas > Storage", "natural-gas/stor")

# Step 3: Weekly storage metadata
result = probe("Natural Gas > Storage > Weekly", "natural-gas/stor/wkly")

# Step 4: Get facet values for 'process' (this is likely the key facet)
if result:
    resp = result.get("response", {})
    facets = resp.get("facets", [])
    for f in facets:
        if isinstance(f, dict):
            facet_id = f.get("id")
            if facet_id:
                probe_facet_values("natural-gas/stor/wkly", facet_id, f.get("description", facet_id))

# Step 5: Try pulling sample data with NO facet filter
probe_sample_data("natural-gas/stor/wkly", "Weekly Storage - No Filter", "&data[0]=value")

# Step 6: Try with frequency=weekly explicitly
probe_sample_data("natural-gas/stor/wkly", "Weekly Storage - freq=weekly", "&frequency=weekly&data[0]=value")

# Step 7: Try the v1 series ID translation
print(f"\n{'='*60}")
print("V1 SERIES ID TRANSLATION")
print(f"{'='*60}")
v1_ids = [
    "NG.NW2_EPG0_SWO_R48_BCF.W",   # Lower 48
    "NG.NW2_EPG0_SWO_REC_BCF.W",    # East
    "NG.NW2_EPG0_SWO_RMW_BCF.W",    # Midwest
]
for sid in v1_ids:
    print(f"\n  Translating: {sid}")
    url = f"{BASE}/seriesid/{sid}?api_key={API_KEY}"
    data = fetch_json(url)
    if data:
        resp = data.get("response", {})
        rows = resp.get("data", [])
        if rows:
            print(f"    Got {len(rows)} rows! First: {json.dumps(rows[0])}")
            total = resp.get("total")
            print(f"    Total available: {total}")
        else:
            print(f"    No data. Response: {json.dumps(resp)[:300]}")
    time.sleep(1)

print(f"\n{'='*60}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*60}")
