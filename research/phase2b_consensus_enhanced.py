#!/usr/bin/env python3
"""
PHASE 2B: CONSENSUS PROXY & ENHANCED MODEL
Three tracks:
  A) 5-year average as consensus proxy — backtest surprise vs price
  B) Enhanced model features to beat consensus
  C) Forward-looking consensus capture setup

Usage: python3 phase2b_consensus_enhanced.py
  Run from: ~/Desktop/Claude_Programs/Trading_Programs/
  Expects: data/nat_gas_weather.db with completed master_weekly table
"""

import os
import sys
import sqlite3
import math
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "data/nat_gas_weather.db"
RESULTS_DIR = "results"

# ============================================================
# UTILITY FUNCTIONS (same as Phase 2)
# ============================================================
def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def stdev(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def correlation(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None, None, len(pairs)
    n = len(pairs)
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0, None, n
    r = sum((a - mx) * (b - my) for a, b in pairs) / ((n - 1) * sx * sy)
    if abs(r) >= 1:
        t_stat = float('inf')
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    return r, t_stat, n

def t_test_two_sample(vals1, vals2):
    v1 = [v for v in vals1 if v is not None]
    v2 = [v for v in vals2 if v is not None]
    if len(v1) < 2 or len(v2) < 2:
        return None, None, None, None
    m1, m2 = mean(v1), mean(v2)
    s1, s2 = stdev(v1), stdev(v2)
    n1, n2 = len(v1), len(v2)
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    if se == 0:
        return m1 - m2, float('inf'), n1, n2
    t_stat = (m1 - m2) / se
    return m1 - m2, t_stat, n1, n2

def percentile(vals, p):
    vals = sorted([v for v in vals if v is not None])
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    d = k - f
    return vals[f] + d * (vals[c] - vals[f])

def simple_regression(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    ss_xy = sum((a - mx) * (b - my) for a, b in pairs)
    ss_xx = sum((a - mx) ** 2 for a in xs)
    if ss_xx == 0:
        return None
    b = ss_xy / ss_xx
    a = my - b * mx
    ss_tot = sum((b_val - my) ** 2 for b_val in ys)
    predictions = [a + b * x_val for x_val in xs]
    ss_res = sum((y_val - p) ** 2 for y_val, p in zip(ys, predictions))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    se = math.sqrt(ss_res / (n - 2)) if n > 2 else None
    return {"intercept": a, "slope": b, "r_squared": r_sq, "se": se, "n": n}

def multi_reg(xs_list, y):
    """Multi-variable OLS using normal equations. xs_list = list of x vectors."""
    # Filter to complete cases
    n_vars = len(xs_list)
    complete = []
    for i in range(len(y)):
        if y[i] is None:
            continue
        all_good = True
        for xs in xs_list:
            if xs[i] is None:
                all_good = False
                break
        if all_good:
            row = [xs[i] for xs in xs_list] + [y[i]]
            complete.append(row)
    
    if len(complete) < n_vars + 2:
        return None
    
    n = len(complete)
    
    # Extract columns
    x_cols = [[row[j] for row in complete] for j in range(n_vars)]
    y_col = [row[n_vars] for row in complete]
    
    means_x = [mean(col) for col in x_cols]
    mean_y = mean(y_col)
    
    # Build moment matrix S and vector Sy
    S = [[0]*n_vars for _ in range(n_vars)]
    Sy = [0]*n_vars
    
    for i in range(n_vars):
        for j in range(n_vars):
            S[i][j] = sum((x_cols[i][k] - means_x[i]) * (x_cols[j][k] - means_x[j]) for k in range(n))
        Sy[i] = sum((x_cols[i][k] - means_x[i]) * (y_col[k] - mean_y) for k in range(n))
    
    # Solve using Gaussian elimination
    aug = [S[i][:] + [Sy[i]] for i in range(n_vars)]
    for i in range(n_vars):
        # Find pivot
        max_row = max(range(i, n_vars), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        if abs(aug[i][i]) < 1e-12:
            return None
        for j in range(i+1, n_vars):
            factor = aug[j][i] / aug[i][i]
            for k in range(i, n_vars + 1):
                aug[j][k] -= factor * aug[i][k]
    
    # Back substitution
    betas = [0]*n_vars
    for i in range(n_vars - 1, -1, -1):
        betas[i] = aug[i][n_vars]
        for j in range(i+1, n_vars):
            betas[i] -= aug[i][j] * betas[j]
        betas[i] /= aug[i][i]
    
    intercept = mean_y - sum(betas[i] * means_x[i] for i in range(n_vars))
    
    # R-squared
    predictions = [intercept + sum(betas[j] * x_cols[j][k] for j in range(n_vars)) for k in range(n)]
    ss_res = sum((y_col[k] - predictions[k])**2 for k in range(n))
    ss_tot = sum((y_col[k] - mean_y)**2 for k in range(n))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    se = math.sqrt(ss_res / (n - n_vars - 1)) if n > n_vars + 1 else None
    
    return {"intercept": intercept, "betas": betas, "r_squared": r_sq, "se": se, "n": n}

def predict_multi(model, x_vals):
    """Predict from multi_reg model"""
    if model is None:
        return None
    return model["intercept"] + sum(model["betas"][i] * x_vals[i] for i in range(len(x_vals)))


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("PHASE 2B: CONSENSUS PROXY & ENHANCED MODEL")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

if not os.path.exists(DB_PATH):
    print(f"ERROR: {DB_PATH} not found")
    sys.exit(1)

os.makedirs(RESULTS_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM master_weekly ORDER BY report_date")
rows = cursor.fetchall()
data = [dict(row) for row in rows]
print(f"\nLoaded {len(data)} weeks from master_weekly")


# ============================================================
# TRACK A: 5-YEAR AVERAGE AS CONSENSUS PROXY
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK A: 5-YEAR AVERAGE CONSENSUS PROXY")
print(f"{'=' * 70}")

# For each report date, compute what the 5-year average change was
# for the same ISO week in the 5 prior years
# Also compute: storage level vs 5-year average level

# Build lookup: (year, iso_week) -> storage_change and storage_level
from datetime import date

def iso_week(date_str):
    """Get ISO week number from date string"""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.isocalendar()[1]

# Build year-week lookup for storage changes
year_week_change = {}
year_week_level = {}
for d in data:
    if d["storage_change_bcf"] is None:
        continue
    yr = int(d["report_date"][:4])
    wk = iso_week(d["report_date"])
    year_week_change[(yr, wk)] = d["storage_change_bcf"]
    year_week_level[(yr, wk)] = d["storage_level_bcf"]

# Compute 5-year average for each week
enriched = []
for d in data:
    e = dict(d)
    yr = int(d["report_date"][:4])
    wk = iso_week(d["report_date"])
    
    # 5-year average change for this week
    prior_changes = []
    prior_levels = []
    for y_offset in range(1, 6):
        prior_yr = yr - y_offset
        # Look at same week +/- 1 week for flexibility
        for wk_offset in [0, -1, 1]:
            key = (prior_yr, wk + wk_offset)
            if key in year_week_change:
                prior_changes.append(year_week_change[key])
                prior_levels.append(year_week_level[key])
                break
    
    if len(prior_changes) >= 3:
        e["avg5yr_change"] = mean(prior_changes)
        e["avg5yr_level"] = mean(prior_levels)
        
        # CONSENSUS SURPRISE: actual - 5yr avg
        if d["storage_change_bcf"] is not None:
            e["consensus_surprise"] = d["storage_change_bcf"] - e["avg5yr_change"]
        
        # STORAGE VS NORM: how full vs 5yr avg
        if d["storage_level_bcf"] is not None and e["avg5yr_level"] is not None:
            e["storage_vs_5yr"] = d["storage_level_bcf"] - e["avg5yr_level"]
            e["storage_vs_5yr_pct"] = (d["storage_level_bcf"] / e["avg5yr_level"] - 1) * 100
    
    enriched.append(e)

# Filter to rows with consensus surprise
with_consensus = [e for e in enriched if "consensus_surprise" in e and e["ng_return_1d"] is not None]
print(f"\n  Rows with 5yr-avg consensus proxy: {len(with_consensus)}")
print(f"  Date range: {with_consensus[0]['report_date']} to {with_consensus[-1]['report_date']}")

# Basic stats
c_surps = [e["consensus_surprise"] for e in with_consensus]
print(f"\n  CONSENSUS SURPRISE (Actual - 5yr Avg) STATS:")
print(f"    Mean:     {mean(c_surps):.1f} BCF")
print(f"    Std dev:  {stdev(c_surps):.1f} BCF")
print(f"    Min:      {min(c_surps):.0f} BCF")
print(f"    Max:      {max(c_surps):.0f} BCF")

# Core test: consensus surprise vs price
surp = [e["consensus_surprise"] for e in with_consensus]
ret1 = [e["ng_return_1d"] for e in with_consensus]
ret5 = [e["ng_return_5d"] for e in with_consensus]

r1, t1, n1 = correlation(surp, ret1)
r5, t5, n5 = correlation(surp, ret5)
print(f"\n  CONSENSUS SURPRISE vs PRICE:")
print(f"    vs 1-day return:  r={r1:.4f}, t={t1:.2f}, n={n1}")
print(f"    vs 5-day return:  r={r5:.4f}, t={t5:.2f}, n={n5}")

# Quintile analysis
with_consensus.sort(key=lambda x: x["consensus_surprise"])
q_size = len(with_consensus) // 5
print(f"\n  QUINTILE ANALYSIS (sorted by consensus surprise):")
print(f"  {'Quintile':<14s} {'n':>4s} {'Avg Surprise':>14s} {'Avg 1d%':>10s} {'Avg 5d%':>10s} {'1d Win%':>10s}")
print(f"  {'-'*56}")

for i in range(5):
    start = i * q_size
    end = start + q_size if i < 4 else len(with_consensus)
    q = with_consensus[start:end]
    
    surps_q = [p["consensus_surprise"] for p in q]
    rets_1d = [p["ng_return_1d"] for p in q if p["ng_return_1d"] is not None]
    rets_5d = [p["ng_return_5d"] for p in q if p["ng_return_5d"] is not None]
    
    # Bearish surprise (draw > expected) should push price UP
    wins = sum(1 for p in q if p["ng_return_1d"] is not None and
               ((p["consensus_surprise"] < 0 and p["ng_return_1d"] > 0) or
                (p["consensus_surprise"] > 0 and p["ng_return_1d"] < 0)))
    win_pct = wins / len(rets_1d) * 100 if rets_1d else 0
    
    label = f"Q{i+1}"
    if i == 0: label += " (bearish)"
    elif i == 4: label += " (bullish)"
    
    print(f"  {label:<14s} {len(q):>4d} {mean(surps_q):>14.1f} {mean(rets_1d):>10.2f} {mean(rets_5d):>10.2f} {win_pct:>9.1f}%")

# Binary signal
bearish = [e for e in with_consensus if e["consensus_surprise"] < 0]
bullish = [e for e in with_consensus if e["consensus_surprise"] > 0]

bear_1d = [e["ng_return_1d"] for e in bearish if e["ng_return_1d"] is not None]
bull_1d = [e["ng_return_1d"] for e in bullish if e["ng_return_1d"] is not None]
bear_5d = [e["ng_return_5d"] for e in bearish if e["ng_return_5d"] is not None]
bull_5d = [e["ng_return_5d"] for e in bullish if e["ng_return_5d"] is not None]

bear_w1 = sum(1 for r in bear_1d if r > 0) / len(bear_1d) * 100 if bear_1d else 0
bull_w1 = sum(1 for r in bull_1d if r < 0) / len(bull_1d) * 100 if bull_1d else 0
bear_w5 = sum(1 for r in bear_5d if r > 0) / len(bear_5d) * 100 if bear_5d else 0
bull_w5 = sum(1 for r in bull_5d if r < 0) / len(bull_5d) * 100 if bull_5d else 0

print(f"\n  BINARY SIGNAL (5yr-avg consensus proxy):")
print(f"  {'Signal':<28s} {'n':>5s} {'Avg 1d%':>10s} {'Avg 5d%':>10s} {'1d Win%':>10s} {'5d Win%':>10s}")
print(f"  {'-'*68}")
print(f"  {'Bearish surprise (LONG)':<28s} {len(bearish):>5d} {mean(bear_1d):>10.2f} {mean(bear_5d):>10.2f} {bear_w1:>9.1f}% {bear_w5:>9.1f}%")
print(f"  {'Bullish surprise (SHORT)':<28s} {len(bullish):>5d} {mean(bull_1d):>10.2f} {mean(bull_5d):>10.2f} {bull_w1:>9.1f}% {bull_w5:>9.1f}%")

diff1, t_diff1, _, _ = t_test_two_sample(bear_1d, bull_1d)
diff5, t_diff5, _, _ = t_test_two_sample(bear_5d, bull_5d)
print(f"\n  T-test 1d: diff={diff1:.2f}%, t={t_diff1:.2f}")
print(f"  T-test 5d: diff={diff5:.2f}%, t={t_diff5:.2f}")

# Large surprise filter
surp_std = stdev(c_surps)
for mult in [1.0, 1.5, 2.0]:
    threshold = mult * surp_std
    lb = [e for e in with_consensus if e["consensus_surprise"] < -threshold]
    lu = [e for e in with_consensus if e["consensus_surprise"] > threshold]
    
    lb_r1 = [e["ng_return_1d"] for e in lb if e["ng_return_1d"] is not None]
    lu_r1 = [e["ng_return_1d"] for e in lu if e["ng_return_1d"] is not None]
    lb_r5 = [e["ng_return_5d"] for e in lb if e["ng_return_5d"] is not None]
    lu_r5 = [e["ng_return_5d"] for e in lu if e["ng_return_5d"] is not None]
    
    print(f"\n  At {mult}x std ({threshold:.0f} BCF threshold):")
    if lb_r1:
        w = sum(1 for r in lb_r1 if r > 0) / len(lb_r1) * 100
        print(f"    Bearish LONG:  n={len(lb):>4d}, avg_1d={mean(lb_r1):>7.2f}%, avg_5d={mean(lb_r5):>7.2f}%, win_1d={w:.0f}%")
    if lu_r1:
        w = sum(1 for r in lu_r1 if r < 0) / len(lu_r1) * 100
        print(f"    Bullish SHORT: n={len(lu):>4d}, avg_1d={mean(lu_r1):>7.2f}%, avg_5d={mean(lu_r5):>7.2f}%, win_1d={w:.0f}%")


# ============================================================
# TRACK A2: STORAGE LEVEL INTERACTION
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK A2: DOES STORAGE LEVEL AMPLIFY THE SIGNAL?")
print(f"{'=' * 70}")
print("  When storage is LOW, bearish surprises should hit harder (supply fear)")
print("  When storage is HIGH, bullish surprises should hit harder (glut fear)")

# Split by storage vs 5yr avg
low_storage = [e for e in with_consensus if "storage_vs_5yr_pct" in e and e["storage_vs_5yr_pct"] < -5]
high_storage = [e for e in with_consensus if "storage_vs_5yr_pct" in e and e["storage_vs_5yr_pct"] > 5]
normal_storage = [e for e in with_consensus if "storage_vs_5yr_pct" in e and -5 <= e["storage_vs_5yr_pct"] <= 5]

for label, group in [("LOW (>5% below avg)", low_storage), 
                      ("NORMAL (+/- 5%)", normal_storage),
                      ("HIGH (>5% above avg)", high_storage)]:
    if len(group) < 10:
        continue
    
    s = [e["consensus_surprise"] for e in group]
    r = [e["ng_return_1d"] for e in group if e["ng_return_1d"] is not None]
    corr_r, corr_t, corr_n = correlation(s, [e["ng_return_1d"] for e in group])
    
    # Bearish vs bullish within this group
    bear_g = [e["ng_return_1d"] for e in group if e["consensus_surprise"] < 0 and e["ng_return_1d"] is not None]
    bull_g = [e["ng_return_1d"] for e in group if e["consensus_surprise"] > 0 and e["ng_return_1d"] is not None]
    
    corr_str = f"{corr_r:.3f}" if corr_r is not None else "N/A"
    bear_str = f"{mean(bear_g):.2f}%" if bear_g else "N/A"
    bull_str = f"{mean(bull_g):.2f}%" if bull_g else "N/A"
    
    print(f"\n  {label}: n={len(group)}")
    print(f"    Corr(surprise, 1d ret): {corr_str}")
    print(f"    Bearish surprise avg 1d: {bear_str} (n={len(bear_g)})")
    print(f"    Bullish surprise avg 1d: {bull_str} (n={len(bull_g)})")
    
    if bear_g and bull_g:
        diff, t_val, _, _ = t_test_two_sample(bear_g, bull_g)
        print(f"    T-test: diff={diff:.2f}%, t={t_val:.2f}")


# ============================================================
# TRACK A3: SEASONAL SPLIT WITH CONSENSUS
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK A3: SEASONAL CONSENSUS SIGNAL")
print(f"{'=' * 70}")

for season_label, season_data in [("WITHDRAWAL", [e for e in with_consensus if e["season"] == "WITHDRAWAL"]),
                                   ("INJECTION", [e for e in with_consensus if e["season"] == "INJECTION"])]:
    s = [e["consensus_surprise"] for e in season_data]
    r1s = [e["ng_return_1d"] for e in season_data]
    r5s = [e["ng_return_5d"] for e in season_data]
    
    cr1, ct1, cn1 = correlation(s, r1s)
    cr5, ct5, cn5 = correlation(s, r5s)
    
    bear = [e for e in season_data if e["consensus_surprise"] < 0]
    bull = [e for e in season_data if e["consensus_surprise"] > 0]
    
    br1 = [e["ng_return_1d"] for e in bear if e["ng_return_1d"] is not None]
    bu1 = [e["ng_return_1d"] for e in bull if e["ng_return_1d"] is not None]
    br5 = [e["ng_return_5d"] for e in bear if e["ng_return_5d"] is not None]
    bu5 = [e["ng_return_5d"] for e in bull if e["ng_return_5d"] is not None]
    
    print(f"\n  {season_label} SEASON (n={len(season_data)}):")
    print(f"    Corr(surprise, 1d): r={cr1:.4f}, t={ct1:.2f}" if cr1 else f"    Corr: N/A")
    print(f"    Corr(surprise, 5d): r={cr5:.4f}, t={ct5:.2f}" if cr5 else f"    Corr: N/A")
    
    if br1 and bu1:
        bw1 = sum(1 for r in br1 if r > 0) / len(br1) * 100
        uw1 = sum(1 for r in bu1 if r < 0) / len(bu1) * 100
        print(f"    Bearish LONG:  n={len(bear)}, avg_1d={mean(br1):.2f}%, avg_5d={mean(br5):.2f}%, win_1d={bw1:.0f}%")
        print(f"    Bullish SHORT: n={len(bull)}, avg_1d={mean(bu1):.2f}%, avg_5d={mean(bu5):.2f}%, win_1d={uw1:.0f}%")
        diff, t_val, _, _ = t_test_two_sample(br1, bu1)
        print(f"    T-test 1d: diff={diff:.2f}%, t={t_val:.2f}")


# ============================================================
# TRACK B: ENHANCED MODEL (BEAT CONSENSUS)
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK B: ENHANCED PREDICTION MODEL")
print(f"{'=' * 70}")

# Build features for each week
print("\n  Building enhanced feature set...")

features = []
for i in range(1, len(enriched)):
    e = enriched[i]
    prev = enriched[i-1]
    
    if e.get("storage_change_bcf") is None or e.get("national_hdd") is None:
        continue
    
    f = {
        "report_date": e["report_date"],
        "actual_change": e["storage_change_bcf"],
        "season": e["season"],
        "ng_return_1d": e.get("ng_return_1d"),
        "ng_return_5d": e.get("ng_return_5d"),
        
        # Weather features
        "national_hdd": e["national_hdd"],
        "national_cdd": e.get("national_cdd", 0) or 0,
        
        # Momentum: prior week's storage change
        "prior_change": prev.get("storage_change_bcf"),
        
        # Storage level context
        "storage_level": e.get("storage_level_bcf"),
        "storage_vs_5yr": e.get("storage_vs_5yr"),
        "storage_vs_5yr_pct": e.get("storage_vs_5yr_pct"),
        
        # 5yr avg consensus proxy
        "avg5yr_change": e.get("avg5yr_change"),
        "consensus_surprise": e.get("consensus_surprise"),
        
        # Regional weather deviation (anomalies)
        # If one region is much colder than usual while others aren't,
        # that regional anomaly might be underweighted by consensus
    }
    
    # Regional HDD deviation from national proportion
    # (higher = that region is disproportionately cold)
    if e.get("national_hdd") and e["national_hdd"] > 0:
        for reg in ["east", "midwest", "south_central", "mountain", "pacific"]:
            reg_hdd = e.get(f"{reg}_hdd")
            if reg_hdd is not None:
                f[f"{reg}_hdd_share"] = reg_hdd / e["national_hdd"]
    
    # Week-over-week HDD change (is it getting colder or warmer?)
    if prev.get("national_hdd") is not None and e.get("national_hdd") is not None:
        f["hdd_change"] = e["national_hdd"] - prev["national_hdd"]
    
    # NG price level (high price = different market dynamics)
    f["ng_price"] = e.get("ng_close_report_day")
    
    features.append(f)

print(f"  Feature rows: {len(features)}")

# Walk-forward enhanced model test
print(f"\n  WALK-FORWARD OUT-OF-SAMPLE TEST (Enhanced Model)")
print(f"  Training on expanding window, min 3 years")

train_min = 156  # ~3 years

oos_results = []

for i in range(train_min, len(features)):
    f = features[i]
    
    # Skip if missing key features
    if f["actual_change"] is None or f["national_hdd"] is None:
        continue
    if f["ng_return_1d"] is None:
        continue
    
    # Training set: all prior rows in same season
    train = [features[j] for j in range(i)
             if features[j]["season"] == f["season"]
             and features[j]["actual_change"] is not None
             and features[j]["national_hdd"] is not None
             and features[j].get("prior_change") is not None]
    
    if len(train) < 30:
        continue
    
    # MODEL A: Basic (HDD + CDD) - same as Phase 2
    x1_t = [t["national_hdd"] for t in train]
    x2_t = [t["national_cdd"] for t in train]
    y_t = [t["actual_change"] for t in train]
    
    basic_model = multi_reg([x1_t, x2_t], y_t)
    if basic_model:
        pred_basic = predict_multi(basic_model, [f["national_hdd"], f["national_cdd"]])
    else:
        pred_basic = None
    
    # MODEL B: Enhanced (HDD + CDD + prior_change + storage_vs_5yr)
    x3_t = [t["prior_change"] for t in train]
    x4_t = [t.get("storage_vs_5yr") for t in train]
    
    # Only use storage_vs_5yr if available for enough training rows
    train_with_sv = [j for j, t in enumerate(train) if t.get("storage_vs_5yr") is not None]
    
    if len(train_with_sv) > 50 and f.get("storage_vs_5yr") is not None:
        x1_e = [train[j]["national_hdd"] for j in train_with_sv]
        x2_e = [train[j]["national_cdd"] for j in train_with_sv]
        x3_e = [train[j]["prior_change"] for j in train_with_sv]
        x4_e = [train[j]["storage_vs_5yr"] for j in train_with_sv]
        y_e = [train[j]["actual_change"] for j in train_with_sv]
        
        enhanced_model = multi_reg([x1_e, x2_e, x3_e, x4_e], y_e)
        if enhanced_model:
            pred_enhanced = predict_multi(enhanced_model, [
                f["national_hdd"], f["national_cdd"], 
                f["prior_change"], f["storage_vs_5yr"]
            ])
        else:
            pred_enhanced = None
    else:
        pred_enhanced = None
        enhanced_model = None
    
    # MODEL C: Enhanced + HDD change momentum
    if (len(train_with_sv) > 50 and f.get("storage_vs_5yr") is not None 
        and f.get("hdd_change") is not None):
        
        train_with_all = [j for j in train_with_sv if train[j].get("hdd_change") is not None]
        
        if len(train_with_all) > 50:
            x1_c = [train[j]["national_hdd"] for j in train_with_all]
            x2_c = [train[j]["national_cdd"] for j in train_with_all]
            x3_c = [train[j]["prior_change"] for j in train_with_all]
            x4_c = [train[j]["storage_vs_5yr"] for j in train_with_all]
            x5_c = [train[j]["hdd_change"] for j in train_with_all]
            y_c = [train[j]["actual_change"] for j in train_with_all]
            
            full_model = multi_reg([x1_c, x2_c, x3_c, x4_c, x5_c], y_c)
            if full_model:
                pred_full = predict_multi(full_model, [
                    f["national_hdd"], f["national_cdd"], 
                    f["prior_change"], f["storage_vs_5yr"], f["hdd_change"]
                ])
            else:
                pred_full = None
        else:
            pred_full = None
    else:
        pred_full = None
    
    # 5yr avg prediction
    pred_5yr = f.get("avg5yr_change")
    
    result = {
        "report_date": f["report_date"],
        "actual": f["actual_change"],
        "season": f["season"],
        "ng_return_1d": f["ng_return_1d"],
        "ng_return_5d": f["ng_return_5d"],
        "pred_basic": pred_basic,
        "pred_enhanced": pred_enhanced,
        "pred_full": pred_full,
        "pred_5yr_avg": pred_5yr,
    }
    
    # Compute surprises for each model
    for model_name, pred in [("basic", pred_basic), ("enhanced", pred_enhanced), 
                              ("full", pred_full), ("5yr_avg", pred_5yr)]:
        if pred is not None:
            result[f"surprise_{model_name}"] = f["actual_change"] - pred
            result[f"abs_error_{model_name}"] = abs(f["actual_change"] - pred)
    
    oos_results.append(result)

print(f"\n  OOS predictions: {len(oos_results)}")

# Compare model accuracy
print(f"\n  MODEL ACCURACY COMPARISON (Out-of-Sample MAE):")
print(f"  {'Model':<20s} {'MAE (BCF)':>10s} {'Median AE':>10s} {'n':>6s}")
print(f"  {'-'*50}")

for model_name in ["5yr_avg", "basic", "enhanced", "full"]:
    errors = [r[f"abs_error_{model_name}"] for r in oos_results if f"abs_error_{model_name}" in r]
    if errors:
        print(f"  {model_name:<20s} {mean(errors):>10.1f} {percentile(errors, 50):>10.1f} {len(errors):>6d}")

# Compare surprise-to-price correlations
print(f"\n  SURPRISE-TO-PRICE CORRELATION (OOS, 1-day return):")
print(f"  {'Model':<20s} {'r':>8s} {'t-stat':>8s} {'n':>6s}")
print(f"  {'-'*45}")

for model_name in ["5yr_avg", "basic", "enhanced", "full"]:
    surps = [r.get(f"surprise_{model_name}") for r in oos_results]
    rets = [r["ng_return_1d"] for r in oos_results]
    r_val, t_val, n_val = correlation(surps, rets)
    if r_val is not None:
        print(f"  {model_name:<20s} {r_val:>8.4f} {t_val:>8.2f} {n_val:>6d}")

print(f"\n  SURPRISE-TO-PRICE CORRELATION (OOS, 5-day return):")
print(f"  {'Model':<20s} {'r':>8s} {'t-stat':>8s} {'n':>6s}")
print(f"  {'-'*45}")

for model_name in ["5yr_avg", "basic", "enhanced", "full"]:
    surps = [r.get(f"surprise_{model_name}") for r in oos_results]
    rets = [r["ng_return_5d"] for r in oos_results]
    r_val, t_val, n_val = correlation(surps, rets)
    if r_val is not None:
        print(f"  {model_name:<20s} {r_val:>8.4f} {t_val:>8.2f} {n_val:>6d}")

# Binary signal test for each model
print(f"\n  BINARY SIGNAL TEST (OOS):")
print(f"  {'Model':<20s} {'Bear n':>7s} {'Bear 1d%':>10s} {'Bull n':>7s} {'Bull 1d%':>10s} {'Diff':>8s} {'t-stat':>8s}")
print(f"  {'-'*75}")

for model_name in ["5yr_avg", "basic", "enhanced", "full"]:
    bear = [r for r in oos_results if f"surprise_{model_name}" in r and r[f"surprise_{model_name}"] < 0]
    bull = [r for r in oos_results if f"surprise_{model_name}" in r and r[f"surprise_{model_name}"] > 0]
    
    br = [r["ng_return_1d"] for r in bear if r["ng_return_1d"] is not None]
    bu = [r["ng_return_1d"] for r in bull if r["ng_return_1d"] is not None]
    
    if br and bu:
        diff, t_val, _, _ = t_test_two_sample(br, bu)
        print(f"  {model_name:<20s} {len(bear):>7d} {mean(br):>10.2f} {len(bull):>7d} {mean(bu):>10.2f} {diff:>8.2f} {t_val:>8.2f}")


# ============================================================
# TRACK B2: WHAT IF OUR MODEL BEATS 5YR AVG?
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK B2: MODEL EDGE OVER CONSENSUS PROXY")
print(f"{'=' * 70}")
print("  If our model predicts BETTER than 5yr avg, the difference between")
print("  our prediction and 5yr avg tells us which side of consensus to bet.")

# Compute: model says X, consensus proxy says Y, difference = our edge
edge_results = []
for r in oos_results:
    if r.get("pred_enhanced") is not None and r.get("pred_5yr_avg") is not None:
        model_edge = r["pred_enhanced"] - r["pred_5yr_avg"]
        # If model_edge < 0: our model predicts bigger draw than consensus -> bearish signal -> go long
        # If model_edge > 0: our model predicts smaller draw than consensus -> bullish signal -> go short
        edge_results.append({
            "report_date": r["report_date"],
            "model_edge": model_edge,
            "actual": r["actual"],
            "ng_return_1d": r["ng_return_1d"],
            "ng_return_5d": r["ng_return_5d"],
        })

if edge_results:
    print(f"\n  Model edge observations: {len(edge_results)}")
    
    # Does model edge predict direction correctly?
    edge_correct = sum(1 for e in edge_results 
                       if (e["model_edge"] < 0 and e["actual"] < (e["actual"] - e["model_edge"])) or
                          (e["model_edge"] > 0 and e["actual"] > (e["actual"] - e["model_edge"])))
    # Simpler: does direction of model edge correlate with return?
    edges = [e["model_edge"] for e in edge_results]
    rets_1d = [e["ng_return_1d"] for e in edge_results]
    rets_5d = [e["ng_return_5d"] for e in edge_results]
    
    r_e1, t_e1, n_e1 = correlation(edges, rets_1d)
    r_e5, t_e5, n_e5 = correlation(edges, rets_5d)
    
    print(f"  Model edge vs 1d return: r={r_e1:.4f}, t={t_e1:.2f}")
    print(f"  Model edge vs 5d return: r={r_e5:.4f}, t={t_e5:.2f}")
    
    # Binary: model says more bearish than consensus -> go long
    bear_edge = [e for e in edge_results if e["model_edge"] < -5]  # at least 5 BCF more bearish
    bull_edge = [e for e in edge_results if e["model_edge"] > 5]
    
    if bear_edge and bull_edge:
        be_r1 = [e["ng_return_1d"] for e in bear_edge if e["ng_return_1d"] is not None]
        bu_r1 = [e["ng_return_1d"] for e in bull_edge if e["ng_return_1d"] is not None]
        
        print(f"\n  Model says >5BCF more bearish than consensus (LONG): n={len(bear_edge)}, avg_1d={mean(be_r1):.2f}%")
        print(f"  Model says >5BCF more bullish than consensus (SHORT): n={len(bull_edge)}, avg_1d={mean(bu_r1):.2f}%")
        
        diff, t_val, _, _ = t_test_two_sample(be_r1, bu_r1)
        print(f"  T-test: diff={diff:.2f}%, t={t_val:.2f}")


# ============================================================
# TRACK C: LONGER-TERM POSITIONING SIGNAL
# ============================================================
print(f"\n{'=' * 70}")
print("TRACK C: LONGER-TERM POSITIONING")
print(f"{'=' * 70}")
print("  Does cumulative storage deviation from 5yr avg predict multi-week returns?")

# For each week, compute 4-week and 8-week forward NG returns
# and test if storage_vs_5yr_pct predicts them

# Build forward returns
for i in range(len(enriched)):
    # 4-week forward
    if i + 4 < len(enriched):
        close_now = enriched[i].get("ng_close_report_day")
        close_4w = enriched[i+4].get("ng_close_report_day")
        if close_now and close_4w:
            enriched[i]["ng_return_4w"] = (close_4w / close_now - 1) * 100
    
    # 8-week forward
    if i + 8 < len(enriched):
        close_8w = enriched[i+8].get("ng_close_report_day")
        if close_now and close_8w:
            enriched[i]["ng_return_8w"] = (close_8w / close_now - 1) * 100

long_term = [e for e in enriched if "storage_vs_5yr_pct" in e and "ng_return_4w" in e]

if long_term:
    sv5_vals = [e["storage_vs_5yr_pct"] for e in long_term]
    ret4w = [e["ng_return_4w"] for e in long_term]
    ret8w = [e.get("ng_return_8w") for e in long_term]
    
    r4, t4, n4 = correlation(sv5_vals, ret4w)
    r8, t8, n8 = correlation(sv5_vals, ret8w)
    
    print(f"\n  Storage vs 5yr avg % -> 4-week return: r={r4:.4f}, t={t4:.2f}, n={n4}")
    print(f"  Storage vs 5yr avg % -> 8-week return: r={r8:.4f}, t={t8:.2f}, n={n8}")
    
    # Quartile analysis on storage deficit/surplus
    long_term.sort(key=lambda x: x["storage_vs_5yr_pct"])
    qt_size = len(long_term) // 4
    
    print(f"\n  STORAGE DEVIATION QUARTILES vs FORWARD RETURNS:")
    print(f"  {'Quartile':<20s} {'n':>4s} {'Avg Deficit%':>14s} {'Avg 4w Ret%':>14s} {'Avg 8w Ret%':>14s}")
    print(f"  {'-'*70}")
    
    for qi in range(4):
        start = qi * qt_size
        end = start + qt_size if qi < 3 else len(long_term)
        q = long_term[start:end]
        
        dev = [e["storage_vs_5yr_pct"] for e in q]
        r4q = [e["ng_return_4w"] for e in q if e.get("ng_return_4w") is not None]
        r8q = [e.get("ng_return_8w") for e in q if e.get("ng_return_8w") is not None]
        
        label = f"Q{qi+1}"
        if qi == 0: label += " (most below)"
        elif qi == 3: label += " (most above)"
        
        r8_str = f"{mean(r8q):.2f}" if r8q else "N/A"
        print(f"  {label:<20s} {len(q):>4d} {mean(dev):>14.1f} {mean(r4q):>14.2f} {r8_str:>14s}")
    
    # Withdrawal season specifically
    lt_withdrawal = [e for e in long_term if e["season"] == "WITHDRAWAL"]
    if lt_withdrawal:
        sv_w = [e["storage_vs_5yr_pct"] for e in lt_withdrawal]
        r4_w = [e["ng_return_4w"] for e in lt_withdrawal]
        r_w, t_w, n_w = correlation(sv_w, r4_w)
        print(f"\n  WITHDRAWAL SEASON ONLY:")
        print(f"  Storage deficit % -> 4-week return: r={r_w:.4f}, t={t_w:.2f}, n={n_w}")


# ============================================================
# EXPORT ENRICHED DATA
# ============================================================
print(f"\n{'=' * 70}")
print("EXPORTING RESULTS")
print(f"{'=' * 70}")

# Export OOS model comparison
with open(f"{RESULTS_DIR}/oos_model_comparison.csv", "w") as f:
    cols = ["report_date", "season", "actual", "pred_5yr_avg", "pred_basic", 
            "pred_enhanced", "pred_full", "surprise_5yr_avg", "surprise_basic",
            "surprise_enhanced", "surprise_full", "ng_return_1d", "ng_return_5d"]
    f.write(",".join(cols) + "\n")
    for r in oos_results:
        vals = [str(r.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")
print(f"  OOS model comparison: {RESULTS_DIR}/oos_model_comparison.csv ({len(oos_results)} rows)")

# Export enriched master data
with open(f"{RESULTS_DIR}/enriched_weekly.csv", "w") as f:
    cols = ["report_date", "season", "storage_change_bcf", "storage_level_bcf",
            "avg5yr_change", "consensus_surprise", "storage_vs_5yr", "storage_vs_5yr_pct",
            "national_hdd", "national_cdd", "ng_close_report_day", 
            "ng_return_1d", "ng_return_5d"]
    f.write(",".join(cols) + "\n")
    for e in enriched:
        vals = [str(e.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")
print(f"  Enriched weekly data: {RESULTS_DIR}/enriched_weekly.csv ({len(enriched)} rows)")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print("FINAL SUMMARY")
print(f"{'=' * 70}")

print(f"""
  TRACK A — 5yr-avg consensus proxy:
    1d correlation: r={r1:.4f} (t={t1:.2f})
    5d correlation: r={r5:.4f} (t={t5:.2f})
    Binary signal 1d t-test: t={t_diff1:.2f}
    Binary signal 5d t-test: t={t_diff5:.2f}
    
  TRACK B — Enhanced model:
    Compared 4 models on OOS accuracy (see MAE table above)
    Tested surprise-to-price for each model
    Tested model edge over consensus proxy

  TRACK C — Longer-term positioning:
    Storage deficit vs 4-week return: r={r4:.4f} (t={t4:.2f})
    Storage deficit vs 8-week return: r={r8:.4f} (t={t8:.2f})
    
  INTERPRETATION:
    |t| > 2.0: Statistically significant at 95% confidence
    |t| > 2.6: Significant at 99%
    |t| > 3.3: Significant at 99.9%
    |t| < 1.5: Not significant, likely no edge
    
  BOTTOM LINE:
    Look for the analysis with the strongest t-statistics.
    That's where the tradeable signal lives, if one exists.
""")

conn.close()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
