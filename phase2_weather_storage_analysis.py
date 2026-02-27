#!/usr/bin/env python3
"""
PHASE 2: WEATHER-STORAGE RELATIONSHIP EXPLORATION
Analyzes the predictive power of HDD/CDD on natural gas storage changes,
builds a consensus-replacement model, and measures price reaction to surprises.

Usage: python3 phase2_weather_storage_analysis.py
  Run from: ~/Desktop/Claude_Programs/Trading_Programs/
  Expects: data/nat_gas_weather.db with completed master_weekly table
"""

import os
import sys
import sqlite3
import math
from datetime import datetime
from collections import defaultdict

DB_PATH = "data/nat_gas_weather.db"
RESULTS_DIR = "results"

# ============================================================
# UTILITY FUNCTIONS
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
    """Pearson correlation coefficient"""
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
    # t-statistic
    if abs(r) >= 1:
        t_stat = float('inf')
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    return r, t_stat, n

def simple_regression(x, y):
    """Simple OLS: y = a + b*x. Returns (a, b, r_squared, se, n)"""
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
    
    # R-squared
    ss_tot = sum((b_val - my) ** 2 for b_val in ys)
    predictions = [a + b * x_val for x_val in xs]
    ss_res = sum((y_val - p) ** 2 for y_val, p in zip(ys, predictions))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Standard error of estimate
    se = math.sqrt(ss_res / (n - 2)) if n > 2 else None
    
    return {"intercept": a, "slope": b, "r_squared": r_sq, "se": se, "n": n}

def multi_regression_2var(x1, x2, y):
    """Two-variable OLS: y = a + b1*x1 + b2*x2. Manual normal equations."""
    triples = [(a, b, c) for a, b, c in zip(x1, x2, y) 
               if a is not None and b is not None and c is not None]
    if len(triples) < 5:
        return None
    n = len(triples)
    x1s, x2s, ys = zip(*triples)
    
    mx1, mx2, my = mean(x1s), mean(x2s), mean(ys)
    
    # Normal equations for 2-var regression
    s11 = sum((a - mx1) ** 2 for a in x1s)
    s22 = sum((a - mx2) ** 2 for a in x2s)
    s12 = sum((a - mx1) * (b - mx2) for a, b in zip(x1s, x2s))
    s1y = sum((a - mx1) * (b - my) for a, b in zip(x1s, ys))
    s2y = sum((a - mx2) * (b - my) for a, b in zip(x2s, ys))
    
    denom = s11 * s22 - s12 ** 2
    if abs(denom) < 1e-10:
        return None
    
    b1 = (s22 * s1y - s12 * s2y) / denom
    b2 = (s11 * s2y - s12 * s1y) / denom
    a = my - b1 * mx1 - b2 * mx2
    
    predictions = [a + b1 * v1 + b2 * v2 for v1, v2 in zip(x1s, x2s)]
    ss_res = sum((y_val - p) ** 2 for y_val, p in zip(ys, predictions))
    ss_tot = sum((y_val - my) ** 2 for y_val in ys)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    se = math.sqrt(ss_res / (n - 3)) if n > 3 else None
    
    return {"intercept": a, "b_x1": b1, "b_x2": b2, "r_squared": r_sq, "se": se, "n": n}

def multi_regression_3var(x1, x2, x3, y):
    """Three-variable OLS: y = a + b1*x1 + b2*x2 + b3*x3"""
    quads = [(a, b, c, d) for a, b, c, d in zip(x1, x2, x3, y) 
             if a is not None and b is not None and c is not None and d is not None]
    if len(quads) < 6:
        return None
    n = len(quads)
    x1s, x2s, x3s, ys = zip(*quads)
    
    mx1, mx2, mx3, my = mean(x1s), mean(x2s), mean(x3s), mean(ys)
    
    # Build moment matrix manually
    s = [[0]*3 for _ in range(3)]
    sy = [0]*3
    xs_all = [x1s, x2s, x3s]
    mxs = [mx1, mx2, mx3]
    
    for i in range(3):
        for j in range(3):
            s[i][j] = sum((xs_all[i][k] - mxs[i]) * (xs_all[j][k] - mxs[j]) for k in range(n))
        sy[i] = sum((xs_all[i][k] - mxs[i]) * (ys[k] - my) for k in range(n))
    
    # Solve 3x3 system using Cramer's rule
    def det3(m):
        return (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
              - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
              + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))
    
    D = det3(s)
    if abs(D) < 1e-10:
        return None
    
    def replace_col(mat, col, vec):
        m = [row[:] for row in mat]
        for i in range(3):
            m[i][col] = vec[i]
        return m
    
    b1 = det3(replace_col(s, 0, sy)) / D
    b2 = det3(replace_col(s, 1, sy)) / D
    b3 = det3(replace_col(s, 2, sy)) / D
    a = my - b1*mx1 - b2*mx2 - b3*mx3
    
    predictions = [a + b1*v1 + b2*v2 + b3*v3 for v1, v2, v3 in zip(x1s, x2s, x3s)]
    ss_res = sum((y_val - p)**2 for y_val, p in zip(ys, predictions))
    ss_tot = sum((y_val - my)**2 for y_val in ys)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    se = math.sqrt(ss_res / (n - 4)) if n > 4 else None
    
    return {"intercept": a, "b_x1": b1, "b_x2": b2, "b_x3": b3, 
            "r_squared": r_sq, "se": se, "n": n}

def percentile(vals, p):
    """Simple percentile calculation"""
    vals = sorted([v for v in vals if v is not None])
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    d = k - f
    return vals[f] + d * (vals[c] - vals[f])

def t_test_two_sample(vals1, vals2):
    """Two-sample t-test (unequal variance)"""
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


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("PHASE 2: WEATHER-STORAGE RELATIONSHIP EXPLORATION")
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

# Split by season
withdrawal = [d for d in data if d["season"] == "WITHDRAWAL"]
injection = [d for d in data if d["season"] == "INJECTION"]
print(f"  Withdrawal season: {len(withdrawal)} weeks")
print(f"  Injection season:  {len(injection)} weeks")

# ============================================================
# ANALYSIS 1: BASIC CORRELATIONS
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 1: BASIC CORRELATIONS")
print(f"{'=' * 70}")

corr_pairs = [
    ("national_hdd", "storage_change_bcf", "National HDD vs Storage Change"),
    ("national_cdd", "storage_change_bcf", "National CDD vs Storage Change"),
    ("east_hdd", "east_change", "East HDD vs East Change"),
    ("midwest_hdd", "midwest_change", "Midwest HDD vs Midwest Change"),
    ("south_central_hdd", "south_central_change", "South Central HDD vs SC Change"),
    ("mountain_hdd", "mountain_change", "Mountain HDD vs Mountain Change"),
    ("pacific_hdd", "pacific_change", "Pacific HDD vs Pacific Change"),
]

print(f"\n  ALL SEASONS:")
print(f"  {'Pair':<45s} {'r':>8s} {'t-stat':>8s} {'n':>6s}")
print(f"  {'-'*70}")
for x_col, y_col, label in corr_pairs:
    x = [d[x_col] for d in data]
    y = [d[y_col] for d in data]
    r, t, n = correlation(x, y)
    r_str = f"{r:.4f}" if r is not None else "N/A"
    t_str = f"{t:.2f}" if t is not None else "N/A"
    print(f"  {label:<45s} {r_str:>8s} {t_str:>8s} {n:>6d}")

print(f"\n  WITHDRAWAL SEASON ONLY:")
print(f"  {'Pair':<45s} {'r':>8s} {'t-stat':>8s} {'n':>6s}")
print(f"  {'-'*70}")
for x_col, y_col, label in corr_pairs:
    x = [d[x_col] for d in withdrawal]
    y = [d[y_col] for d in withdrawal]
    r, t, n = correlation(x, y)
    r_str = f"{r:.4f}" if r is not None else "N/A"
    t_str = f"{t:.2f}" if t is not None else "N/A"
    print(f"  {label:<45s} {r_str:>8s} {t_str:>8s} {n:>6d}")

print(f"\n  INJECTION SEASON ONLY:")
print(f"  {'Pair':<45s} {'r':>8s} {'t-stat':>8s} {'n':>6s}")
print(f"  {'-'*70}")
for x_col, y_col, label in corr_pairs:
    x = [d[x_col] for d in injection]
    y = [d[y_col] for d in injection]
    r, t, n = correlation(x, y)
    r_str = f"{r:.4f}" if r is not None else "N/A"
    t_str = f"{t:.2f}" if t is not None else "N/A"
    print(f"  {label:<45s} {r_str:>8s} {t_str:>8s} {n:>6d}")


# ============================================================
# ANALYSIS 2: REGRESSION MODELS
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 2: REGRESSION MODELS (Predicting Storage Change)")
print(f"{'=' * 70}")

# Model 1: Simple - HDD only (all seasons)
print(f"\n  MODEL 1: storage_change = a + b * national_hdd  (ALL SEASONS)")
x = [d["national_hdd"] for d in data]
y = [d["storage_change_bcf"] for d in data]
reg = simple_regression(x, y)
if reg:
    print(f"    Intercept:  {reg['intercept']:.4f}")
    print(f"    Slope:      {reg['slope']:.4f} BCF per HDD")
    print(f"    R-squared:  {reg['r_squared']:.4f}")
    print(f"    Std Error:  {reg['se']:.2f} BCF")
    print(f"    n:          {reg['n']}")
    print(f"    Interpretation: Each 1-unit increase in national HDD predicts")
    print(f"    a {abs(reg['slope']):.4f} BCF {'draw' if reg['slope'] < 0 else 'injection'}")

# Model 2: HDD + CDD (all seasons)
print(f"\n  MODEL 2: storage_change = a + b1*HDD + b2*CDD  (ALL SEASONS)")
x1 = [d["national_hdd"] for d in data]
x2 = [d["national_cdd"] for d in data]
y = [d["storage_change_bcf"] for d in data]
reg2 = multi_regression_2var(x1, x2, y)
if reg2:
    print(f"    Intercept:  {reg2['intercept']:.4f}")
    print(f"    b_HDD:      {reg2['b_x1']:.4f}")
    print(f"    b_CDD:      {reg2['b_x2']:.4f}")
    print(f"    R-squared:  {reg2['r_squared']:.4f}")
    print(f"    Std Error:  {reg2['se']:.2f} BCF")
    print(f"    n:          {reg2['n']}")

# Model 3: Withdrawal season only - HDD
print(f"\n  MODEL 3: storage_change = a + b*HDD  (WITHDRAWAL ONLY)")
x = [d["national_hdd"] for d in withdrawal]
y = [d["storage_change_bcf"] for d in withdrawal]
reg_w = simple_regression(x, y)
if reg_w:
    print(f"    Intercept:  {reg_w['intercept']:.4f}")
    print(f"    Slope:      {reg_w['slope']:.4f}")
    print(f"    R-squared:  {reg_w['r_squared']:.4f}")
    print(f"    Std Error:  {reg_w['se']:.2f} BCF")
    print(f"    n:          {reg_w['n']}")

# Model 4: Injection season only - CDD
print(f"\n  MODEL 4: storage_change = a + b*CDD  (INJECTION ONLY)")
x = [d["national_cdd"] for d in injection]
y = [d["storage_change_bcf"] for d in injection]
reg_i = simple_regression(x, y)
if reg_i:
    print(f"    Intercept:  {reg_i['intercept']:.4f}")
    print(f"    Slope:      {reg_i['slope']:.4f}")
    print(f"    R-squared:  {reg_i['r_squared']:.4f}")
    print(f"    Std Error:  {reg_i['se']:.2f} BCF")
    print(f"    n:          {reg_i['n']}")

# Model 5: Injection season - HDD + CDD (shoulder months have both)
print(f"\n  MODEL 5: storage_change = a + b1*HDD + b2*CDD  (INJECTION ONLY)")
x1 = [d["national_hdd"] for d in injection]
x2 = [d["national_cdd"] for d in injection]
y = [d["storage_change_bcf"] for d in injection]
reg_i2 = multi_regression_2var(x1, x2, y)
if reg_i2:
    print(f"    Intercept:  {reg_i2['intercept']:.4f}")
    print(f"    b_HDD:      {reg_i2['b_x1']:.4f}")
    print(f"    b_CDD:      {reg_i2['b_x2']:.4f}")
    print(f"    R-squared:  {reg_i2['r_squared']:.4f}")
    print(f"    Std Error:  {reg_i2['se']:.2f} BCF")
    print(f"    n:          {reg_i2['n']}")

# Model 6: Prior week storage change as momentum factor
print(f"\n  MODEL 6: storage_change = a + b1*HDD + b2*CDD + b3*prior_change  (ALL)")
# Build lagged change
data_with_lag = []
for i in range(1, len(data)):
    d = dict(data[i])
    d["prior_change"] = data[i-1]["storage_change_bcf"]
    data_with_lag.append(d)

x1 = [d["national_hdd"] for d in data_with_lag]
x2 = [d["national_cdd"] for d in data_with_lag]
x3 = [d["prior_change"] for d in data_with_lag]
y = [d["storage_change_bcf"] for d in data_with_lag]
reg6 = multi_regression_3var(x1, x2, x3, y)
if reg6:
    print(f"    Intercept:      {reg6['intercept']:.4f}")
    print(f"    b_HDD:          {reg6['b_x1']:.4f}")
    print(f"    b_CDD:          {reg6['b_x2']:.4f}")
    print(f"    b_prior_change: {reg6['b_x3']:.4f}")
    print(f"    R-squared:      {reg6['r_squared']:.4f}")
    print(f"    Std Error:      {reg6['se']:.2f} BCF")
    print(f"    n:              {reg6['n']}")


# ============================================================
# ANALYSIS 3: BUILD PREDICTIONS & CALCULATE SURPRISES
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 3: MODEL PREDICTIONS & SURPRISE CALCULATION")
print(f"{'=' * 70}")

# Use the best seasonal models: withdrawal = HDD only, injection = HDD+CDD
# This is our "consensus replacement" model

# Also build an all-season model (Model 2: HDD + CDD) for comparison
predictions = []
for i, d in enumerate(data):
    pred = {}
    pred["report_date"] = d["report_date"]
    pred["actual_change"] = d["storage_change_bcf"]
    pred["season"] = d["season"]
    pred["ng_return_1d"] = d["ng_return_1d"]
    pred["ng_return_5d"] = d["ng_return_5d"]
    pred["ng_close"] = d["ng_close_report_day"]
    pred["national_hdd"] = d["national_hdd"]
    pred["national_cdd"] = d["national_cdd"]
    pred["storage_level"] = d["storage_level_bcf"]
    
    # All-season model (Model 2)
    if reg2 and d["national_hdd"] is not None and d["national_cdd"] is not None:
        pred["pred_all_season"] = (reg2["intercept"] + 
                                    reg2["b_x1"] * d["national_hdd"] + 
                                    reg2["b_x2"] * d["national_cdd"])
        if d["storage_change_bcf"] is not None:
            pred["surprise_all"] = d["storage_change_bcf"] - pred["pred_all_season"]
    
    # Seasonal model
    if d["season"] == "WITHDRAWAL" and reg_w and d["national_hdd"] is not None:
        pred["pred_seasonal"] = reg_w["intercept"] + reg_w["slope"] * d["national_hdd"]
        if d["storage_change_bcf"] is not None:
            pred["surprise_seasonal"] = d["storage_change_bcf"] - pred["pred_seasonal"]
    elif d["season"] == "INJECTION" and reg_i2 and d["national_hdd"] is not None and d["national_cdd"] is not None:
        pred["pred_seasonal"] = (reg_i2["intercept"] + 
                                  reg_i2["b_x1"] * d["national_hdd"] + 
                                  reg_i2["b_x2"] * d["national_cdd"])
        if d["storage_change_bcf"] is not None:
            pred["surprise_seasonal"] = d["storage_change_bcf"] - pred["pred_seasonal"]
    
    predictions.append(pred)

# Surprise statistics
all_surprises = [p["surprise_seasonal"] for p in predictions if "surprise_seasonal" in p]
print(f"\n  SEASONAL MODEL SURPRISE STATS:")
print(f"    Mean surprise:    {mean(all_surprises):.2f} BCF")
print(f"    Std dev:          {stdev(all_surprises):.2f} BCF")
print(f"    Min:              {min(all_surprises):.1f} BCF")
print(f"    Max:              {max(all_surprises):.1f} BCF")
print(f"    25th percentile:  {percentile(all_surprises, 25):.1f} BCF")
print(f"    75th percentile:  {percentile(all_surprises, 75):.1f} BCF")

# By season
w_surp = [p["surprise_seasonal"] for p in predictions if "surprise_seasonal" in p and p["season"] == "WITHDRAWAL"]
i_surp = [p["surprise_seasonal"] for p in predictions if "surprise_seasonal" in p and p["season"] == "INJECTION"]
print(f"\n    Withdrawal season: mean={mean(w_surp):.2f}, std={stdev(w_surp):.2f}, n={len(w_surp)}")
print(f"    Injection season:  mean={mean(i_surp):.2f}, std={stdev(i_surp):.2f}, n={len(i_surp)}")


# ============================================================
# ANALYSIS 4: SURPRISE vs PRICE REACTION
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 4: SURPRISE vs PRICE REACTION")
print(f"{'=' * 70}")

# Core question: does surprise direction predict price direction?

# 4a: Correlation of surprise with next-day return
surp_x = [p.get("surprise_seasonal") for p in predictions]
ret_1d = [p.get("ng_return_1d") for p in predictions]
ret_5d = [p.get("ng_return_5d") for p in predictions]

r, t, n = correlation(surp_x, ret_1d)
print(f"\n  Surprise vs 1-day return:  r={r:.4f}, t={t:.2f}, n={n}")
r5, t5, n5 = correlation(surp_x, ret_5d)
print(f"  Surprise vs 5-day return:  r={r5:.4f}, t={t5:.2f}, n={n5}")

# 4b: Quintile analysis - sort by surprise magnitude, check returns
valid_preds = [p for p in predictions 
               if "surprise_seasonal" in p and p["ng_return_1d"] is not None]
valid_preds.sort(key=lambda x: x["surprise_seasonal"])

q_size = len(valid_preds) // 5
quintiles = []
for i in range(5):
    start = i * q_size
    end = start + q_size if i < 4 else len(valid_preds)
    q = valid_preds[start:end]
    quintiles.append(q)

print(f"\n  QUINTILE ANALYSIS (sorted by surprise, Q1=most bearish/largest draw surprise):")
print(f"  {'Quintile':<12s} {'n':>4s} {'Avg Surprise':>14s} {'Avg 1d Ret%':>14s} {'Avg 5d Ret%':>14s} {'1d Win%':>10s}")
print(f"  {'-'*72}")

for i, q in enumerate(quintiles):
    surps = [p["surprise_seasonal"] for p in q]
    rets_1d = [p["ng_return_1d"] for p in q if p["ng_return_1d"] is not None]
    rets_5d = [p["ng_return_5d"] for p in q if p["ng_return_5d"] is not None]
    
    avg_surp = mean(surps)
    avg_1d = mean(rets_1d)
    avg_5d = mean(rets_5d)
    
    # For bearish surprise (larger than expected draw), price should go UP
    # For bullish surprise (smaller draw / larger injection), price should go DOWN
    # So negative surprise → positive return expected
    # Win = surprise negative AND return positive, OR surprise positive AND return negative
    wins_1d = sum(1 for p in q if p["ng_return_1d"] is not None and 
                  ((p["surprise_seasonal"] < 0 and p["ng_return_1d"] > 0) or
                   (p["surprise_seasonal"] > 0 and p["ng_return_1d"] < 0)))
    win_pct = (wins_1d / len(rets_1d) * 100) if rets_1d else 0
    
    label = f"Q{i+1}"
    if i == 0:
        label += " (bearish)"
    elif i == 4:
        label += " (bullish)"
    
    print(f"  {label:<12s} {len(q):>4d} {avg_surp:>14.1f} {avg_1d:>14.2f} {avg_5d:>14.2f} {win_pct:>9.1f}%")

# 4c: Binary signal analysis
print(f"\n  BINARY SIGNAL: Bearish surprise (actual draw > predicted) → go LONG")
print(f"  Logic: Larger-than-expected draw = supply tighter than market thinks = bullish")

bearish = [p for p in valid_preds if p["surprise_seasonal"] < 0]
bullish = [p for p in valid_preds if p["surprise_seasonal"] > 0]
neutral = [p for p in valid_preds if p["surprise_seasonal"] == 0]

bear_1d = [p["ng_return_1d"] for p in bearish if p["ng_return_1d"] is not None]
bull_1d = [p["ng_return_1d"] for p in bullish if p["ng_return_1d"] is not None]
bear_5d = [p["ng_return_5d"] for p in bearish if p["ng_return_5d"] is not None]
bull_5d = [p["ng_return_5d"] for p in bullish if p["ng_return_5d"] is not None]

print(f"\n  {'Signal':<25s} {'n':>5s} {'Avg 1d%':>10s} {'Avg 5d%':>10s} {'1d Win%':>10s} {'5d Win%':>10s}")
print(f"  {'-'*65}")

bear_win_1d = sum(1 for r in bear_1d if r > 0) / len(bear_1d) * 100 if bear_1d else 0
bear_win_5d = sum(1 for r in bear_5d if r > 0) / len(bear_5d) * 100 if bear_5d else 0
bull_win_1d = sum(1 for r in bull_1d if r < 0) / len(bull_1d) * 100 if bull_1d else 0
bull_win_5d = sum(1 for r in bull_5d if r < 0) / len(bull_5d) * 100 if bull_5d else 0

print(f"  {'Bearish surprise (LONG)':<25s} {len(bearish):>5d} {mean(bear_1d):>10.2f} {mean(bear_5d):>10.2f} {bear_win_1d:>9.1f}% {bear_win_5d:>9.1f}%")
print(f"  {'Bullish surprise (SHORT)':<25s} {len(bullish):>5d} {mean(bull_1d):>10.2f} {mean(bull_5d):>10.2f} {bull_win_1d:>9.1f}% {bull_win_5d:>9.1f}%")

diff_1d, t_1d, n1, n2 = t_test_two_sample(bear_1d, bull_1d)
diff_5d, t_5d, _, _ = t_test_two_sample(bear_5d, bull_5d)
print(f"\n  T-test (bearish vs bullish 1d returns): diff={diff_1d:.2f}%, t={t_1d:.2f}")
print(f"  T-test (bearish vs bullish 5d returns): diff={diff_5d:.2f}%, t={t_5d:.2f}")


# ============================================================
# ANALYSIS 5: LARGE SURPRISE FILTER
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 5: LARGE SURPRISE FILTER (>1 std dev)")
print(f"{'=' * 70}")

surp_std = stdev(all_surprises)
surp_mean = mean(all_surprises)

large_bearish = [p for p in valid_preds if "surprise_seasonal" in p and p["surprise_seasonal"] < surp_mean - surp_std]
large_bullish = [p for p in valid_preds if "surprise_seasonal" in p and p["surprise_seasonal"] > surp_mean + surp_std]

print(f"\n  Surprise std dev: {surp_std:.1f} BCF")
print(f"  Large bearish threshold: < {surp_mean - surp_std:.1f} BCF")
print(f"  Large bullish threshold: > {surp_mean + surp_std:.1f} BCF")

lb_1d = [p["ng_return_1d"] for p in large_bearish if p["ng_return_1d"] is not None]
lb_5d = [p["ng_return_5d"] for p in large_bearish if p["ng_return_5d"] is not None]
lu_1d = [p["ng_return_1d"] for p in large_bullish if p["ng_return_1d"] is not None]
lu_5d = [p["ng_return_5d"] for p in large_bullish if p["ng_return_5d"] is not None]

print(f"\n  {'Signal':<30s} {'n':>5s} {'Avg 1d%':>10s} {'Avg 5d%':>10s} {'1d Win%':>10s}")
print(f"  {'-'*60}")

if lb_1d:
    lb_win = sum(1 for r in lb_1d if r > 0) / len(lb_1d) * 100
    print(f"  {'Large bearish (LONG)':<30s} {len(large_bearish):>5d} {mean(lb_1d):>10.2f} {mean(lb_5d):>10.2f} {lb_win:>9.1f}%")

if lu_1d:
    lu_win = sum(1 for r in lu_1d if r < 0) / len(lu_1d) * 100
    print(f"  {'Large bullish (SHORT)':<30s} {len(large_bullish):>5d} {mean(lu_1d):>10.2f} {mean(lu_5d):>10.2f} {lu_win:>9.1f}%")

# Also test 1.5 std and 2 std
for threshold_mult in [1.5, 2.0]:
    lb = [p for p in valid_preds if "surprise_seasonal" in p and p["surprise_seasonal"] < surp_mean - threshold_mult * surp_std]
    lu = [p for p in valid_preds if "surprise_seasonal" in p and p["surprise_seasonal"] > surp_mean + threshold_mult * surp_std]
    
    lb_r = [p["ng_return_1d"] for p in lb if p["ng_return_1d"] is not None]
    lu_r = [p["ng_return_1d"] for p in lu if p["ng_return_1d"] is not None]
    lb_r5 = [p["ng_return_5d"] for p in lb if p["ng_return_5d"] is not None]
    lu_r5 = [p["ng_return_5d"] for p in lu if p["ng_return_5d"] is not None]
    
    print(f"\n  At {threshold_mult}x std dev ({threshold_mult * surp_std:.0f} BCF threshold):")
    if lb_r:
        w = sum(1 for r in lb_r if r > 0) / len(lb_r) * 100
        print(f"    Large bearish LONG:  n={len(lb):>4d}, avg_1d={mean(lb_r):>7.2f}%, avg_5d={mean(lb_r5):>7.2f}%, win={w:.0f}%")
    if lu_r:
        w = sum(1 for r in lu_r if r < 0) / len(lu_r) * 100
        print(f"    Large bullish SHORT: n={len(lu):>4d}, avg_1d={mean(lu_r):>7.2f}%, avg_5d={mean(lu_r5):>7.2f}%, win={w:.0f}%")


# ============================================================
# ANALYSIS 6: SEASONAL DEEP DIVE
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 6: MONTHLY BREAKDOWN")
print(f"{'=' * 70}")

monthly = defaultdict(list)
for p in predictions:
    if "surprise_seasonal" in p and p["ng_return_1d"] is not None:
        month = int(p["report_date"][5:7])
        monthly[month].append(p)

print(f"\n  {'Month':<8s} {'n':>4s} {'Avg Surprise':>14s} {'Surprise Std':>14s} {'Avg 1d Ret':>12s} {'Corr(S,R)':>10s}")
print(f"  {'-'*66}")

for m in range(1, 13):
    if m not in monthly:
        continue
    ps = monthly[m]
    surps = [p["surprise_seasonal"] for p in ps]
    rets = [p["ng_return_1d"] for p in ps]
    r, _, _ = correlation(surps, rets)
    r_str = f"{r:.3f}" if r is not None else "N/A"
    
    month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m]
    print(f"  {month_name:<8s} {len(ps):>4d} {mean(surps):>14.1f} {stdev(surps):>14.1f} {mean(rets):>12.2f} {r_str:>10s}")


# ============================================================
# ANALYSIS 7: STORAGE LEVEL CONTEXT
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 7: STORAGE LEVEL CONTEXT")
print(f"{'=' * 70}")
print("  Does the price reaction depend on HOW FULL storage is?")

# Split by storage level quartiles
storage_levels = [p["storage_level"] for p in valid_preds if p["storage_level"] is not None]
q25 = percentile(storage_levels, 25)
q50 = percentile(storage_levels, 50)
q75 = percentile(storage_levels, 75)

level_groups = {
    f"Low (<{q25:.0f} BCF)": [p for p in valid_preds if p["storage_level"] and p["storage_level"] < q25],
    f"Med-Low ({q25:.0f}-{q50:.0f})": [p for p in valid_preds if p["storage_level"] and q25 <= p["storage_level"] < q50],
    f"Med-High ({q50:.0f}-{q75:.0f})": [p for p in valid_preds if p["storage_level"] and q50 <= p["storage_level"] < q75],
    f"High (>{q75:.0f} BCF)": [p for p in valid_preds if p["storage_level"] and p["storage_level"] >= q75],
}

print(f"\n  {'Storage Level':<25s} {'n':>4s} {'Corr(S,1d)':>12s} {'Bearish 1d':>12s} {'Bullish 1d':>12s}")
print(f"  {'-'*60}")

for label, group in level_groups.items():
    surps = [p["surprise_seasonal"] for p in group if "surprise_seasonal" in p]
    rets = [p["ng_return_1d"] for p in group if p["ng_return_1d"] is not None]
    r, _, _ = correlation(surps, rets)
    
    bear = [p["ng_return_1d"] for p in group if "surprise_seasonal" in p and p["surprise_seasonal"] < 0 and p["ng_return_1d"] is not None]
    bull = [p["ng_return_1d"] for p in group if "surprise_seasonal" in p and p["surprise_seasonal"] > 0 and p["ng_return_1d"] is not None]
    
    r_str = f"{r:.3f}" if r is not None else "N/A"
    bear_str = f"{mean(bear):.2f}%" if bear else "N/A"
    bull_str = f"{mean(bull):.2f}%" if bull else "N/A"
    
    print(f"  {label:<25s} {len(group):>4d} {r_str:>12s} {bear_str:>12s} {bull_str:>12s}")


# ============================================================
# ANALYSIS 8: YEAR-OVER-YEAR STABILITY
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 8: YEARLY STABILITY (is the signal consistent?)")
print(f"{'=' * 70}")

yearly = defaultdict(list)
for p in valid_preds:
    if "surprise_seasonal" in p:
        year = int(p["report_date"][:4])
        yearly[year].append(p)

print(f"\n  {'Year':<6s} {'n':>4s} {'R-sq Model':>12s} {'Avg Surprise':>14s} {'Surp Std':>10s} {'Avg 1d%':>10s} {'Corr(S,R)':>10s}")
print(f"  {'-'*70}")

for year in sorted(yearly.keys()):
    ps = yearly[year]
    surps = [p["surprise_seasonal"] for p in ps]
    rets = [p["ng_return_1d"] for p in ps if p["ng_return_1d"] is not None]
    
    # In-sample R-squared for this year
    hdd = [p["national_hdd"] for p in ps]
    chg = [p["actual_change"] for p in ps]
    yr_reg = simple_regression(hdd, chg)
    rsq_str = f"{yr_reg['r_squared']:.3f}" if yr_reg else "N/A"
    
    r, _, _ = correlation(surps, [p["ng_return_1d"] for p in ps])
    r_str = f"{r:.3f}" if r is not None else "N/A"
    
    print(f"  {year:<6d} {len(ps):>4d} {rsq_str:>12s} {mean(surps):>14.1f} {stdev(surps):>10.1f} {mean(rets):>10.2f} {r_str:>10s}")


# ============================================================
# ANALYSIS 9: OUT-OF-SAMPLE WALK-FORWARD TEST
# ============================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 9: WALK-FORWARD OUT-OF-SAMPLE TEST")
print(f"{'=' * 70}")
print("  Training on first 3 years, predicting next week, rolling forward")

# Need at least 3 years of training data
train_min_weeks = 156  # ~3 years

oos_predictions = []

for i in range(train_min_weeks, len(data)):
    d = data[i]
    if d["national_hdd"] is None or d["storage_change_bcf"] is None:
        continue
    
    # Training set: all prior data in same season
    train = [data[j] for j in range(i) 
             if data[j]["season"] == d["season"]
             and data[j]["national_hdd"] is not None 
             and data[j]["storage_change_bcf"] is not None]
    
    if len(train) < 20:
        continue
    
    # Fit regression on training data
    if d["season"] == "WITHDRAWAL":
        x_train = [t["national_hdd"] for t in train]
        y_train = [t["storage_change_bcf"] for t in train]
        reg_oos = simple_regression(x_train, y_train)
        if reg_oos:
            pred_val = reg_oos["intercept"] + reg_oos["slope"] * d["national_hdd"]
    else:
        x1_train = [t["national_hdd"] for t in train]
        x2_train = [t["national_cdd"] for t in train]
        y_train = [t["storage_change_bcf"] for t in train]
        reg_oos = multi_regression_2var(x1_train, x2_train, y_train)
        if reg_oos and d["national_cdd"] is not None:
            pred_val = (reg_oos["intercept"] + 
                       reg_oos["b_x1"] * d["national_hdd"] + 
                       reg_oos["b_x2"] * d["national_cdd"])
        else:
            continue
    
    if reg_oos is None:
        continue
    
    surprise_oos = d["storage_change_bcf"] - pred_val
    
    oos_predictions.append({
        "report_date": d["report_date"],
        "actual": d["storage_change_bcf"],
        "predicted": pred_val,
        "surprise": surprise_oos,
        "ng_return_1d": d["ng_return_1d"],
        "ng_return_5d": d["ng_return_5d"],
        "season": d["season"],
    })

print(f"\n  Out-of-sample predictions: {len(oos_predictions)}")

oos_surp = [p["surprise"] for p in oos_predictions]
print(f"  OOS Surprise: mean={mean(oos_surp):.2f}, std={stdev(oos_surp):.2f}")

# OOS prediction accuracy
oos_errors = [abs(p["surprise"]) for p in oos_predictions]
print(f"  OOS Mean Absolute Error: {mean(oos_errors):.1f} BCF")
print(f"  OOS Median Absolute Error: {percentile(oos_errors, 50):.1f} BCF")

# OOS surprise vs return correlation
oos_s = [p["surprise"] for p in oos_predictions]
oos_r1 = [p["ng_return_1d"] for p in oos_predictions]
oos_r5 = [p["ng_return_5d"] for p in oos_predictions]

r_oos1, t_oos1, n_oos1 = correlation(oos_s, oos_r1)
r_oos5, t_oos5, n_oos5 = correlation(oos_s, oos_r5)
print(f"\n  OOS Surprise vs 1d return: r={r_oos1:.4f}, t={t_oos1:.2f}, n={n_oos1}")
print(f"  OOS Surprise vs 5d return: r={r_oos5:.4f}, t={t_oos5:.2f}, n={n_oos5}")

# OOS binary signal
oos_bear = [p for p in oos_predictions if p["surprise"] < 0]
oos_bull = [p for p in oos_predictions if p["surprise"] > 0]

oos_bear_1d = [p["ng_return_1d"] for p in oos_bear if p["ng_return_1d"] is not None]
oos_bull_1d = [p["ng_return_1d"] for p in oos_bull if p["ng_return_1d"] is not None]

if oos_bear_1d and oos_bull_1d:
    bear_w = sum(1 for r in oos_bear_1d if r > 0) / len(oos_bear_1d) * 100
    bull_w = sum(1 for r in oos_bull_1d if r < 0) / len(oos_bull_1d) * 100
    
    print(f"\n  OOS BINARY SIGNAL:")
    print(f"    Bearish surprise LONG:  n={len(oos_bear)}, avg_1d={mean(oos_bear_1d):.2f}%, win={bear_w:.1f}%")
    print(f"    Bullish surprise SHORT: n={len(oos_bull)}, avg_1d={mean(oos_bull_1d):.2f}%, win={bull_w:.1f}%")
    
    diff, t_val, _, _ = t_test_two_sample(oos_bear_1d, oos_bull_1d)
    print(f"    T-test: diff={diff:.2f}%, t={t_val:.2f}")


# ============================================================
# EXPORT PREDICTIONS
# ============================================================
print(f"\n{'=' * 70}")
print("EXPORTING RESULTS")
print(f"{'=' * 70}")

# Export in-sample predictions
with open(f"{RESULTS_DIR}/predictions_in_sample.csv", "w") as f:
    cols = ["report_date", "season", "actual_change", "national_hdd", "national_cdd",
            "pred_all_season", "surprise_all", "pred_seasonal", "surprise_seasonal",
            "storage_level", "ng_close", "ng_return_1d", "ng_return_5d"]
    f.write(",".join(cols) + "\n")
    for p in predictions:
        vals = [str(p.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")
print(f"  In-sample predictions: {RESULTS_DIR}/predictions_in_sample.csv")

# Export OOS predictions
with open(f"{RESULTS_DIR}/predictions_oos.csv", "w") as f:
    cols = ["report_date", "season", "actual", "predicted", "surprise", 
            "ng_return_1d", "ng_return_5d"]
    f.write(",".join(cols) + "\n")
    for p in oos_predictions:
        vals = [str(p.get(c, "")) for c in cols]
        f.write(",".join(vals) + "\n")
print(f"  Out-of-sample predictions: {RESULTS_DIR}/predictions_oos.csv")


# ============================================================
# FINAL ASSESSMENT
# ============================================================
print(f"\n{'=' * 70}")
print("FINAL ASSESSMENT")
print(f"{'=' * 70}")

# Compute display values
rsq_all = f"{reg2['r_squared']:.4f}" if reg2 else "N/A"
rsq_w = f"{reg_w['r_squared']:.4f}" if reg_w else "N/A"
rsq_i = f"{reg_i2['r_squared']:.4f}" if reg_i2 else "N/A"
se_all = f"{reg2['se']:.1f}" if reg2 else "N/A"
corr_is = f"{r:.4f}" if r else "N/A"
corr_oos = f"{r_oos1:.4f}" if r_oos1 else "N/A"

print(f"""
  KEY FINDINGS:
  
  1. WEATHER-STORAGE RELATIONSHIP:
     - All-season R-squared (HDD+CDD):  {rsq_all}
     - Withdrawal R-squared (HDD only): {rsq_w}
     - Injection R-squared (HDD+CDD):   {rsq_i}
     - Model std error:                 {se_all} BCF
  
  2. SURPRISE-PRICE SIGNAL:
     - In-sample corr (surprise vs 1d): {corr_is}
     - OOS corr (surprise vs 1d):       {corr_oos}
  
  3. INTERPRETATION:
     - R-sq > 0.7: Weather is a strong predictor of storage changes
     - R-sq 0.3-0.7: Moderate predictor, other factors matter
     - R-sq < 0.3: Weak predictor, model needs more features
     
     - |Corr(surprise, return)| > 0.15: Tradeable signal likely exists
     - |Corr| 0.05-0.15: Weak signal, may need filtering
     - |Corr| < 0.05: No reliable signal from weather alone
  
  NEXT STEPS (Phase 3):
  - If signal exists: Build trading strategy with entry/exit rules
  - If weak: Add features (regional weather, prior week momentum, 
    storage vs 5yr avg, season progression, NG price level)
  - If none: Weather alone insufficient, need actual consensus data
""")

conn.close()
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
