#!/usr/bin/env python3
"""
PHASE 2C: STORAGE DEVIATION DEEP DIVE
Is the storage-vs-5yr-avg signal genuine informational content,
or just generic price mean-reversion?

Tests:
  1. Price mean-reversion baseline (does NG price alone predict forward returns?)
  2. Storage deviation AFTER controlling for price (multivariate)
  3. Storage deviation velocity (getting worse vs improving)
  4. Seasonal timing (which months/transitions matter?)
  5. Storage regime (deficit vs surplus as separate signals)
  6. Combined signal: storage deviation + weather momentum
  7. Walk-forward OOS trading strategy
  8. Drawdown & risk analysis

Usage: python3 phase2c_storage_deviation.py
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
        t_stat = float('inf') if r > 0 else float('-inf')
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

def multi_reg(xs_list, y):
    """Multi-variable OLS using normal equations."""
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
    x_cols = [[row[j] for row in complete] for j in range(n_vars)]
    y_col = [row[n_vars] for row in complete]
    
    means_x = [mean(col) for col in x_cols]
    mean_y = mean(y_col)
    
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
    
    predictions = [intercept + sum(betas[j] * x_cols[j][k] for j in range(n_vars)) for k in range(n)]
    ss_res = sum((y_col[k] - predictions[k])**2 for k in range(n))
    ss_tot = sum((y_col[k] - mean_y)**2 for k in range(n))
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    se = math.sqrt(ss_res / (n - n_vars - 1)) if n > n_vars + 1 else None
    
    # Compute t-statistics for each beta
    # Need (X'X)^-1 diagonal elements
    # For simplicity, compute residual variance and approximate
    resid_var = ss_res / (n - n_vars - 1) if n > n_vars + 1 else None
    beta_t_stats = []
    if resid_var:
        # Invert S matrix for standard errors (S = X'X centered)
        # Use the already-row-reduced augmented matrix trick
        # Actually let's just do it properly with Gauss-Jordan
        inv_size = n_vars
        inv_aug = [[0]*(2*inv_size) for _ in range(inv_size)]
        for i in range(inv_size):
            for j in range(inv_size):
                inv_aug[i][j] = S[i][j]
            inv_aug[i][inv_size + i] = 1.0
        
        for i in range(inv_size):
            max_row = max(range(i, inv_size), key=lambda r: abs(inv_aug[r][i]))
            inv_aug[i], inv_aug[max_row] = inv_aug[max_row], inv_aug[i]
            if abs(inv_aug[i][i]) < 1e-12:
                beta_t_stats = [None] * n_vars
                break
            pivot = inv_aug[i][i]
            for k in range(2*inv_size):
                inv_aug[i][k] /= pivot
            for j in range(inv_size):
                if j != i:
                    factor = inv_aug[j][i]
                    for k in range(2*inv_size):
                        inv_aug[j][k] -= factor * inv_aug[i][k]
        
        if not beta_t_stats:
            for i in range(inv_size):
                diag = inv_aug[i][inv_size + i]
                se_beta = math.sqrt(resid_var * diag) if diag > 0 else None
                if se_beta and se_beta > 0:
                    beta_t_stats.append(betas[i] / se_beta)
                else:
                    beta_t_stats.append(None)
    else:
        beta_t_stats = [None] * n_vars
    
    return {
        "intercept": intercept, "betas": betas, "r_squared": r_sq, 
        "se": se, "n": n, "beta_t_stats": beta_t_stats,
        "predictions": predictions, "y_actual": y_col, 
        "x_cols": x_cols, "resid_var": resid_var
    }


# ============================================================
# LOAD & ENRICH DATA
# ============================================================
print("=" * 70)
print("PHASE 2C: STORAGE DEVIATION DEEP DIVE")
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

# Enrich with 5yr avg, forward returns, price features
def iso_week(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.isocalendar()[1]

year_week_change = {}
year_week_level = {}
for d in data:
    if d["storage_change_bcf"] is None:
        continue
    yr = int(d["report_date"][:4])
    wk = iso_week(d["report_date"])
    year_week_change[(yr, wk)] = d["storage_change_bcf"]
    year_week_level[(yr, wk)] = d["storage_level_bcf"]

enriched = []
for d in data:
    e = dict(d)
    yr = int(d["report_date"][:4])
    wk = iso_week(d["report_date"])
    
    prior_changes = []
    prior_levels = []
    for y_offset in range(1, 6):
        prior_yr = yr - y_offset
        for wk_offset in [0, -1, 1]:
            key = (prior_yr, wk + wk_offset)
            if key in year_week_change:
                prior_changes.append(year_week_change[key])
                prior_levels.append(year_week_level[key])
                break
    
    if len(prior_changes) >= 3:
        e["avg5yr_change"] = mean(prior_changes)
        e["avg5yr_level"] = mean(prior_levels)
        if d["storage_level_bcf"] is not None and e["avg5yr_level"] is not None:
            e["storage_vs_5yr"] = d["storage_level_bcf"] - e["avg5yr_level"]
            e["storage_vs_5yr_pct"] = (d["storage_level_bcf"] / e["avg5yr_level"] - 1) * 100
    
    e["month"] = int(d["report_date"][5:7])
    enriched.append(e)

# Build forward returns: 4w, 8w, 12w
for i in range(len(enriched)):
    close_now = enriched[i].get("ng_close_report_day")
    for weeks, label in [(4, "4w"), (8, "8w"), (12, "12w")]:
        if i + weeks < len(enriched) and close_now:
            close_fwd = enriched[i + weeks].get("ng_close_report_day")
            if close_fwd:
                enriched[i][f"ng_return_{label}"] = (close_fwd / close_now - 1) * 100

# Build trailing returns (for mean-reversion control)
for i in range(len(enriched)):
    close_now = enriched[i].get("ng_close_report_day")
    for weeks, label in [(4, "trail_4w"), (8, "trail_8w"), (12, "trail_12w")]:
        if i - weeks >= 0 and close_now:
            close_past = enriched[i - weeks].get("ng_close_report_day")
            if close_past:
                enriched[i][label] = (close_now / close_past - 1) * 100

# Storage deviation velocity (change in deviation over past 4 weeks)
for i in range(4, len(enriched)):
    sv_now = enriched[i].get("storage_vs_5yr_pct")
    sv_past = enriched[i-4].get("storage_vs_5yr_pct")
    if sv_now is not None and sv_past is not None:
        enriched[i]["sv_velocity_4w"] = sv_now - sv_past

# Filter to rows with storage deviation data
full = [e for e in enriched if "storage_vs_5yr_pct" in e and e.get("ng_return_4w") is not None]
print(f"Rows with full data: {len(full)}")
print(f"Date range: {full[0]['report_date']} to {full[-1]['report_date']}")


# ============================================================
# TEST 1: PRICE MEAN-REVERSION BASELINE
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 1: PRICE MEAN-REVERSION BASELINE")
print("  Does trailing NG price performance alone predict forward returns?")
print(f"{'=' * 70}")

for trail_label, fwd_label in [("trail_4w", "ng_return_4w"), ("trail_4w", "ng_return_8w"),
                                 ("trail_8w", "ng_return_4w"), ("trail_8w", "ng_return_8w"),
                                 ("trail_12w", "ng_return_8w")]:
    x = [e.get(trail_label) for e in full]
    y = [e.get(fwd_label) for e in full]
    r, t, n = correlation(x, y)
    if r is not None:
        print(f"  {trail_label:>10s} -> {fwd_label:<14s}: r={r:.4f}, t={t:.2f}, n={n}")

print(f"\n  STORAGE DEVIATION alone (for comparison):")
for fwd_label in ["ng_return_4w", "ng_return_8w", "ng_return_12w"]:
    x = [e.get("storage_vs_5yr_pct") for e in full]
    y = [e.get(fwd_label) for e in full]
    r, t, n = correlation(x, y)
    if r is not None:
        print(f"  storage_vs_5yr_pct -> {fwd_label:<14s}: r={r:.4f}, t={t:.2f}, n={n}")


# ============================================================
# TEST 2: STORAGE DEVIATION AFTER CONTROLLING FOR PRICE
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 2: MULTIVARIATE — STORAGE DEVIATION AFTER PRICE CONTROL")
print("  Regress forward return on BOTH trailing price AND storage deviation")
print("  If storage beta is significant, it has info beyond price reversion")
print(f"{'=' * 70}")

for fwd_label, trail_label in [("ng_return_4w", "trail_4w"), ("ng_return_8w", "trail_8w"),
                                 ("ng_return_8w", "trail_4w"), ("ng_return_12w", "trail_8w")]:
    x1 = [e.get(trail_label) for e in full]
    x2 = [e.get("storage_vs_5yr_pct") for e in full]
    y = [e.get(fwd_label) for e in full]
    
    result = multi_reg([x1, x2], y)
    if result:
        b_trail = result["betas"][0]
        b_storage = result["betas"][1]
        t_trail = result["beta_t_stats"][0] if result["beta_t_stats"][0] else 0
        t_storage = result["beta_t_stats"][1] if result["beta_t_stats"][1] else 0
        
        print(f"\n  {fwd_label} = a + b1*{trail_label} + b2*storage_vs_5yr_pct")
        print(f"    b_trail={b_trail:.4f} (t={t_trail:.2f}), b_storage={b_storage:.4f} (t={t_storage:.2f})")
        print(f"    R²={result['r_squared']:.4f}, n={result['n']}")
        
        if abs(t_storage) > 2.0:
            print(f"    *** STORAGE ADDS SIGNIFICANT INFO BEYOND PRICE REVERSION ***")
        elif abs(t_storage) > 1.5:
            print(f"    ~~ Storage marginally significant ~~")
        else:
            print(f"    Storage NOT significant after controlling for price")


# ============================================================
# TEST 3: STORAGE DEVIATION VELOCITY
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 3: STORAGE DEVIATION VELOCITY")
print("  Is the deficit/surplus GROWING or SHRINKING?")
print("  Velocity = change in storage_vs_5yr_pct over past 4 weeks")
print(f"{'=' * 70}")

vel_data = [e for e in full if e.get("sv_velocity_4w") is not None]
print(f"  Rows with velocity data: {len(vel_data)}")

# Velocity alone
for fwd_label in ["ng_return_4w", "ng_return_8w"]:
    x = [e.get("sv_velocity_4w") for e in vel_data]
    y = [e.get(fwd_label) for e in vel_data]
    r, t, n = correlation(x, y)
    if r is not None:
        print(f"\n  sv_velocity_4w -> {fwd_label}: r={r:.4f}, t={t:.2f}, n={n}")

# Velocity + level combined
print(f"\n  Multivariate: velocity + level + trailing price")
for fwd_label, trail_label in [("ng_return_8w", "trail_8w")]:
    x1 = [e.get("storage_vs_5yr_pct") for e in vel_data]
    x2 = [e.get("sv_velocity_4w") for e in vel_data]
    x3 = [e.get(trail_label) for e in vel_data]
    y = [e.get(fwd_label) for e in vel_data]
    
    result = multi_reg([x1, x2, x3], y)
    if result:
        print(f"\n  {fwd_label} = a + b1*level + b2*velocity + b3*{trail_label}")
        for j, name in enumerate(["storage_level", "velocity", "trail_price"]):
            b = result["betas"][j]
            t_val = result["beta_t_stats"][j] if result["beta_t_stats"][j] else 0
            sig = "***" if abs(t_val) > 2.6 else "**" if abs(t_val) > 2.0 else "*" if abs(t_val) > 1.5 else ""
            print(f"    b_{name}={b:.4f} (t={t_val:.2f}) {sig}")
        print(f"    R²={result['r_squared']:.4f}, n={result['n']}")


# ============================================================
# TEST 4: MONTHLY/SEASONAL TIMING
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 4: MONTHLY BREAKDOWN — WHEN DOES THE SIGNAL WORK?")
print(f"{'=' * 70}")

print(f"\n  {'Month':>5s} {'n':>5s} {'Corr(Dev,4wRet)':>16s} {'t-stat':>8s} {'Corr(Dev,8wRet)':>16s} {'t-stat':>8s}")
print(f"  {'-'*62}")

for month in range(1, 13):
    month_data = [e for e in full if e["month"] == month]
    if len(month_data) < 10:
        continue
    
    x = [e["storage_vs_5yr_pct"] for e in month_data]
    y4 = [e.get("ng_return_4w") for e in month_data]
    y8 = [e.get("ng_return_8w") for e in month_data]
    
    r4, t4, n4 = correlation(x, y4)
    r8, t8, n8 = correlation(x, y8)
    
    r4_str = f"{r4:.4f}" if r4 is not None else "N/A"
    t4_str = f"{t4:.2f}" if t4 is not None else "N/A"
    r8_str = f"{r8:.4f}" if r8 is not None else "N/A"
    t8_str = f"{t8:.2f}" if t8 is not None else "N/A"
    
    print(f"  {month:>5d} {len(month_data):>5d} {r4_str:>16s} {t4_str:>8s} {r8_str:>16s} {t8_str:>8s}")

# Transition periods: end of withdrawal (Mar-Apr) and start of withdrawal (Oct-Nov)
print(f"\n  TRANSITION PERIODS:")
for label, months in [("End withdrawal (Mar-Apr)", [3,4]), 
                       ("Peak injection (Jun-Aug)", [6,7,8]),
                       ("Start withdrawal (Oct-Nov)", [10,11]),
                       ("Peak withdrawal (Dec-Feb)", [12,1,2])]:
    trans_data = [e for e in full if e["month"] in months]
    if len(trans_data) < 20:
        continue
    
    x = [e["storage_vs_5yr_pct"] for e in trans_data]
    y4 = [e.get("ng_return_4w") for e in trans_data]
    y8 = [e.get("ng_return_8w") for e in trans_data]
    
    r4, t4, n4 = correlation(x, y4)
    r8, t8, n8 = correlation(x, y8)
    
    print(f"  {label:<35s}: n={len(trans_data):>3d}, 4w r={r4:.3f} t={t4:.2f}, 8w r={r8:.3f} t={t8:.2f}")


# ============================================================
# TEST 5: DEFICIT vs SURPLUS AS SEPARATE SIGNALS
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 5: DEFICIT vs SURPLUS — ASYMMETRIC EFFECTS?")
print("  Does below-average storage predict differently than above-average?")
print(f"{'=' * 70}")

deficit = [e for e in full if e["storage_vs_5yr_pct"] < 0]
surplus = [e for e in full if e["storage_vs_5yr_pct"] > 0]

print(f"\n  DEFICIT weeks (below 5yr avg): {len(deficit)}")
print(f"  SURPLUS weeks (above 5yr avg): {len(surplus)}")

for label, group in [("DEFICIT", deficit), ("SURPLUS", surplus)]:
    if len(group) < 20:
        continue
    
    x = [abs(e["storage_vs_5yr_pct"]) for e in group]  # magnitude of deviation
    
    for fwd in ["ng_return_4w", "ng_return_8w"]:
        if label == "DEFICIT":
            # Deficit: more negative deviation -> expect price INCREASE
            y = [e.get(fwd) for e in group]
        else:
            # Surplus: more positive deviation -> expect price DECREASE
            y = [-e.get(fwd) if e.get(fwd) is not None else None for e in group]
        
        r, t, n = correlation(x, y)
        if r is not None:
            print(f"  {label:>8s}: |deviation| vs {fwd}: r={r:.4f}, t={t:.2f}, n={n}")

# Quartile analysis within deficit and surplus separately
for label, group in [("DEFICIT", deficit), ("SURPLUS", surplus)]:
    if len(group) < 40:
        continue
    
    group_sorted = sorted(group, key=lambda e: e["storage_vs_5yr_pct"])
    half = len(group_sorted) // 2
    
    mild = group_sorted[half:] if label == "DEFICIT" else group_sorted[:half]
    severe = group_sorted[:half] if label == "DEFICIT" else group_sorted[half:]
    
    mild_4w = [e.get("ng_return_4w") for e in mild if e.get("ng_return_4w") is not None]
    severe_4w = [e.get("ng_return_4w") for e in severe if e.get("ng_return_4w") is not None]
    mild_8w = [e.get("ng_return_8w") for e in mild if e.get("ng_return_8w") is not None]
    severe_8w = [e.get("ng_return_8w") for e in severe if e.get("ng_return_8w") is not None]
    
    mild_dev = mean([e["storage_vs_5yr_pct"] for e in mild])
    severe_dev = mean([e["storage_vs_5yr_pct"] for e in severe])
    
    print(f"\n  {label} SPLIT:")
    print(f"    Mild (avg dev={mild_dev:.1f}%):   4w={mean(mild_4w):.2f}%, 8w={mean(mild_8w):.2f}%, n={len(mild)}")
    print(f"    Severe (avg dev={severe_dev:.1f}%): 4w={mean(severe_4w):.2f}%, 8w={mean(severe_8w):.2f}%, n={len(severe)}")
    
    diff4, t4, _, _ = t_test_two_sample(severe_4w, mild_4w)
    diff8, t8, _, _ = t_test_two_sample(severe_8w, mild_8w)
    if diff4 is not None:
        print(f"    T-test 4w: diff={diff4:.2f}%, t={t4:.2f}")
    if diff8 is not None:
        print(f"    T-test 8w: diff={diff8:.2f}%, t={t8:.2f}")


# ============================================================
# TEST 6: COMBINED SIGNAL — STORAGE + WEATHER MOMENTUM
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 6: COMBINED SIGNAL — STORAGE DEVIATION + WEATHER CONTEXT")
print("  Does knowing current weather trend improve the storage signal?")
print(f"{'=' * 70}")

# HDD anomaly: is it currently colder/warmer than the same week's 5yr avg?
# Build HDD 5yr avg by week
year_week_hdd = {}
for e in enriched:
    if e.get("national_hdd") is None:
        continue
    yr = int(e["report_date"][:4])
    wk = iso_week(e["report_date"])
    year_week_hdd[(yr, wk)] = e["national_hdd"]

for e in enriched:
    yr = int(e["report_date"][:4])
    wk = iso_week(e["report_date"])
    
    prior_hdds = []
    for y_offset in range(1, 6):
        prior_yr = yr - y_offset
        for wk_offset in [0, -1, 1]:
            key = (prior_yr, wk + wk_offset)
            if key in year_week_hdd:
                prior_hdds.append(year_week_hdd[key])
                break
    
    if len(prior_hdds) >= 3 and e.get("national_hdd") is not None:
        avg5yr_hdd = mean(prior_hdds)
        e["hdd_anomaly"] = e["national_hdd"] - avg5yr_hdd
        e["hdd_anomaly_pct"] = (e["national_hdd"] / avg5yr_hdd - 1) * 100 if avg5yr_hdd > 0 else 0

# Interaction: storage below avg AND colder than normal -> extra bullish
# storage above avg AND warmer than normal -> extra bearish
combo_data = [e for e in full if e.get("hdd_anomaly") is not None and e.get("storage_vs_5yr_pct") is not None]
print(f"\n  Rows with combined data: {len(combo_data)}")

# Create interaction term
for e in combo_data:
    # Both deficit + cold = "supply squeeze" signal
    e["squeeze_score"] = -e["storage_vs_5yr_pct"] + e.get("hdd_anomaly_pct", 0)
    # storage deficit (neg) becomes positive contribution, cold anomaly (positive) adds

# Test squeeze score
for fwd in ["ng_return_4w", "ng_return_8w"]:
    x = [e.get("squeeze_score") for e in combo_data]
    y = [e.get(fwd) for e in combo_data]
    r, t, n = correlation(x, y)
    if r is not None:
        print(f"  squeeze_score -> {fwd}: r={r:.4f}, t={t:.2f}, n={n}")

# Multivariate: storage dev + hdd anomaly + trailing price
print(f"\n  Multivariate: storage_dev + hdd_anomaly + trail_8w -> 8w return")
x1 = [e.get("storage_vs_5yr_pct") for e in combo_data]
x2 = [e.get("hdd_anomaly_pct") for e in combo_data]
x3 = [e.get("trail_8w") for e in combo_data]
y = [e.get("ng_return_8w") for e in combo_data]

result = multi_reg([x1, x2, x3], y)
if result:
    for j, name in enumerate(["storage_dev", "hdd_anomaly", "trail_8w"]):
        b = result["betas"][j]
        t_val = result["beta_t_stats"][j] if result["beta_t_stats"][j] else 0
        sig = "***" if abs(t_val) > 2.6 else "**" if abs(t_val) > 2.0 else "*" if abs(t_val) > 1.5 else ""
        print(f"    b_{name}={b:.4f} (t={t_val:.2f}) {sig}")
    print(f"    R²={result['r_squared']:.4f}, n={result['n']}")

# 4-way split: deficit/surplus x cold/warm
print(f"\n  4-WAY SPLIT: Storage x Weather")
print(f"  {'Regime':<30s} {'n':>5s} {'Avg 4w%':>10s} {'Avg 8w%':>10s}")
print(f"  {'-'*60}")

for s_label, s_filter in [("Deficit", lambda e: e["storage_vs_5yr_pct"] < -5),
                           ("Surplus", lambda e: e["storage_vs_5yr_pct"] > 5)]:
    for w_label, w_filter in [("+ Cold anomaly", lambda e: e.get("hdd_anomaly", 0) > 0),
                               ("+ Warm anomaly", lambda e: e.get("hdd_anomaly", 0) < 0)]:
        group = [e for e in combo_data if s_filter(e) and w_filter(e)]
        if len(group) < 10:
            continue
        
        r4 = [e.get("ng_return_4w") for e in group if e.get("ng_return_4w") is not None]
        r8 = [e.get("ng_return_8w") for e in group if e.get("ng_return_8w") is not None]
        
        regime = f"{s_label} {w_label}"
        r4_str = f"{mean(r4):.2f}" if r4 else "N/A"
        r8_str = f"{mean(r8):.2f}" if r8 else "N/A"
        print(f"  {regime:<30s} {len(group):>5d} {r4_str:>10s} {r8_str:>10s}")


# ============================================================
# TEST 7: WALK-FORWARD OOS TRADING STRATEGY
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 7: WALK-FORWARD OUT-OF-SAMPLE STRATEGY")
print("  Rules: Use storage deviation + trailing price to position")
print("  8-week holding period, rebalance weekly")
print(f"{'=' * 70}")

# Simple strategy:
# If storage_vs_5yr_pct < -X AND trail_8w > Y: LONG (deficit + prices already rose = more to go)
# Wait — based on the data, surplus -> positive returns. Let's just use the signal directly.

# Strategy: Go LONG NG when model predicts positive 8w return
# Training: expanding window regression of storage_dev + trail_price -> 8w return
# Position: predicted return > 0 -> long, < 0 -> short (or flat)

min_train = 156  # 3 years
oos_trades = []

for i in range(min_train, len(full)):
    current = full[i]
    
    if current.get("ng_return_8w") is None:
        continue
    if current.get("storage_vs_5yr_pct") is None or current.get("trail_8w") is None:
        continue
    
    # Training data: all prior weeks
    train = full[:i]
    train_valid = [t for t in train 
                   if t.get("storage_vs_5yr_pct") is not None 
                   and t.get("trail_8w") is not None
                   and t.get("ng_return_8w") is not None]
    
    if len(train_valid) < 80:
        continue
    
    x1_t = [t["storage_vs_5yr_pct"] for t in train_valid]
    x2_t = [t["trail_8w"] for t in train_valid]
    y_t = [t["ng_return_8w"] for t in train_valid]
    
    model = multi_reg([x1_t, x2_t], y_t)
    if model is None:
        continue
    
    # Predict
    pred_8w = model["intercept"] + model["betas"][0] * current["storage_vs_5yr_pct"] + model["betas"][1] * current["trail_8w"]
    
    actual_8w = current["ng_return_8w"]
    
    # Signal: long if pred > threshold, short if pred < -threshold
    threshold = 2.0  # only trade if expected return > 2%
    
    if pred_8w > threshold:
        position = "LONG"
    elif pred_8w < -threshold:
        position = "SHORT"
    else:
        position = "FLAT"
    
    trade_return = actual_8w if position == "LONG" else (-actual_8w if position == "SHORT" else 0)
    
    oos_trades.append({
        "report_date": current["report_date"],
        "pred_8w": pred_8w,
        "actual_8w": actual_8w,
        "position": position,
        "trade_return": trade_return,
        "storage_dev": current["storage_vs_5yr_pct"],
        "trail_8w": current["trail_8w"],
    })

print(f"\n  OOS trade observations: {len(oos_trades)}")
print(f"  Date range: {oos_trades[0]['report_date']} to {oos_trades[-1]['report_date']}")

# Position summary
for pos in ["LONG", "SHORT", "FLAT"]:
    trades = [t for t in oos_trades if t["position"] == pos]
    if not trades:
        continue
    
    rets = [t["trade_return"] for t in trades]
    wins = sum(1 for r in rets if r > 0)
    
    print(f"\n  {pos} positions: n={len(trades)}")
    print(f"    Avg return: {mean(rets):.2f}%")
    print(f"    Median return: {percentile(rets, 50):.2f}%")
    print(f"    Win rate: {wins/len(trades)*100:.1f}%")
    print(f"    Std dev: {stdev(rets):.2f}%")
    print(f"    Sharpe (8w): {mean(rets)/stdev(rets):.3f}" if stdev(rets) and stdev(rets) > 0 else "")
    print(f"    Best: {max(rets):.1f}%, Worst: {min(rets):.1f}%")

# All active trades
active = [t for t in oos_trades if t["position"] != "FLAT"]
if active:
    active_rets = [t["trade_return"] for t in active]
    active_wins = sum(1 for r in active_rets if r > 0)
    print(f"\n  ALL ACTIVE TRADES (LONG + SHORT): n={len(active)}")
    print(f"    Avg return: {mean(active_rets):.2f}%")
    print(f"    Win rate: {active_wins/len(active)*100:.1f}%")
    print(f"    Sharpe (8w): {mean(active_rets)/stdev(active_rets):.3f}" if stdev(active_rets) and stdev(active_rets) > 0 else "")

# Yearly breakdown of active trades
print(f"\n  YEARLY OOS PERFORMANCE (active trades only):")
print(f"  {'Year':>6s} {'n':>5s} {'Avg Ret%':>10s} {'Win%':>8s} {'Long':>6s} {'Short':>6s}")
print(f"  {'-'*45}")

years = sorted(set(int(t["report_date"][:4]) for t in active))
for yr in years:
    yr_trades = [t for t in active if int(t["report_date"][:4]) == yr]
    yr_rets = [t["trade_return"] for t in yr_trades]
    yr_wins = sum(1 for r in yr_rets if r > 0)
    yr_longs = sum(1 for t in yr_trades if t["position"] == "LONG")
    yr_shorts = sum(1 for t in yr_trades if t["position"] == "SHORT")
    
    print(f"  {yr:>6d} {len(yr_trades):>5d} {mean(yr_rets):>10.2f} {yr_wins/len(yr_trades)*100:>7.1f}% {yr_longs:>6d} {yr_shorts:>6d}")


# ============================================================
# TEST 8: DRAWDOWN & RISK ANALYSIS
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 8: DRAWDOWN & RISK ANALYSIS")
print(f"{'=' * 70}")

# Compute cumulative equity curve for active trades
# Since these are overlapping 8-week periods, we approximate
# by computing weekly contribution

# Non-overlapping approach: take every 8th trade
non_overlap = []
last_entry = None
for t in oos_trades:
    if t["position"] == "FLAT":
        continue
    entry_date = datetime.strptime(t["report_date"], "%Y-%m-%d")
    if last_entry is None or (entry_date - last_entry).days >= 56:
        non_overlap.append(t)
        last_entry = entry_date

if non_overlap:
    no_rets = [t["trade_return"] for t in non_overlap]
    no_wins = sum(1 for r in no_rets if r > 0)
    print(f"\n  NON-OVERLAPPING TRADES (8-week spacing): n={len(non_overlap)}")
    print(f"    Avg return per trade: {mean(no_rets):.2f}%")
    print(f"    Win rate: {no_wins/len(non_overlap)*100:.1f}%")
    print(f"    Std dev: {stdev(no_rets):.2f}%")
    if stdev(no_rets) and stdev(no_rets) > 0:
        # Annualize: ~6.5 non-overlapping 8-week periods per year
        ann_return = mean(no_rets) * 6.5
        ann_vol = stdev(no_rets) * math.sqrt(6.5)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        print(f"    Annualized return (approx): {ann_return:.1f}%")
        print(f"    Annualized vol (approx): {ann_vol:.1f}%")
        print(f"    Annualized Sharpe: {sharpe:.3f}")
    
    # Cumulative return
    cum = 100
    peak = 100
    max_dd = 0
    dd_start = None
    
    for t in non_overlap:
        cum *= (1 + t["trade_return"] / 100)
        if cum > peak:
            peak = cum
        dd = (cum / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd
            dd_date = t["report_date"]
    
    print(f"    Final equity: {cum:.1f} (starting 100)")
    print(f"    Max drawdown: {max_dd:.1f}%")
    if max_dd < 0:
        print(f"    Max DD date: {dd_date}")
    
    # Consecutive losses
    streak = 0
    max_streak = 0
    for r in no_rets:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"    Max consecutive losses: {max_streak}")

# Also test with different thresholds
print(f"\n  THRESHOLD SENSITIVITY (non-overlapping, 8-week trades):")
print(f"  {'Threshold':>10s} {'n':>5s} {'Avg Ret%':>10s} {'Win%':>8s} {'Sharpe':>8s}")
print(f"  {'-'*45}")

for thresh in [0, 1, 2, 3, 5, 8]:
    # Re-run with different threshold
    trades_t = []
    last_entry = None
    for t in oos_trades:
        pos = t["position"]
        # Override position based on threshold
        if t["pred_8w"] > thresh:
            pos = "LONG"
        elif t["pred_8w"] < -thresh:
            pos = "SHORT"
        else:
            pos = "FLAT"
        
        if pos == "FLAT":
            continue
        
        entry_date = datetime.strptime(t["report_date"], "%Y-%m-%d")
        if last_entry is None or (entry_date - last_entry).days >= 56:
            ret = t["actual_8w"] if pos == "LONG" else -t["actual_8w"]
            trades_t.append(ret)
            last_entry = entry_date
    
    if len(trades_t) > 5:
        avg_r = mean(trades_t)
        wins = sum(1 for r in trades_t if r > 0)
        std_r = stdev(trades_t)
        sharpe = (avg_r / std_r * math.sqrt(6.5)) if std_r and std_r > 0 else 0
        print(f"  {thresh:>10d} {len(trades_t):>5d} {avg_r:>10.2f} {wins/len(trades_t)*100:>7.1f}% {sharpe:>8.3f}")


# ============================================================
# TEST 9: CURRENT STATE ASSESSMENT
# ============================================================
print(f"\n{'=' * 70}")
print("TEST 9: CURRENT STATE — WHERE ARE WE NOW?")
print(f"{'=' * 70}")

latest = enriched[-1]
print(f"\n  Latest report date: {latest['report_date']}")
print(f"  Storage level: {latest.get('storage_level_bcf', 'N/A')} BCF")

if "avg5yr_level" in latest:
    print(f"  5yr avg level: {latest['avg5yr_level']:.0f} BCF")
if "storage_vs_5yr_pct" in latest:
    print(f"  Storage vs 5yr avg: {latest['storage_vs_5yr_pct']:.1f}%")
if "sv_velocity_4w" in latest:
    print(f"  Deviation velocity (4w): {latest.get('sv_velocity_4w', 'N/A')}")

print(f"  NG price: ${latest.get('ng_close_report_day', 'N/A')}")
if latest.get("trail_8w") is not None:
    print(f"  Trailing 8w return: {latest['trail_8w']:.1f}%")

# What would the model say now?
if "storage_vs_5yr_pct" in latest and latest.get("trail_8w") is not None:
    # Use full training set to predict
    train_all = [e for e in full if e.get("storage_vs_5yr_pct") is not None 
                 and e.get("trail_8w") is not None and e.get("ng_return_8w") is not None]
    
    x1 = [t["storage_vs_5yr_pct"] for t in train_all]
    x2 = [t["trail_8w"] for t in train_all]
    y = [t["ng_return_8w"] for t in train_all]
    
    model = multi_reg([x1, x2], y)
    if model:
        pred = model["intercept"] + model["betas"][0] * latest["storage_vs_5yr_pct"] + model["betas"][1] * latest["trail_8w"]
        print(f"\n  MODEL PREDICTION (8-week forward):")
        print(f"    Predicted return: {pred:.2f}%")
        if pred > 2:
            print(f"    Signal: LONG")
        elif pred < -2:
            print(f"    Signal: SHORT")
        else:
            print(f"    Signal: FLAT (below threshold)")


# ============================================================
# EXPORT
# ============================================================
print(f"\n{'=' * 70}")
print("EXPORTING")
print(f"{'=' * 70}")

# Export OOS trades
with open(f"{RESULTS_DIR}/oos_strategy_trades.csv", "w") as f:
    f.write("report_date,position,pred_8w,actual_8w,trade_return,storage_dev,trail_8w\n")
    for t in oos_trades:
        f.write(f"{t['report_date']},{t['position']},{t['pred_8w']:.2f},{t['actual_8w']:.2f},{t['trade_return']:.2f},{t['storage_dev']:.1f},{t['trail_8w']:.1f}\n")
print(f"  Strategy trades: {RESULTS_DIR}/oos_strategy_trades.csv ({len(oos_trades)} rows)")


# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'=' * 70}")
print("PHASE 2C — FINAL VERDICT")
print(f"{'=' * 70}")

print(f"""
  TEST 1: Price mean-reversion exists in NG (trailing returns predict forward)
  
  TEST 2: After controlling for price, storage deviation MAY add incremental
          information — check t-statistics above for significance.
  
  TEST 3: Velocity adds context — rapidly worsening deficits may strengthen signal.
  
  TEST 4: Monthly breakdown reveals WHEN the signal works best.
  
  TEST 5: Deficit and surplus may have asymmetric price impact.
  
  TEST 6: Weather anomaly + storage creates "supply squeeze" scoring.
  
  TEST 7: Walk-forward OOS strategy shows realistic trading performance.
  
  TEST 8: Drawdown analysis reveals practical risk.
  
  TEST 9: Current market positioning.
  
  KEY QUESTION ANSWERED:
  Is the storage signal just price mean-reversion?
  -> Look at Test 2's t-statistics for storage after price control.
  -> If |t| > 2.0: Storage has genuine informational content.
  -> If |t| < 1.5: It's mostly price mean-reversion in disguise.
""")

conn.close()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
