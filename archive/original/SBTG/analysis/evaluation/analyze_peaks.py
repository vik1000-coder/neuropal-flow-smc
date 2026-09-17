#!/usr/bin/env python3
"""
Analyze F1 Peak Lags

For each evaluation target (Cook structural, Leifer functional, monoamine
networks), identifies the time lag at which SBTG achieves the highest F1
score.  Reports both the discrete peak and a parabolic-interpolated
peak for finer temporal resolution.  Reads evaluation CSVs produced by
prepare_merged_results.py.

Usage:
    python analysis/evaluation/analyze_peaks.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path("merged_results/figures")
CSV_COOK = RESULTS_DIR / "eval_cook_C.csv"
CSV_LEIFER = RESULTS_DIR / "eval_leifer_C.csv"
CSV_MONO = RESULTS_DIR / "eval_monoamine_all_methods.csv"

def get_peak_stats(df, label, method="SBTG", time_col="time_s", metric_col="f1"):
    """
    Find discrete and interpolated peak.
    """
    # Filter
    subset = df[df['method'] == method].sort_values(time_col)
    if len(subset) < 3:
        return None
    
    x = subset[time_col].values
    y = subset[metric_col].values
    
    # Discrete Peak
    idx_max = np.argmax(y)
    t_peak_discrete = x[idx_max]
    y_peak_discrete = y[idx_max]
    
    # Parabolic Interpolation
    # Needs valid neighbors
    if 0 < idx_max < len(x) - 1:
        x0, x1, x2 = x[idx_max-1], x[idx_max], x[idx_max+1]
        y0, y1, y2 = y[idx_max-1], y[idx_max], y[idx_max+1]
        
        # Parabolic fit formulas for uniform grid (x1-x0 == x2-x1)
        # But our grid might be non-uniform? Assuming uniform for now (lags 1,2,3...)
        # General parabola: y = a(x-x1)^2 + b(x-x1) + c
        # Vertex at x = x1 - b/(2a)
        
        # Using numerical approximation derived from 3 points:
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        C = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
        
        # Peak location x = -B / (2A)
        # Check if A < 0 (concave down)
        if A < 0:
            t_peak_interp = -B / (2 * A)
            y_peak_interp = A * t_peak_interp**2 + B * t_peak_interp + C
        else:
            # Not a peak (valley), fallback
            t_peak_interp = t_peak_discrete
            y_peak_interp = y_peak_discrete
    else:
        # Edge case
        t_peak_interp = t_peak_discrete
        y_peak_interp = y_peak_discrete
        
    return {
        "Network": label,
        "Method": method,
        "Peak Lag (s)": t_peak_discrete,
        "Peak F1": y_peak_discrete,
        "Interp Lag (s)": t_peak_interp,
        "Interp F1": y_peak_interp
    }

print(f"Loading data from {RESULTS_DIR}...")

results = []

# Monoamines
if CSV_MONO.exists():
    df_mono = pd.read_csv(CSV_MONO)
    transmitters = df_mono['transmitter'].unique()
    for trans in transmitters:
        # Filter for this transmitter
        df_trans = df_mono[df_mono['transmitter'] == trans]
        stats = get_peak_stats(df_trans, label=f"Monoamine: {trans}", method="SBTG", metric_col="f1")
        if stats:
            results.append(stats)

# Cook
if CSV_COOK.exists():
    df_cook = pd.read_csv(CSV_COOK)
    stats = get_peak_stats(df_cook, label="Connectome: Cook", method="SBTG", metric_col="f1")
    if stats:
        results.append(stats)

# Leifer
if CSV_LEIFER.exists():
    df_leifer = pd.read_csv(CSV_LEIFER)
    stats = get_peak_stats(df_leifer, label="Connectome: Leifer", method="SBTG", metric_col="f1")
    if stats:
        results.append(stats)

# Create DataFrame
summary = pd.DataFrame(results)
# Reorder cols
summary = summary[["Network", "Peak Lag (s)", "Interp Lag (s)", "Peak F1", "Interp F1"]]

# Save as CSV
csv_path = RESULTS_DIR / "peak_analysis.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSaved CSV to: {csv_path}")

# Save as LaTeX
tex_path = RESULTS_DIR / "peak_analysis.tex"
latex_table = summary.to_latex(
    index=False,
    float_format="%.2f",
    caption="Peak time lags for functional connectivity benchmarks. Discrete peaks are the lags with maximum sampled F1 score. Interpolated peaks are estimated using parabolic interpolation around the discrete maximum to find the sub-sample peak location.",
    label="tab:peak_lags",
    column_format="lrrrr"
)
with open(tex_path, "w") as f:
    f.write(latex_table)
print(f"Saved LaTeX to: {tex_path}")

print("\nPeak Analysis Summary (SBTG F1):")
print(summary.to_markdown(index=False, floatfmt=".3f"))
