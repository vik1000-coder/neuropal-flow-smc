#!/usr/bin/env python3
"""
Separate Evaluation of Chemical Synapses vs Gap Junctions

Loads the SBTG multi-lag mean-transfer matrices (result_C_merged.npz) and
evaluates them independently against the chemical-synapse and gap-junction
sub-networks of the Cook_Synapses_2019 structural benchmark.  Computes AUROC,
AUPRC, and best-threshold F1 for each sub-network at every available lag,
then generates comparison bar charts saved to merged_results/figures/.

Usage:
    python analysis/evaluation/analyze_chem_gap.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.align import align_matrices

RESULTS_DIR = Path("merged_results")
CONNECTOME_DIR = Path("results/intermediate/connectome")
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    res_file = RESULTS_DIR / "result_C_merged.npz"
    print(f"Loading results from {res_file}...")
    res_data = np.load(res_file, allow_pickle=True)

    print(f"Loading connectomes from {CONNECTOME_DIR}...")
    A_chem = np.load(CONNECTOME_DIR / "A_chem.npy")
    A_gap = np.load(CONNECTOME_DIR / "A_gap.npy")

    with open(CONNECTOME_DIR / "nodes.json") as f:
        nodes_conn = json.load(f)

    nodes_res = res_data['neuron_names'].tolist()

    return res_data, A_chem, A_gap, nodes_conn, nodes_res

def compute_metrics(y_true, y_score):
    """Compute AUROC, AUPRC, F1 (at best threshold)."""
    # Flatten
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    
    # Remove diagonal (self-loops) usually ignored in eval
    mask = ~np.eye(y_true.shape[0], dtype=bool).flatten()
    y_true_eval = y_true_flat[mask] > 0 # Binarize
    y_score_eval = y_score_flat[mask]
    
    if np.sum(y_true_eval) == 0:
        return np.nan, np.nan, np.nan

    auroc = roc_auc_score(y_true_eval, y_score_eval)
    auprc = average_precision_score(y_true_eval, y_score_eval)
    
    # F1 at best threshold (based on precision-recall curve)
    prec, rec, thresholds = precision_recall_curve(y_true_eval, y_score_eval)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_f1 = np.max(f1_scores)
    
    return auroc, auprc, best_f1

def main():
    res_data, A_chem_raw, A_gap_raw, nodes_conn, nodes_res = load_data()

    print("Aligning connectomes...")
    A_chem, common = align_matrices(A_chem_raw, nodes_conn, nodes_res)
    A_gap, _ = align_matrices(A_gap_raw, nodes_conn, nodes_res)

    lags = sorted([int(k.replace('mu_hat_lag', '')) for k in res_data.keys() if k.startswith('mu_hat_lag')])
    print(f"Found lags: {lags}")

    results = []

    for lag in lags:
        mu_hat = res_data[f'mu_hat_lag{lag}']
        mu_aligned, _ = align_matrices(mu_hat, nodes_res, common)
        score = np.abs(mu_aligned)

        auro_c, aupr_c, f1_c = compute_metrics(A_chem, score)
        auro_g, aupr_g, f1_g = compute_metrics(A_gap, score)
        
        results.append({
            'lag': lag,
            'time_s': lag * 0.25,
            'auroc_chem': auro_c, 'auprc_chem': aupr_c, 'f1_chem': f1_c,
            'auroc_gap': auro_g, 'auprc_gap': aupr_g, 'f1_gap': f1_g
        })
        print(f"Lag {lag}: Chem AUROC={auro_c:.3f}, Gap AUROC={auro_g:.3f}")
        
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "chem_gap_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Plotting
    metrics = ['auroc', 'auprc', 'f1']
    metric_names = ['AUROC', 'AUPRC', 'F1 Score (Max)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (met, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        ax.plot(df['time_s'], df[f'{met}_chem'], 'o-', label='Chemical', color='tab:blue')
        ax.plot(df['time_s'], df[f'{met}_gap'], 's--', label='Gap Junction', color='tab:orange')
        
        ax.set_title(name)
        ax.set_xlabel("Lag (s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "fig_chem_gap_performance.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    plt.close()

    # Dedicated F1 Plot (Separate lines)
    plt.figure(figsize=(8, 6))
    plt.plot(df['time_s'], df['f1_chem'], 'o-', label='Chemical', color='tab:blue', linewidth=2)
    plt.plot(df['time_s'], df['f1_gap'], 's--', label='Gap Junction', color='tab:orange', linewidth=2)
    plt.title("F1 Score vs Discrete Lag")
    plt.ylabel("F1 Score")
    plt.xlabel("Lag (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    f1_plot_path = OUTPUT_DIR / "fig_chem_gap_f1.png"
    plt.savefig(f1_plot_path, dpi=300)
    print(f"Saved dedicated F1 plot to {f1_plot_path}")
    plt.close()

    # --- Peak Analysis ---
    print("\n--- Peak Analysis (Chemical vs Gap) ---")
    
    def get_peak_stats(time_s, f1_scores):
        """Calculate discrete and interpolated peak."""
        # Discrete
        idx_max = np.argmax(f1_scores)
        t_max = time_s[idx_max]
        y_max = f1_scores[idx_max]
        
        # Interpolated
        if 0 < idx_max < len(time_s) - 1:
            y_prev = f1_scores[idx_max - 1]
            y_next = f1_scores[idx_max + 1]
            dt = time_s[1] - time_s[0] # Assumes uniform 0.25s
            
            denom = y_prev - 2*y_max + y_next
            if denom != 0:
                delta = 0.5 * (y_prev - y_next) / denom
                t_peak = t_max + delta * dt
                y_peak = y_max - 0.25 * (y_prev - y_next) * delta 
            else:
                t_peak = t_max
                y_peak = y_max
        else:
            t_peak = t_max
            y_peak = y_max
            
        return t_max, t_peak, y_max, y_peak

    # Extract arrays
    times = df['time_s'].values
    f1_chem = df['f1_chem'].values
    f1_gap = df['f1_gap'].values
    
    # Calculate
    tc_d, tc_i, yc_d, yc_i = get_peak_stats(times, f1_chem)
    tg_d, tg_i, yg_d, yg_i = get_peak_stats(times, f1_gap)
    
    print("-" * 60)
    print(f"{'Network':<20} | {'Peak (s)':<10} | {'Interp (s)':<10} | {'F1 (Max)':<10}")
    print("-" * 60)
    print(f"{'Chemical':<20} | {tc_d:<10.2f} | {tc_i:<10.2f} | {yc_d:<10.3f}")
    print(f"{'Gap Junction':<20} | {tg_d:<10.2f} | {tg_i:<10.2f} | {yg_d:<10.3f}")
    print("-" * 60)
    
    print("\nLatex Rows:")
    print(f"Chemical Synapses & {tc_d:.2f} & {tc_i:.2f} & {yc_d:.2f} & {yc_i:.2f} \\\\")
    print(f"Gap Junctions     & {tg_d:.2f} & {tg_i:.2f} & {yg_d:.2f} & {yg_i:.2f} \\\\")


if __name__ == "__main__":
    main()
