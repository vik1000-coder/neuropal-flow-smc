#!/usr/bin/env python3
"""
Synaptic vs Electrical (GJ) GABA Edge Recovery Analysis

Builds two GABA-focused reference sets from `connectome_syn_gj.csv`:
1) Chemical synapses with GABA source and GABA receptor targets
2) Electrical (gap-junction) edges with the same receptor filter, excluding
   source-target pairs already counted in the chemical set

Evaluates SBTG predictions against each set using AUROC, AUPRC, and F1 over lag.

Usage:
    python analysis/evaluation/analyze_synaptic_vs_nonsynaptic.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from pipeline.utils.display_names import GABA_LABEL

# Paths
RESULTS_DIR = Path("merged_results")
DATA_DIR = Path("data")
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    res_file = RESULTS_DIR / "result_C_merged.npz"
    print(f"Loading results from {res_file}...")
    res_data = np.load(res_file, allow_pickle=True)
    
    csv_file = DATA_DIR / "connectome_syn_gj.csv"
    print(f"Loading connectome from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    return res_data, df

def normalize_name(name):
    """Normalize neuron name by removing L/R suffix if present."""
    name = str(name).strip().upper()
    if len(name) > 2 and name[-1] in ['L', 'R'] and name[-2] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        return name[:-1]
    return name

def build_networks(df, neuron_names):
    """Build chemical-vs-electrical GABA adjacency matrices."""
    n = len(neuron_names)
    name_to_idx = {name: i for i, name in enumerate(neuron_names)}
    
    # Pre-calculate normalized names
    df['Src_Norm'] = df['Source'].apply(normalize_name)
    df['Tgt_Norm'] = df['Target'].apply(normalize_name)
    
    # 1. Define Synaptic Set
    # Chemical edges with GABA source and GABA receptor target
    syn_mask = (df['Type'] == 'chemical') & (df['GABA'] == 1) & ((df['target_GABAa'] == 1) | (df['target_GBB'] == 1))
    syn_pairs = set()
    
    A_syn = np.zeros((n, n))
    syn_count = 0
    
    for _, row in df[syn_mask].iterrows():
        src, tgt = row['Src_Norm'], row['Tgt_Norm']
        if src in name_to_idx and tgt in name_to_idx:
            # Add to set for exclusion logic
            syn_pairs.add((src, tgt))
            
            i, j = name_to_idx[tgt], name_to_idx[src]
            if A_syn[i, j] == 0:
                syn_count += 1
            A_syn[i, j] = 1

    print(f"Built Synaptic Network: {syn_count} edges (from {len(df[syn_mask])} raw rows)")

    # 2. Define Non-Synaptic (GJ) Set
    # Electrical edges with GABA source and GABA receptor target
    # STRICTLY EXCLUDING pairs that are already synaptic
    gj_mask = (df['Type'] == 'electrical') & (df['GABA'] == 1) & ((df['target_GABAa'] == 1) | (df['target_GBB'] == 1))
    
    A_nonsyn = np.zeros((n, n))
    nonsyn_count = 0
    excluded_count = 0
    
    for _, row in df[gj_mask].iterrows():
        src, tgt = row['Src_Norm'], row['Tgt_Norm']
        if src in name_to_idx and tgt in name_to_idx:
            # Check if this pair is already synaptic
            if (src, tgt) in syn_pairs:
                excluded_count += 1
                continue
                
            i, j = name_to_idx[tgt], name_to_idx[src]
            if A_nonsyn[i, j] == 0:
                nonsyn_count += 1
            A_nonsyn[i, j] = 1
            
    print(f"Built Non-Synaptic Network: {nonsyn_count} edges (Excluded {excluded_count} pairs overlapping with synaptic)")
    
    return A_syn, A_nonsyn

def compute_metrics(y_true, y_score):
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    mask = ~np.eye(y_true.shape[0], dtype=bool).flatten()
    y_true_eval = y_true_flat[mask] > 0
    y_score_eval = y_score_flat[mask]
    
    if np.sum(y_true_eval) == 0:
        return np.nan, np.nan, np.nan

    auroc = roc_auc_score(y_true_eval, y_score_eval)
    auprc = average_precision_score(y_true_eval, y_score_eval)
    
    prec, rec, thresholds = precision_recall_curve(y_true_eval, y_score_eval)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_f1 = np.max(f1_scores)
    
    return auroc, auprc, best_f1

def get_peak_stats(time_s, f1_scores):
    if np.all(np.isnan(f1_scores)):
        return np.nan, np.nan, np.nan, np.nan

    idx_max = np.nanargmax(f1_scores)
    t_max = time_s[idx_max]
    y_max = f1_scores[idx_max]
    
    if 0 < idx_max < len(time_s) - 1:
        y_prev = f1_scores[idx_max - 1]
        y_next = f1_scores[idx_max + 1]
        dt = time_s[1] - time_s[0]
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

def main():
    res_data, df_conn = load_data()
    neuron_names = res_data['neuron_names'].tolist()
    
    # Build
    A_syn, A_nonsyn = build_networks(df_conn, neuron_names)
    
    lags = sorted([int(k.replace('mu_hat_lag', '')) for k in res_data.keys() if k.startswith('mu_hat_lag')])
    
    results = []
    for lag in lags:
        mu_hat = res_data[f'mu_hat_lag{lag}']
        score = np.abs(mu_hat)
        
        auc_s, prc_s, f1_s = compute_metrics(A_syn, score)
        auc_ns, prc_ns, f1_ns = compute_metrics(A_nonsyn, score)
        
        results.append({
            'lag': lag, 'time_s': lag * 0.25,
            'f1_syn': f1_s, 'auroc_syn': auc_s, 'auprc_syn': prc_s,
            'f1_nonsyn': f1_ns, 'auroc_nonsyn': auc_ns, 'auprc_nonsyn': prc_ns
        })
        print(f"Lag {lag} ({lag*0.25}s): Syn F1={f1_s:.3f}, Non-Syn F1={f1_ns:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "synaptic_vs_nonsynaptic_analysis.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('f1', 'F1 Score')]
    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        ax.plot(df_res['time_s'], df_res[f'{metric}_syn'], 'o-', label=f'{GABA_LABEL} Synaptic (Wired)', color='tab:blue', linewidth=2)
        ax.plot(df_res['time_s'], df_res[f'{metric}_nonsyn'], 's--', label=f'{GABA_LABEL} Non-Synaptic (Volume Proxy)', color='tab:orange', linewidth=2)
        ax.set_title(f"{label} vs Lag")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if metric == 'auroc': ax.axhline(0.5, color='gray', linestyle=':')
        ax.legend()

    plt.suptitle(f"{GABA_LABEL}: Synaptic vs Non-Synaptic Signaling Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_synaptic_vs_nonsynaptic.png", dpi=300)
    print(f"Saved plot to {OUTPUT_DIR / 'fig_synaptic_vs_nonsynaptic.png'}")
    
    # Peak Analysis
    ts, ti, ys, yi = get_peak_stats(df_res['time_s'].values, df_res['f1_syn'].values)
    tn, tni, yn, yni = get_peak_stats(df_res['time_s'].values, df_res['f1_nonsyn'].values)
    
    print("\n--- Peak Analysis ---")
    print(f"Synaptic Peak: {ts}s (Interp: {ti:.2f}s), F1: {ys:.3f}")
    print(f"Non-Synaptic Peak: {tn}s (Interp: {tni:.2f}s), F1: {yn:.3f}")

if __name__ == "__main__":
    main()
