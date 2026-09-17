#!/usr/bin/env python3
"""
GABA Receptor Sub-network Analysis

Compares SBTG recovery of synaptic connections mediated by different GABA
receptor families:
  - Yemini_GABA_2021 (GABA-A ionotropic, fast): UNC-49, EXP-1, GAB-1, LGC-35/36/37/38
  - Yemini_GABA_2021 (GABA-B metabotropic, slow): GBB-1, GBB-2

Builds receptor-specific adjacency matrices from the connectome CSV, aligns
them with the SBTG neuron set, computes AUROC/AUPRC/F1 at each lag, and
generates comparison figures.

Usage:
    python analysis/evaluation/analyze_gaba_receptors.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from pipeline.utils.display_names import GABA_LABEL

# Paths
RESULTS_DIR = Path("merged_results")
DATA_DIR = Path("data")
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    # Load SBTG Results
    res_file = RESULTS_DIR / "result_C_merged.npz"
    print(f"Loading results from {res_file}...")
    res_data = np.load(res_file, allow_pickle=True)
    
    # Load Connectome CSV
    csv_file = DATA_DIR / "connectome_syn_gj.csv"
    print(f"Loading connectome from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    return res_data, df

def normalize_name(name):
    """Normalize neuron name by removing L/R suffix if present."""
    name = str(name).strip().upper()
    # Strip L/R laterality suffix from standard C. elegans neuron names
    if len(name) > 2 and name[-1] in ['L', 'R'] and name[-2] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        return name[:-1]
    return name

def build_adjacency(df, neuron_names, network_type='gaba_a'):
    """Build adjacency matrix for specific GABA receptor type."""
    n = len(neuron_names)
    
    # Check naming convention of SBTG data
    print(f"  SBTG Neuron Examples: {neuron_names[:5]}")
    sbtg_is_collapsed = all([name[-1] not in ['L', 'R'] for name in neuron_names[:5] if len(name)>2])
    print(f"  SBTG names appear collapsed? {sbtg_is_collapsed}")
    
    name_to_idx = {name: i for i, name in enumerate(neuron_names)}
    A = np.zeros((n, n))
    
    # Filter edges based on network type
    if network_type == 'gaba_a':
        mask = (df['GABA'] == 1) & (df['target_GABAa'] == 1) & (df['Type'] == 'chemical')
    elif network_type == 'gaba_b':
        mask = (df['GABA'] == 1) & (df['target_GBB'] == 1) & (df['Type'] == 'chemical')
    elif network_type == 'gaba_dual':
        # Co-expression: Target has BOTH 
        mask = (df['GABA'] == 1) & (df['target_GABAa'] == 1) & (df['target_GBB'] == 1) & (df['Type'] == 'chemical')
    elif network_type == 'gaba_any':
        # Union: Target has EITHER
        mask = (df['GABA'] == 1) & ((df['target_GABAa'] == 1) | (df['target_GBB'] == 1)) & (df['Type'] == 'chemical')
    else:
        raise ValueError(f"Unknown network type: {network_type}")
        
    subset = df[mask]
    print(f"  Building {network_type} network: {len(subset)} edges found in CSV.")
    
    count = 0
    mapped_edges = 0
    
    for _, row in subset.iterrows():
        src = row['Source']
        tgt = row['Target']
        
        # Normalize if needed
        if sbtg_is_collapsed:
            src = normalize_name(src)
            tgt = normalize_name(tgt)
            
        if src in name_to_idx and tgt in name_to_idx:
            i = name_to_idx[tgt]
            j = name_to_idx[src]
            
            if A[i, j] == 0:
                count += 1
            A[i, j] = 1 
            mapped_edges += 1
            
    print(f"  Mapped {mapped_edges} biological edges to {count} unique SBTG links.")
    return A

def compute_metrics(y_true, y_score):
    """Compute AUROC, AUPRC, F1 (at best threshold)."""
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
    """Calculate discrete and interpolated peak."""
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
    print("--- STARTING GABA RECEPTOR ANALYSIS ---", flush=True)
    res_data, df_conn = load_data()
    neuron_names = res_data['neuron_names'].tolist()
    
    # Build Networks
    print("Building GABA networks...", flush=True)
    networks = {
        'GABA-A':    build_adjacency(df_conn, neuron_names, 'gaba_a'),
        'GABA-B':    build_adjacency(df_conn, neuron_names, 'gaba_b')
    }
    
    lags = sorted([int(k.replace('mu_hat_lag', '')) for k in res_data.keys() if k.startswith('mu_hat_lag')])
    
    results = []
    
    for lag in lags:
        mu_hat = res_data[f'mu_hat_lag{lag}']
        score = np.abs(mu_hat)
        
        row = {'lag': lag, 'time_s': lag * 0.25}
        for name, A in networks.items():
            auc, prc, f1 = compute_metrics(A, score)
            row[f'f1_{name}'] = f1
            row[f'auroc_{name}'] = auc
            row[f'auprc_{name}'] = prc
            
        results.append(row)
        print(
            f"Lag {lag} ({lag*0.25}s): "
            f"{GABA_LABEL} (GABA-A) F1={row['f1_GABA-A']:.3f}, "
            f"{GABA_LABEL} (GABA-B) F1={row['f1_GABA-B']:.3f}"
        )
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "gaba_receptor_synaptic_analysis.csv", index=False)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.rcParams.update({'font.size': 12})
    
    metrics = [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('f1', 'F1 Score')]
    styles = {
        'GABA-A':    {'c': '#66c2a5', 'ls': '-', 'm': 'o'}, # Teal
        'GABA-B':    {'c': '#fc8d62', 'ls': '-', 'm': 's'}, # Orange
    }
    display_names = {
        'GABA-A': f'{GABA_LABEL} (GABA-A)',
        'GABA-B': f'{GABA_LABEL} (GABA-B)',
    }
    
    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        for name in networks.keys():
            col_name = f'{metric}_{name}'
            st = styles[name]
            ax.plot(df_res['time_s'], df_res[col_name], 
                    label=display_names.get(name, name), color=st['c'], linestyle=st['ls'], marker=st['m'], 
                    linewidth=2.5, markersize=8, alpha=0.9)
            
        ax.set_title(f"{label} vs Lag", fontweight='bold')
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if metric == 'auroc': 
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Legend only on last plot
        if i == 2:
            ax.legend(borderaxespad=0.)
            
    plt.suptitle(f"{GABA_LABEL} Receptor Connectome Performance", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_gaba_receptor_synaptic.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_DIR / 'fig_gaba_receptor_synaptic.png'}")

if __name__ == "__main__":
    main()
