#!/usr/bin/env python3
"""
Neurotransmitter-Specific Subnetwork Comparison

Evaluates SBTG mean-transfer predictions against five network categories:
  1. Chemical synapses (Cook_Synapses_2019)
  2. Gap junctions (Cook_Synapses_2019)
  3. Dopamine / Serotonin / Tyramine / Octopamine (Bentley_Monoamines_2016)

For each category and lag, computes AUROC, AUPRC, and best-threshold F1.
Generates grouped bar and line plots comparing recovery across categories.

Usage:
    python analysis/evaluation/analyze_nt_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from pipeline.utils.display_names import GABA_LABEL, MONOAMINE_LABEL

RESULTS_DIR = Path("merged_results")
DATA_DIR = Path("data")
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    res_data = np.load(RESULTS_DIR / "result_C_merged.npz", allow_pickle=True)
    df_conn = pd.read_csv(DATA_DIR / "connectome_syn_gj.csv")
    
    # Load Monoamine Edge List
    mono_path = DATA_DIR / "S1_Dataset" / "edge_lists" / "edgelist_MA.csv"
    print(f"Loading monoamines from {mono_path}...")
    df_mono = pd.read_csv(mono_path, header=None, names=['source', 'target', 'transmitter', 'receptor'])
    
    return res_data, df_conn, df_mono

def normalize_name(name):
    name = str(name).strip().upper()
    
    # Special Handling for Monoamine dataset conventions
    if name in ['CEPD', 'CEPV']:
        return 'CEP'
        
    if len(name) > 2 and name[-1] in ['L', 'R'] and name[-2] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        return name[:-1]
    return name

def build_network(df, neuron_names, net_type, mono_df=None):
    n = len(neuron_names)
    name_to_idx = {name: i for i, name in enumerate(neuron_names)}
    A = np.zeros((n, n))
    
    # 1. Monoamine Logic (using secondary dataframe)
    if net_type in ['dopamine', 'serotonin', 'tyramine', 'octopamine']:
        if mono_df is None: raise ValueError("Monoamine DF required")
        subset = mono_df[mono_df['transmitter'] == net_type]
        print(f"  Processing {net_type}: {len(subset)} raw edges")
        
        mapped = 0
        for _, row in subset.iterrows():
            src = normalize_name(row['source'])
            tgt = normalize_name(row['target'])
            if src in name_to_idx and tgt in name_to_idx:
                i, j = name_to_idx[tgt], name_to_idx[src]
                if i != j:
                    if A[i, j] == 0: mapped += 1
                    A[i, j] = 1

        print(f"Built {net_type:<14}: {mapped} unique edges")
        return A

    # 2. Main Connectome Logic
    if net_type == 'gaba_a':
        mask = (df['Type'] == 'chemical') & (df['GABA'] == 1) & (df['target_GABAa'] == 1)
    elif net_type == 'gaba_b':
        mask = (df['Type'] == 'chemical') & (df['GABA'] == 1) & (df['target_GBB'] == 1)
    elif net_type == 'ach':
        mask = (df['Type'] == 'chemical') & (df['acetylcholine'] == 1)
    elif net_type == 'glu':
        mask = (df['Type'] == 'chemical') & (df['glutamate'] == 1)
    else:
        raise ValueError(f"Unknown type: {net_type}")
        
    subset = df[mask]
    mapped = 0
    
    for _, row in subset.iterrows():
        src = normalize_name(row['Source'])
        tgt = normalize_name(row['Target'])
        
        if src in name_to_idx and tgt in name_to_idx:
            i, j = name_to_idx[tgt], name_to_idx[src]
            if A[i, j] == 0:
                mapped += 1
            A[i, j] = 1
            
    print(f"Built {net_type:<14}: {mapped} unique edges")
    return A

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
    
    prec, rec, _ = precision_recall_curve(y_true_eval, y_score_eval)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_f1 = np.max(f1_scores)
    
    return auroc, auprc, best_f1

def main():
    res_data, df_conn, df_mono = load_data()
    names = res_data['neuron_names'].tolist()
    
    # Build Networks
    networks = {
        'GABA-A':      build_network(df_conn, names, 'gaba_a'),
        'GABA-B':      build_network(df_conn, names, 'gaba_b'),
        'ACh':         build_network(df_conn, names, 'ach'),
        'Glutamate':   build_network(df_conn, names, 'glu'),
        'Dopamine':    build_network(df_conn, names, 'dopamine', df_mono),
        'Serotonin':   build_network(df_conn, names, 'serotonin', df_mono),
        'Octopamine':  build_network(df_conn, names, 'octopamine', df_mono),
        'Tyramine':    build_network(df_conn, names, 'tyramine', df_mono)
    }
    
    styles = {
        'GABA-A':      {'c': '#1b9e77', 'ls': '-', 'm': 'o'},  # Teal
        'GABA-B':      {'c': '#7570b3', 'ls': '-', 'm': 's'},  # Purple-ish
        'ACh':         {'c': '#d95f02', 'ls': '-', 'm': 'D'},  # Orange
        'Glutamate':   {'c': '#e7298a', 'ls': '-', 'm': '^'},  # Pink
        'Dopamine':    {'c': '#66a61e', 'ls': '--', 'm': 'v'}, # Green
        'Serotonin':   {'c': '#e6ab02', 'ls': '--', 'm': 'p'}, # Yellow
        'Octopamine':  {'c': '#a6761d', 'ls': '--', 'm': '*'}, # Brown
        'Tyramine':    {'c': '#666666', 'ls': '--', 'm': 'x'}  # Grey
    }
    display_names = {
        'GABA-A': f'{GABA_LABEL} (GABA-A)',
        'GABA-B': f'{GABA_LABEL} (GABA-B)',
        'Dopamine': f'{MONOAMINE_LABEL} (Dopamine)',
        'Serotonin': f'{MONOAMINE_LABEL} (Serotonin)',
        'Octopamine': f'{MONOAMINE_LABEL} (Octopamine)',
        'Tyramine': f'{MONOAMINE_LABEL} (Tyramine)',
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
        print(f"Lag {lag}: ACh={row['f1_ACh']:.3f}, DA={row['f1_Dopamine']:.3f}, 5-HT={row['f1_Serotonin']:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / "full_nt_comparison.csv", index=False)
    
    # Plot Comparison (AUROC, AUPRC, F1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.rcParams.update({'font.size': 12})
    
    metrics = [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('f1', 'F1 Score')]
    
    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        for name in networks.keys():
            col_name = f'{metric}_{name}'
            st = styles.get(name, {'c': 'black', 'ls': '-', 'm': '.'})
            ax.plot(df_res['time_s'], df_res[col_name], 
                    label=display_names.get(name, name), color=st['c'], linestyle=st['ls'], marker=st['m'], 
                    linewidth=2, ms=6, alpha=0.9)
            
        ax.set_title(f"{label} vs Lag", fontweight='bold')
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if metric == 'auroc': 
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        if i == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.suptitle(f"Neurotransmitter & {MONOAMINE_LABEL} Subnetwork Performance", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_nt_comparison_performance.png", dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {OUTPUT_DIR / 'fig_nt_comparison_performance.png'}")
    
    # --- Additional Plot: Modulators Only (No ACh/Glu) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Filter networks to plot
    modulator_networks = [n for n in networks.keys() if n not in ['ACh', 'Glutamate']]
    
    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        for name in modulator_networks:
            col_name = f'{metric}_{name}'
            st = styles.get(name, {'c': 'black', 'ls': '-', 'm': '.'})
            ax.plot(df_res['time_s'], df_res[col_name], 
                    label=display_names.get(name, name), color=st['c'], linestyle=st['ls'], marker=st['m'], 
                    linewidth=2, ms=6, alpha=0.9)
            
        ax.set_title(f"{label} vs Lag", fontweight='bold')
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if metric == 'auroc': 
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        if i == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.suptitle("Inhibitory & Modulatory Subnetwork Performance (No Drivers)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_nt_comparison_modulators.png", dpi=300, bbox_inches='tight')
    print(f"Saved filtered plot to {OUTPUT_DIR / 'fig_nt_comparison_modulators.png'}")

if __name__ == "__main__":
    main()
