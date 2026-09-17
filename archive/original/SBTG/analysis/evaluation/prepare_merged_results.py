#!/usr/bin/env python3
"""
Prepare Merged Results for Analysis

This script:
1. Loads SBTG results from result_C.npz
2. Loads baseline methods (Pearson, VAR, Granger) from baselines.npz
3. Evaluates all methods against structural and functional benchmarks
4. Evaluates all methods against monoamine networks
5. Saves evaluation results as CSVs

Run this script first, then run generate_merged_figures.py to create visualizations.

Usage:
    python analysis/evaluation/prepare_merged_results.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import spearmanr

# Add pipeline to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from pipeline.utils.io import load_structural_connectome
from pipeline.utils.align import align_matrices
from pipeline.utils.metrics import compute_weight_correlation

# Constants
SAMPLING_RATE = 4.0
MERGED_DIR = PROJECT_ROOT / "merged_results"
OUTPUT_DIR = MERGED_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"

print("="*80)
print("PREPARE MERGED RESULTS FOR ANALYSIS")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/4] Loading SBTG and baseline results...")
data = np.load(MERGED_DIR / "result_C.npz", allow_pickle=True)
baselines = np.load(MERGED_DIR / "baselines.npz", allow_pickle=True)

# Replace lag-1 with the regime-gated model if available
best_lag1_file = MERGED_DIR / "regime_gated_full_traces_imputed_imputed_best_20260118_022121.npz"
if best_lag1_file.exists():
    print("  Found better lag-1 model, replacing...")
    best_lag1 = np.load(best_lag1_file, allow_pickle=True)
    
    # Extract lag-1 data from the better model
    # This model has results at multiple lags, we want lag 1
    if 'mu_hat' in best_lag1:
        # Single-lag format
        mu_hat_lag1_new = best_lag1['mu_hat']
        # Fix: Input file has 'sign_adj', not 'sig_mask'
        if 'sign_adj' in best_lag1:
            sig_lag1_new = (best_lag1['sign_adj'] != 0)
        else:
            sig_lag1_new = best_lag1.get('sig_mask', np.ones_like(mu_hat_lag1_new))
        
        pval_lag1_new = best_lag1.get('pval', np.ones_like(mu_hat_lag1_new))
    else:
        # Multi-lag format - extract lag 1
        mu_hat_lag1_new = best_lag1['mu_hat_lag1']
        sig_lag1_new = best_lag1.get('sig_lag1', np.ones_like(mu_hat_lag1_new))
        pval_lag1_new = best_lag1.get('pval_lag1', np.ones_like(mu_hat_lag1_new))
    
    # Replace lag-1 in the main data
    data_dict = {key: data[key] for key in data.keys()}
    data_dict['mu_hat_lag1'] = mu_hat_lag1_new
    data_dict['sig_lag1'] = sig_lag1_new
    data_dict['pval_lag1'] = pval_lag1_new
    
    # Recreate data as a dict for easier access
    data = data_dict
    print("  ✓ Replaced lag-1 with better model")
else:
    # Convert to dict for consistent access
    data = {key: data[key] for key in data.keys()}

neuron_names = list(data['neuron_names'])
n_neurons = len(neuron_names)
if 'lags' in data:
    sbtg_lags = sorted(list(data['lags']))
else:
    sbtg_lags = sorted(list(set([int(k.split('lag')[-1]) for k in data.keys() if k.startswith('mu_hat_lag')])))

# Identify baseline lags
baseline_lags = {'Pearson': [], 'VAR': [], 'Granger': []}
for key in baselines.keys():
    for method in baseline_lags.keys():
        if key.startswith(method) and '_lag' in key:
            lag = int(key.split('_lag')[-1])
            baseline_lags[method].append(lag)

for method in baseline_lags.keys():
    baseline_lags[method] = sorted(set(baseline_lags[method]))

print(f"  Neurons: {n_neurons}")
print(f"  SBTG lags: {sbtg_lags}")
for method, lags in baseline_lags.items():
    print(f"  {method} lags: {lags}")

# =============================================================================
# LOAD GROUND TRUTH CONNECTOMES
# =============================================================================

print("\n[2/4] Loading ground truth connectomes...")

# Cook structural connectome
connectome_dir = PROJECT_ROOT / "results" / "intermediate" / "connectome"
A_chem_full = np.load(connectome_dir / "A_chem.npy")
A_gap_full = np.load(connectome_dir / "A_gap.npy")
with open(connectome_dir / "nodes.json") as f:
    cook_neurons = json.load(f)
A_cook_full = A_chem_full + A_gap_full

# Align Cook to our neuron order
A_cook, common_cook = align_matrices(A_cook_full, cook_neurons, neuron_names)

# Leifer functional connectome
leifer_dir = PROJECT_ROOT / "results" / "leifer_evaluation"
leifer_wt = np.load(leifer_dir / "aligned_atlas_wild-type.npz", allow_pickle=True)
A_leifer_full = leifer_wt['q']  # q is the connectivity matrix
leifer_neurons = list(leifer_wt['neuron_order'])
# Replace NaNs with 0 (no connection)
A_leifer_full = np.nan_to_num(A_leifer_full, nan=0.0)

# Align Leifer to our neuron order
A_leifer, common_leifer = align_matrices(A_leifer_full, leifer_neurons, neuron_names)

print(f"  Cook connectome: {int(np.sum(A_cook != 0))} edges ({len(common_cook)} neurons)")
print(f"  Leifer connectome: {int(np.sum(A_leifer != 0))} edges ({len(common_leifer)} neurons)")

# Load monoamine networks
bentley_dir = DATA_DIR / "S1_Dataset"
csv_path = bentley_dir / "edge_lists" / "edgelist_MA.csv"

df_mono = pd.read_csv(csv_path, header=None, names=['source', 'target', 'transmitter', 'receptor'])

def normalize_name(name):
    """Normalize neuron names."""
    name = str(name).upper().strip()
    if len(name) > 1 and name[-1] in ['L', 'R']:
        base = name[:-1]
        if len(base) >= 2:
            name = base
    if name in ['CEPD', 'CEPV']:
        name = 'CEP'
    return name

df_mono['source_norm'] = df_mono['source'].apply(normalize_name)
df_mono['target_norm'] = df_mono['target'].apply(normalize_name)

# Build monoamine adjacency matrices
transmitters = ['dopamine', 'serotonin', 'tyramine', 'octopamine']
mono_networks = {}
name_to_idx = {name: i for i, name in enumerate(neuron_names)}

for trans in transmitters:
    df_trans = df_mono[df_mono['transmitter'] == trans]
    adj = np.zeros((n_neurons, n_neurons))
    
    for _, row in df_trans.iterrows():
        src = row['source_norm']
        tgt = row['target_norm']
        if src in name_to_idx and tgt in name_to_idx:
            i = name_to_idx[tgt]
            j = name_to_idx[src]
            if i != j:
                adj[i, j] = 1
    
    mono_networks[trans] = adj
    print(f"  {trans}: {int(adj.sum())} edges")

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_connectome(scores, ground_truth, significance_mask=None):
    """
    Evaluate against a ground truth connectome.
    
    Args:
        scores: (n, n) predicted weights
        ground_truth: (n, n) ground truth weights (synapse counts or functional strength)
        significance_mask: (n, n) boolean mask of significant edges (for SBTG)
    """
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    # Validation
    if scores.shape != ground_truth.shape:
        print(f"Warning: Shape mismatch {scores.shape} vs {ground_truth.shape}")
        return {'auroc': 0.5, 'auprc': 0.0, 'spearman': 0.0}
    
    # 1. Binary Classification Metrics (AUROC, AUPRC)
    # We binarize ground truth for these: edge vs no-edge
    y_score = np.abs(scores[mask])
    y_true_binary = (ground_truth[mask] != 0).astype(int)
    
    valid = np.isfinite(y_score) & np.isfinite(y_true_binary)
    if valid.sum() == 0 or y_true_binary[valid].sum() == 0 or y_true_binary[valid].sum() == len(y_score[valid]):
        auroc, auprc = 0.5, 0.0
    else:
        try:
            auroc = roc_auc_score(y_true_binary[valid], y_score[valid])
            auprc = average_precision_score(y_true_binary[valid], y_score[valid])
        except Exception:
            auroc, auprc = 0.5, 0.0

    # 2. Weight Correlation (Spearman)
    # We correlate continuous weights against continuous weights (synapse counts)
    # Key discrepancy fix: Use weighted GT, and apply significance mask if provided
    
    weight_metrics = compute_weight_correlation(
        scores, 
        ground_truth, 
        exclude_diagonal=True,
        significance_mask=significance_mask
    )
    spearman = weight_metrics.get('spearman_rho', 0.0)
    if np.isnan(spearman):
        spearman = 0.0
    
    # 3. F1 Score (Best)
    f1 = compute_f1(scores, ground_truth)
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'spearman': float(spearman),
        'f1': float(f1),
    }

def compute_f1(scores, ground_truth):
    """Compute best F1 score."""
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = (ground_truth[mask] != 0).astype(float)  # Binarize
    
    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = np.max(f1_scores)
    
    return float(best_f1)

# =============================================================================
# EVALUATE AGAINST COOK AND LEIFER
# =============================================================================

print("\n[3/4] Evaluating against structural and functional benchmarks...")

cook_results = []
leifer_results = []

# SBTG
print("  Evaluating SBTG...")
for lag in sbtg_lags:
    mu_hat = data[f'mu_hat_lag{lag}']
    
    # Get significance mask if available
    sig_key = f'sig_lag{lag}'
    if sig_key in data:
        sig_mask = data[sig_key]
    else:
        sig_mask = None

    # Align to Cook neurons
    # A_cook is already aligned to common_cook order
    # We need to align mu_hat (in neuron_names order) to common_cook order
    mu_aligned_cook, _ = align_matrices(mu_hat, neuron_names, common_cook)
    
    # Also align mask if it exists
    if sig_mask is not None:
        sig_mask_aligned_cook, _ = align_matrices(sig_mask, neuron_names, common_cook)
        sig_mask_aligned_cook = sig_mask_aligned_cook.astype(bool)
    else:
        sig_mask_aligned_cook = None
        
    metrics = evaluate_connectome(mu_aligned_cook, A_cook, significance_mask=sig_mask_aligned_cook)
    cook_results.append({
        'method': 'SBTG',
        'lag': lag,
        'time_s': lag / SAMPLING_RATE,
        **metrics
    })
    
    # Align to Leifer neurons
    mu_aligned_leifer, _ = align_matrices(mu_hat, neuron_names, common_leifer)
    
    # Also align mask for Leifer
    if sig_mask is not None:
        sig_mask_aligned_leifer, _ = align_matrices(sig_mask, neuron_names, common_leifer)
        sig_mask_aligned_leifer = sig_mask_aligned_leifer.astype(bool)
    else:
        sig_mask_aligned_leifer = None

    metrics = evaluate_connectome(mu_aligned_leifer, A_leifer, significance_mask=sig_mask_aligned_leifer)
    leifer_results.append({
        'method': 'SBTG',
        'lag': lag,
        'time_s': lag / SAMPLING_RATE,
        **metrics
    })

# Baselines
for method in ['Pearson', 'VAR', 'Granger']:
    print(f"  Evaluating {method}...")
    for lag in baseline_lags[method]:
        key = f"{method}_lag{lag}"
        if key not in baselines:
            continue
        
        mat = baselines[key]
        
        # Align to Cook neurons
        mat_aligned_cook, _ = align_matrices(mat, neuron_names, common_cook)
        metrics = evaluate_connectome(mat_aligned_cook, A_cook, significance_mask=None)
        cook_results.append({
            'method': method,
            'lag': lag,
            'time_s': lag / SAMPLING_RATE,
            **metrics
        })
        
        # Align to Leifer neurons
        mat_aligned_leifer, _ = align_matrices(mat, neuron_names, common_leifer)
        metrics = evaluate_connectome(mat_aligned_leifer, A_leifer, significance_mask=None)
        leifer_results.append({
            'method': method,
            'lag': lag,
            'time_s': lag / SAMPLING_RATE,
            **metrics
        })

# Save results
df_cook = pd.DataFrame(cook_results)
df_leifer = pd.DataFrame(leifer_results)

df_cook.to_csv(OUTPUT_DIR / 'eval_cook_C.csv', index=False)
df_leifer.to_csv(OUTPUT_DIR / 'eval_leifer_C.csv', index=False)

print(f"  ✓ Saved: eval_cook_C.csv")
print(f"  ✓ Saved: eval_leifer_C.csv")

# =============================================================================
# EVALUATE AGAINST MONOAMINE NETWORKS
# =============================================================================

print("\n[4/4] Evaluating against monoamine networks...")

mono_results = []

# SBTG
print("  Evaluating SBTG...")
for lag in sbtg_lags:
    mu_hat = data[f'mu_hat_lag{lag}']
    
    for trans in transmitters:
        gt = mono_networks[trans]
        metrics = evaluate_connectome(mu_hat, gt)
        
        mono_results.append({
            'method': 'SBTG',
            'lag': lag,
            'time_s': lag / SAMPLING_RATE,
            'transmitter': trans,
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'f1': compute_f1(mu_hat, gt),
        })

# Baselines
for method in ['Pearson', 'VAR', 'Granger']:
    print(f"  Evaluating {method}...")
    for lag in baseline_lags[method]:
        key = f"{method}_lag{lag}"
        if key not in baselines:
            continue
        
        mat = baselines[key]
        
        for trans in transmitters:
            gt = mono_networks[trans]
            metrics = evaluate_connectome(mat, gt)
            
            mono_results.append({
                'method': method,
                'lag': lag,
                'time_s': lag / SAMPLING_RATE,
                'transmitter': trans,
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
                'f1': compute_f1(mat, gt),
            })

# Save results
df_mono = pd.DataFrame(mono_results)
df_mono.to_csv(OUTPUT_DIR / 'eval_monoamine_all_methods.csv', index=False)

# =============================================================================
# SAVE METADATA (DENSITIES)
# =============================================================================

# Calculate densities (prevalence) for AUPRC baselines
def compute_density(adj):
    n = adj.shape[0]
    n_possible = n * (n - 1)
    n_edges = np.sum(adj != 0)
    return n_edges / n_possible

pi_cook = compute_density(A_cook)
pi_leifer = compute_density(A_leifer)

mono_densities = {}
for trans, net in mono_networks.items():
    mono_densities[trans] = compute_density(net)

baselines_info = {
    'pi_cook': pi_cook,
    'pi_leifer': pi_leifer,
    'mono_densities': mono_densities
}

with open(OUTPUT_DIR / 'baselines_info.json', 'w') as f:
    json.dump(baselines_info, f, indent=4)

print(f"  ✓ Saved baselines_info.json (Cook pi={pi_cook:.4f}, Leifer pi={pi_leifer:.4f})")
print(f"  ✓ Saved: eval_monoamine_all_methods.csv")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nCook Connectome (AUROC @ Lag 1):")
lag1_cook = df_cook[df_cook['lag'] == 1].sort_values('auroc', ascending=False)
for _, row in lag1_cook.iterrows():
    print(f"  {row['method']:8s}: {row['auroc']:.4f}")

print("\nLeifer Connectome (AUROC @ Lag 1):")
lag1_leifer = df_leifer[df_leifer['lag'] == 1].sort_values('auroc', ascending=False)
for _, row in lag1_leifer.iterrows():
    print(f"  {row['method']:8s}: {row['auroc']:.4f}")

print("\nMonoamine Networks - Best AUROC per Transmitter @ Lag 1:")
lag1_mono = df_mono[df_mono['lag'] == 1]
for trans in transmitters:
    trans_data = lag1_mono[lag1_mono['transmitter'] == trans]
    best = trans_data.loc[trans_data['auroc'].idxmax()]
    print(f"  {trans:12s}: {best['method']:8s} = {best['auroc']:.4f}")

print("\n" + "="*80)
print("PREPARATION COMPLETE")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nNext step: Run generate_merged_figures.py to create visualizations")
