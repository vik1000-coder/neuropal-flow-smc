#!/usr/bin/env python3
"""
Generate All Figures for Merged Results

This script generates all visualizations from the evaluation results:
1. Cook_Synapses_2019 / Randi_Optogenetics_2023 comparison plots (AUROC, AUPRC, Spearman)
2. Cell-type heatmaps for all methods
3. Bentley_Monoamines_2016 analysis line plots (AUROC, AUPRC, F1)
4. Bentley_Monoamines_2016 comparison heatmaps

Prerequisites:
    Run prepare_merged_results.py first to generate evaluation CSVs

Usage:
    python analysis/figures/generate_merged_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

# Add pipeline to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.neuron_types import get_neuron_type
from pipeline.utils.display_names import FUNCTIONAL_LABEL, MONOAMINE_LABEL, STRUCTURAL_LABEL

# Constants
MERGED_DIR = PROJECT_ROOT / "merged_results"
OUTPUT_DIR = MERGED_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("GENERATE MERGED FIGURES")
print("="*80)

# =============================================================================
# LOAD EVALUATION RESULTS
# =============================================================================

print("\n[1/5] Loading evaluation results...")

df_cook = pd.read_csv(OUTPUT_DIR / 'eval_cook_C.csv')
df_leifer = pd.read_csv(OUTPUT_DIR / 'eval_leifer_C.csv')
df_mono = pd.read_csv(OUTPUT_DIR / 'eval_monoamine_all_methods.csv')

print(f"  ✓ Loaded {STRUCTURAL_LABEL} evaluation: {len(df_cook)} rows")
print(f"  ✓ Loaded {FUNCTIONAL_LABEL} evaluation: {len(df_leifer)} rows")
print(f"  ✓ Loaded {MONOAMINE_LABEL} evaluation: {len(df_mono)} rows")

# Load adjacency matrices for heatmaps
data = np.load(MERGED_DIR / "result_C.npz", allow_pickle=True)
baselines = np.load(MERGED_DIR / "baselines.npz", allow_pickle=True)

# Update data dict to allow item assignment
data = dict(data)

# Replace lag-1 with the regime-gated model if available
best_lag1_file = MERGED_DIR / "regime_gated_full_traces_imputed_imputed_best_20260118_022121.npz"
if best_lag1_file.exists():
    print("  Found better lag-1 model, replacing...")
    best_lag1 = np.load(best_lag1_file, allow_pickle=True)
    
    if 'mu_hat' in best_lag1:
        data['mu_hat_lag1'] = best_lag1['mu_hat']
    elif 'mu_hat_lag1' in best_lag1:
        data['mu_hat_lag1'] = best_lag1['mu_hat_lag1']
    
neuron_names = list(data['neuron_names'])
n_neurons = len(neuron_names)

# Extract lags
if 'lags' in data:
    sbtg_lags = sorted(list(data['lags']))
else:
    # Fallback if lags not explicitly stored
    sbtg_lags = [int(k.split('lag')[-1]) for k in data.keys() if k.startswith('mu_hat_lag')]
    sbtg_lags = sorted(list(set(sbtg_lags)))

# Identify baseline lags
baseline_lags = {'Pearson': [], 'VAR': [], 'Granger': []}
for key in baselines.keys():
    for method in baseline_lags.keys():
        if key.startswith(method) and '_lag' in key:
            try:
                lag = int(key.split('_lag')[-1])
                baseline_lags[method].append(lag)
            except ValueError:
                pass

for method in baseline_lags.keys():
    baseline_lags[method] = sorted(set(baseline_lags[method]))

print(f"  SBTG lags: {sbtg_lags}")
for method, lags in baseline_lags.items():
    print(f"  {method} lags: {lags}")

# =============================================================================
# FIGURE 1: COMPREHENSIVE COMPARISON (COOK + LEIFER)
# =============================================================================

print("\n[2/5] Generating comprehensive comparison plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics = [
    ('auroc', 'AUROC', 0.5),
    ('auprc', 'AUPRC', None),
    ('spearman', 'Spearman ρ', 0.0),
]

datasets = [
    (df_cook, STRUCTURAL_LABEL, 0),
    (df_leifer, FUNCTIONAL_LABEL, 1),
]

colors = {'SBTG': '#e41a1c', 'Pearson': '#377eb8', 'VAR': '#4daf4a', 'Granger': '#984ea3'}

for row_idx, (df, dataset_name, _) in enumerate(datasets):
    for col_idx, (metric, metric_name, baseline) in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        
        for method in ['SBTG', 'Pearson', 'VAR', 'Granger']:
            method_data = df[df['method'] == method].sort_values('lag')
            if len(method_data) > 0:
                ax.plot(method_data['time_s'], method_data[metric],
                       marker='o', label=method, color=colors[method],
                       linewidth=2.5, markersize=8)
        
        if baseline is not None:
            ax.axhline(baseline, color='gray', linestyle=':', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Time Lag (seconds)', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{dataset_name}\n{metric_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle('Connectome Prediction Performance: All Methods', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: fig_comprehensive_comparison.png")

# Individual metric plots
for metric, metric_name, baseline in metrics:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax_idx, (df, dataset_name, _) in enumerate(datasets):
        ax = axes[ax_idx]
        
        for method in ['SBTG', 'Pearson', 'VAR', 'Granger']:
            method_data = df[df['method'] == method].sort_values('lag')
            if len(method_data) > 0:
                ax.plot(method_data['time_s'], method_data[metric],
                       marker='o', label=method, color=colors[method],
                       linewidth=2.5, markersize=8)
        
        if baseline is not None:
            ax.axhline(baseline, color='gray', linestyle=':', lw=1.5, alpha=0.7, label='Random')
        
        ax.set_xlabel('Time Lag (seconds)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(dataset_name, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{metric_name} Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_{metric}_comparison.png")

# =============================================================================
# FIGURE 2: CELL-TYPE HEATMAPS
# =============================================================================

print("\n[3/5] Generating cell-type heatmaps...")

# Get cell types
cell_types = [get_neuron_type(name) for name in neuron_names]
type_to_idx = {'sensory': 0, 'interneuron': 1, 'motor': 2, 'unknown': 3}
type_indices = [type_to_idx.get(ct, 3) for ct in cell_types]

# Sort neurons by type
sorted_indices = np.argsort(type_indices)
sorted_names = [neuron_names[i] for i in sorted_indices]
sorted_types = [cell_types[i] for i in sorted_indices]

# Find type boundaries
boundaries = []
current_type = sorted_types[0]
for i, t in enumerate(sorted_types):
    if t != current_type:
        boundaries.append(i)
        current_type = t

methods_to_plot = [
    ('SBTG', data['mu_hat_lag1']),
    ('Pearson', baselines.get('Pearson_lag1')),
    ('VAR', baselines.get('VAR_lag1')),
    ('Granger', baselines.get('Granger_lag1')),
]

for method_name, adj_matrix in methods_to_plot:
    if adj_matrix is None:
        continue
    
    # Reorder matrix
    adj_sorted = adj_matrix[sorted_indices, :][:, sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    vmax = np.percentile(np.abs(adj_sorted), 99)
    im = ax.imshow(np.abs(adj_sorted), cmap='hot', aspect='auto', vmin=0, vmax=vmax)
    
    # Add type boundaries
    for b in boundaries:
        ax.axhline(b - 0.5, color='cyan', linewidth=2, alpha=0.7)
        ax.axvline(b - 0.5, color='cyan', linewidth=2, alpha=0.7)
    
    # Labels
    ax.set_xlabel('Source Neuron (by type)', fontsize=12)
    ax.set_ylabel('Target Neuron (by type)', fontsize=12)
    ax.set_title(f'{method_name}: Cell-Type Interaction Patterns (Lag 1)',
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|Coupling Strength|', fontsize=11)
    
    # Add type labels
    type_labels = ['Sensory', 'Interneuron', 'Motor']
    type_positions = []
    for i, label in enumerate(type_labels):
        indices = [j for j, t in enumerate(sorted_types) if t == label.lower()]
        if indices:
            pos = (indices[0] + indices[-1]) / 2
            type_positions.append(pos)
    
    if len(type_positions) == 3:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(type_positions)
        ax2.set_yticklabels(type_labels, fontsize=11, fontweight='bold')
        
        ax3 = ax.twiny()
        ax3.set_xlim(ax.get_xlim())
        ax3.set_xticks(type_positions)
        ax3.set_xticklabels(type_labels, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_celltype_heatmap_{method_name.lower()}.png',
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_celltype_heatmap_{method_name.lower()}.png")

# =============================================================================
# FIGURE 3: CELL-TYPE COMPARISON BAR PLOTS
# =============================================================================

print("\n[4/5] Generating cell-type comparison plots...")

# Compute cell-type interaction strengths
def compute_celltype_interactions(adj_matrix, neuron_names):
    """Compute mean coupling strength for each cell-type pair."""
    cell_types = [get_neuron_type(name) for name in neuron_names]
    n = len(neuron_names)
    
    pairs = ['S→I', 'S→M', 'I→M', 'I→I', 'M→M', 'S→S']
    type_map = {'sensory': 'S', 'interneuron': 'I', 'motor': 'M'}
    
    results = {}
    for pair in pairs:
        src_type, tgt_type = pair.split('→')
        src_full = [k for k, v in type_map.items() if v == src_type][0]
        tgt_full = [k for k, v in type_map.items() if v == tgt_type][0]
        
        values = []
        for i in range(n):
            for j in range(n):
                if i != j and cell_types[i] == tgt_full and cell_types[j] == src_full:
                    values.append(abs(adj_matrix[i, j]))
        
        results[pair] = np.mean(values) if values else 0.0
    
    return results

# Compute for all methods
celltype_results = []

for method_name, adj_matrix in methods_to_plot:
    if adj_matrix is None:
        continue
    
    interactions = compute_celltype_interactions(adj_matrix, neuron_names)
    for pair, strength in interactions.items():
        celltype_results.append({
            'method': method_name,
            'pair': pair,
            'strength': strength,
        })

df_celltype = pd.DataFrame(celltype_results)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

pairs = ['S→I', 'S→M', 'I→M', 'I→I', 'M→M', 'S→S']
x = np.arange(len(pairs))
width = 0.2

for i, method in enumerate(['SBTG', 'Pearson', 'VAR', 'Granger']):
    method_data = df_celltype[df_celltype['method'] == method]
    values = [method_data[method_data['pair'] == p]['strength'].values[0] 
              if len(method_data[method_data['pair'] == p]) > 0 else 0
              for p in pairs]
    
    ax.bar(x + i * width, values, width, label=method, color=colors[method], alpha=0.8)

ax.set_xlabel('Cell-Type Interaction', fontsize=12)
ax.set_ylabel('Mean |Coupling Strength|', fontsize=12)
ax.set_title('Cell-Type Interaction Patterns: Method Comparison (Lag 1)',
            fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(pairs, fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_celltype_comparison_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: fig_celltype_comparison_all_methods.png")

# =============================================================================
# FIGURE 4: STATE-DEPENDENT ACTIVITY HEATMAPS
# =============================================================================

print("\n[5/6] Generating state-dependent activity heatmaps...")

# Load calcium traces to compute state percentages
try:
    import json
    from scipy.ndimage import gaussian_filter1d
    
    dataset_dir = PROJECT_ROOT / "results" / "intermediate" / "datasets" / "full_traces_imputed"
    
    X_list = []
    
    # Try X_segments.npy first (like Script 19)
    segments_file = dataset_dir / "X_segments.npy"
    if segments_file.exists():
        X_segments = np.load(segments_file, allow_pickle=True)
        # X_segments is a list or array of traces
        if isinstance(X_segments, np.ndarray):
            if X_segments.dtype == object:
                # List of arrays stored as object array
                X_list = [x for x in X_segments if x is not None and len(x) > 0]
            elif X_segments.ndim == 3:
                # (n_worms, T, n_neurons)
                for i in range(X_segments.shape[0]):
                    X_list.append(X_segments[i])
        else:
            X_list = list(X_segments)
        print(f"  Loaded from X_segments.npy: {len(X_list)} traces")
    else:
        # Fallback to Z_std
        z_std_file = dataset_dir / "Z_std.npy"
        if z_std_file.exists():
            Z_std = np.load(z_std_file)
            if Z_std.ndim == 3:
                # Multiple worms
                X_list = [Z_std[i] for i in range(Z_std.shape[0])]
            else:
                # Single trace
                X_list = [Z_std]
            print(f"  Loaded from Z_std.npy: {len(X_list)} traces")

    if X_list:
        # Identify AVA and AVB indices
        ava_idx = None
        avb_idx = None
        for i, name in enumerate(neuron_names):
            name_upper = str(name).upper().strip()
            if 'AVA' in name_upper and ava_idx is None:
                ava_idx = i
            elif 'AVB' in name_upper and avb_idx is None:
                avb_idx = i
        
        if ava_idx is not None and avb_idx is not None:
            print(f"  Found AVA at index {ava_idx}, AVB at index {avb_idx}")
            
            # Compute state percentages for each method and lag
            state_results = []
            all_states = []
            for X in X_list:
                # Smooth AVA/AVB traces
                ava_trace = gaussian_filter1d(X[:, ava_idx], sigma=1.0)
                avb_trace = gaussian_filter1d(X[:, avb_idx], sigma=1.0)
                
                # Segment states
                T = len(ava_trace)
                state_mask = np.full(T, 2, dtype=int)  # Default: transition
                
                # Forward: AVB high, AVA low
                forward_mask = (avb_trace > 0.5) & (ava_trace < -0.25)
                state_mask[forward_mask] = 0
                
                # Backward: AVA high, AVB low
                backward_mask = (ava_trace > 0.5) & (avb_trace < -0.25)
                state_mask[backward_mask] = 1
                
                all_states.append(state_mask)
            
            # Concatenate all states
            all_states_concat = np.concatenate(all_states)
            total_time = len(all_states_concat)
            
            # Compute percentages
            forward_pct = 100 * np.sum(all_states_concat == 0) / total_time
            backward_pct = 100 * np.sum(all_states_concat == 1) / total_time
            transition_pct = 100 * np.sum(all_states_concat == 2) / total_time
            
            print(f"  Overall state distribution:")
            print(f"    Forward: {forward_pct:.1f}%")
            print(f"    Backward: {backward_pct:.1f}%")
            print(f"    Transition: {transition_pct:.1f}%")
            
            # For each method and lag, compute edge activity in each state
            methods_to_analyze = [
                ('SBTG', data, sbtg_lags),
                ('Pearson', baselines, baseline_lags['Pearson']),
                ('VAR', baselines, baseline_lags['VAR']),
                ('Granger', baselines, baseline_lags['Granger']),
            ]
            
            for method_name, method_data, lags in methods_to_analyze:
                for lag in lags:
                    # Get coupling matrix
                    if method_name == 'SBTG':
                        if f'mu_hat_lag{lag}' not in method_data:
                            continue
                        mu_hat = method_data[f'mu_hat_lag{lag}']
                    else:
                        key = f"{method_name}_lag{lag}"
                        if key not in method_data:
                            continue
                        mu_hat = method_data[key]
                    
                    # Compute edge activity in each state
                    state_activity = {0: [], 1: [], 2: []}  # forward, backward, transition
                    
                    weights = np.abs(mu_hat)
                    threshold = 1e-4 * np.max(weights) if np.max(weights) > 0 else 0
                    active_mask = weights > threshold
                    
                    rows, cols = np.where(active_mask)  # i (target), j (source)
                    edge_weights = weights[rows, cols]  # (n_edges,)
                    
                    if len(rows) == 0:
                        continue

                    state_weighted_sum = {0: 0.0, 1: 0.0, 2: 0.0}
                    state_weight_total = {0: 0.0, 1: 0.0, 2: 0.0}

                    for X, state_mask in zip(X_list, all_states):
                        T = len(state_mask)
                        if T < lag + 1:
                            continue
                            
                        X_source = X[:T-lag, cols]
                        X_target = X[lag:, rows]
                        
                        # activity: (T_eff, n_edges)
                        edge_activities = np.abs(X_source * X_target)
                        
                        # states: (T_eff,)
                        states = state_mask[:T-lag]
                        
                        for s in [0, 1, 2]:
                            mask_s = (states == s)
                            n_s = np.sum(mask_s)
                            if n_s > 0:
                                mean_edge_activity_s = np.nanmean(edge_activities[mask_s], axis=0)
                                weighted_sum = np.sum(edge_weights * mean_edge_activity_s)
                                total_w = np.sum(edge_weights)
                                state_weighted_sum[s] += weighted_sum * n_s
                                state_weight_total[s] += total_w * n_s

                    # Compute final weighted means
                    forward_activity = state_weighted_sum[0] / state_weight_total[0] if state_weight_total[0] > 0 else 0
                    backward_activity = state_weighted_sum[1] / state_weight_total[1] if state_weight_total[1] > 0 else 0
                    transition_activity = state_weighted_sum[2] / state_weight_total[2] if state_weight_total[2] > 0 else 0
                    
                    print(f"  Lag {lag} {method_name}: Fwd={forward_activity:.4f}, Bwd={backward_activity:.4f}, Trans={transition_activity:.4f}")
                    
                    total_activity = forward_activity + backward_activity + transition_activity
                    if total_activity > 0:
                        forward_pct_edges = 100 * forward_activity / total_activity
                        backward_pct_edges = 100 * backward_activity / total_activity
                        transition_pct_edges = 100 * transition_activity / total_activity
                    else:
                        forward_pct_edges = backward_pct_edges = transition_pct_edges = 33.33
                    
                    state_results.append({
                        'method': method_name,
                        'lag': lag,
                        'forward_pct': forward_pct_edges,
                        'backward_pct': backward_pct_edges,
                        'transition_pct': transition_pct_edges,
                    })
            
            # Create heatmaps for each method
            if state_results:
                df_states = pd.DataFrame(state_results)
                
                for method in ['SBTG', 'Pearson', 'VAR', 'Granger']:
                    method_data = df_states[df_states['method'] == method]
                    
                    if len(method_data) == 0:
                        continue
                    
                    # Create matrix: states × lags
                    lags_sorted = sorted(method_data['lag'].unique())
                    matrix = np.zeros((3, len(lags_sorted)))
                    
                    for i, lag in enumerate(lags_sorted):
                        lag_data = method_data[method_data['lag'] == lag]
                        if len(lag_data) > 0:
                            row = lag_data.iloc[0]
                            matrix[0, i] = row['forward_pct']
                            matrix[1, i] = row['backward_pct']
                            matrix[2, i] = row['transition_pct']
                    
                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
                    
                    # Annotations
                    for i in range(3):
                        for j in range(len(lags_sorted)):
                            val = matrix[i, j]
                            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                                   fontsize=11, fontweight='bold',
                                   color='white' if val > 50 else 'black')
                    
                    ax.set_xticks(range(len(lags_sorted)))
                    ax.set_xticklabels([f'{lag}' for lag in lags_sorted], fontsize=11)
                    ax.set_yticks([0, 1, 2])
                    ax.set_yticklabels(['Forward', 'Backward', 'Transition'], fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time Lag', fontsize=13)
                    ax.set_ylabel('Behavioral State', fontsize=13)
                    ax.set_title(f'{method}: Edge Activity by Behavioral State\n(Percentage of total edge activity)',
                                fontsize=14, fontweight='bold')
                    
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('% of Edge Activity', fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / f'fig_state_activity_{method.lower()}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Saved: fig_state_activity_{method.lower()}.png")

            # Generate COMBINED 2x2 heatmap figure
            if state_results:
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                axes = axes.flatten()
                
                methods_ordered = ['SBTG', 'Pearson', 'VAR', 'Granger']
                
                df_states = pd.DataFrame(state_results)

                for idx, method in enumerate(methods_ordered):
                    ax = axes[idx]
                    method_data = df_states[df_states['method'] == method]
                    
                    if len(method_data) == 0:
                        ax.text(0.5, 0.5, f"No data for {method}", ha='center', va='center')
                        ax.axis('off')
                        continue
                    
                    # Create matrix: states × lags
                    lags_sorted = sorted(method_data['lag'].unique())
                    matrix = np.zeros((3, len(lags_sorted)))
                    
                    for i, lag in enumerate(lags_sorted):
                        lag_data = method_data[method_data['lag'] == lag]
                        if len(lag_data) > 0:
                            row = lag_data.iloc[0]
                            matrix[0, i] = row['forward_pct']
                            matrix[1, i] = row['backward_pct']
                            matrix[2, i] = row['transition_pct']
                    
                    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
                    
                    # Annotations
                    for i in range(3):
                        for j in range(len(lags_sorted)):
                            val = matrix[i, j]
                            color = 'white' if val > 50 else 'black'
                            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                                   fontsize=10, fontweight='bold', color=color)
                    
                    ax.set_xticks(range(len(lags_sorted)))
                    ax.set_xticklabels([f'{lag}' for lag in lags_sorted], fontsize=10)
                    ax.set_yticks([0, 1, 2])
                    ax.set_yticklabels(['Forward', 'Backward', 'Transition'], fontsize=11, fontweight='bold')
                    ax.set_xlabel('Time Lag', fontsize=11)
                    if idx % 2 == 0:
                        ax.set_ylabel('Behavioral State', fontsize=11)
                    
                    ax.set_title(f'{method}', fontsize=14, fontweight='bold')
                
                plt.suptitle('State-Dependent Edge Activity Comparison', fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 0.92, 1])
                
                # Add shared colorbar
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('% of Edge Activity', fontsize=12)
                
                plt.savefig(OUTPUT_DIR / 'fig_state_activity_comparison_all.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: fig_state_activity_comparison_all.png")
        else:
            print("  ⚠ Could not find AVA/AVB neurons, skipping state analysis")
    else:
        print("  ⚠ Calcium traces not found, skipping state analysis")

except Exception as e:
    print(f"  ⚠ Error in state analysis: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# FIGURE 5: MONOAMINE LINE PLOTS
# =============================================================================

print("\n[6/6] Generating monoamine analysis plots...")

trans_colors = {
    'dopamine': '#e41a1c',
    'serotonin': '#377eb8',
    'tyramine': '#4daf4a',
    'octopamine': '#984ea3',
}

trans_labels = {
    'dopamine': 'Dopamine',
    'serotonin': 'Serotonin',
    'tyramine': 'Tyramine',
    'octopamine': 'Octopamine',
}

transmitters = ['dopamine', 'serotonin', 'tyramine', 'octopamine']
methods = ['SBTG', 'Pearson', 'VAR', 'Granger']

# AUROC + AUPRC line plots (2-panel)
for method in methods:
    method_data = df_mono[df_mono['method'] == method]
    
    if len(method_data) == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax_idx, (metric, metric_name, baseline) in enumerate([
        ('auroc', 'AUROC', 0.5),
        ('auprc', 'AUPRC', None),
    ]):
        ax = axes[ax_idx]
        
        for trans in transmitters:
            trans_data = method_data[method_data['transmitter'] == trans].sort_values('lag')
            
            if len(trans_data) > 0:
                ax.plot(trans_data['time_s'], trans_data[metric],
                       marker='o', label=trans_labels[trans],
                       color=trans_colors[trans], linewidth=2.5, markersize=8)
        
        if baseline is not None:
            ax.axhline(baseline, color='gray', linestyle=':', lw=1.5, alpha=0.7, label='Random')
        
        ax.set_xlabel('Time Lag (seconds)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{method}: {MONOAMINE_LABEL} Prediction', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_monoamine_{method.lower()}_lineplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_monoamine_{method.lower()}_lineplot.png")

# F1 score line plots (separate)
for method in methods:
    method_data = df_mono[df_mono['method'] == method]
    
    if len(method_data) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for trans in transmitters:
        trans_data = method_data[method_data['transmitter'] == trans].sort_values('lag')
        
        if len(trans_data) > 0:
            ax.plot(trans_data['time_s'], trans_data['f1'],
                   marker='o', label=trans_labels[trans],
                   color=trans_colors[trans], linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'{method}: F1 Score vs Time Lag', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_monoamine_{method.lower()}_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_monoamine_{method.lower()}_f1.png")

# Comparison heatmaps
for metric, metric_name in [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('f1', 'F1 Score')]:
    lag1_data = df_mono[df_mono['lag'] == 1]
    
    if len(lag1_data) == 0:
        continue
    
    pivot = lag1_data.pivot(index='method', columns='transmitter', values=metric)
    pivot = pivot.reindex(methods)
    pivot = pivot[transmitters]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot.values.min() - 0.01, vmax=pivot.values.max() + 0.01)
    
    for i in range(len(methods)):
        for j in range(len(transmitters)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < (pivot.values.max() + pivot.values.min()) / 2 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(transmitters)))
    ax.set_xticklabels([trans_labels[t] for t in transmitters], fontsize=12)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_xlabel('Neuromodulator', fontsize=13)
    ax.set_ylabel('Method', fontsize=13)
    ax.set_title(f'{metric_name} @ Lag 1 (0.25s): Method × Transmitter',
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_monoamine_comparison_{metric}_lag1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_monoamine_comparison_{metric}_lag1.png")

# =============================================================================
# FIGURE 7: CELL-TYPE BY LAG HEATMAPS
# =============================================================================

print("\n[7/7] Generating cell-type by lag heatmaps...")

# Prepare indices for aggregation
type_indices_dict = {'sensory': [], 'interneuron': [], 'motor': []}
for i, name in enumerate(neuron_names):
    ctype = get_neuron_type(name)
    if ctype in type_indices_dict:
        type_indices_dict[ctype].append(i)

pair_order = [
    ('sensory', 'sensory'), ('sensory', 'interneuron'), ('sensory', 'motor'),
    ('interneuron', 'sensory'), ('interneuron', 'interneuron'), ('interneuron', 'motor'),
    ('motor', 'sensory'), ('motor', 'interneuron'), ('motor', 'motor')
]

pair_labels = [
    'Sen → Sen', 'Sen → Int', 'Sen → Mot',
    'Int → Sen', 'Int → Int', 'Int → Mot',
    'Mot → Sen', 'Mot → Int', 'Mot → Mot'
]

methods_config = {
    'SBTG': {'lags': sbtg_lags, 'data_source': data, 'key_fmt': 'mu_hat_lag{}'},
    'Pearson': {'lags': baseline_lags['Pearson'], 'data_source': baselines, 'key_fmt': 'Pearson_lag{}'},
    'VAR': {'lags': baseline_lags['VAR'], 'data_source': baselines, 'key_fmt': 'VAR_lag{}'},
    'Granger': {'lags': baseline_lags['Granger'], 'data_source': baselines, 'key_fmt': 'Granger_lag{}'},
}

for method_name, config in methods_config.items():
    lags = config['lags']
    source = config['data_source']
    key_fmt = config['key_fmt']
    
    if not lags:
        continue
        
    n_pairs = len(pair_order)
    n_lags = len(lags)
    heatmap_data = np.zeros((n_pairs, n_lags))
    
    for j, lag in enumerate(lags):
        key = key_fmt.format(lag)
        if key not in source:
            continue
            
        mu_hat = source[key]
        weights = np.abs(mu_hat)
        
        if method_name == 'SBTG':
            sig_key = f'sig_lag{lag}'
            if sig_key in source:
                weights = weights * source[sig_key].astype(bool)

        for i, (src_type, tgt_type) in enumerate(pair_order):
            src_idx = type_indices_dict[src_type]
            tgt_idx = type_indices_dict[tgt_type]
            
            if not src_idx or not tgt_idx:
                continue
                
            # Extract submatrix
            sub_weights = weights[np.ix_(tgt_idx, src_idx)]
            
            # Compute mean strength
            heatmap_data[i, j] = np.mean(sub_weights)

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, n_lags * 0.8), 8))
    
    # Dynamic vmin/vmax
    vmax = np.percentile(heatmap_data, 98) if np.max(heatmap_data) > 0 else 1.0
    # Ensure min contrast
    if vmax < 1e-4: vmax = 1e-3
    
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=vmax)
    
    # Annotate
    for i in range(n_pairs):
        for j in range(n_lags):
            val = heatmap_data[i, j]
            color = 'white' if val > vmax/2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=8, color=color)
    
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([f'{l}\n({l/4:.2f}s)' for l in lags], fontsize=9)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Cell-Type Pair', fontsize=12)
    ax.set_title(f'{method_name}: Cell-Type Coupling Strength by Lag', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Mean |Weight| or |Correlation|', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'fig_celltype_by_lag_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: fig_celltype_by_lag_{method_name.lower()}.png")

# =============================================================================
# FIGURE 8: DETAILED EDGE HEATMAPS (PER METHOD)
# =============================================================================

print("\n[8/8] Generating detailed edge heatmaps...")

for method_name, config in methods_config.items():
    lags = config['lags']
    source = config['data_source']
    key_fmt = config['key_fmt']
    
    if not lags:
        continue
    
    # Create method directory
    method_dir = OUTPUT_DIR / f'detailed_couplings/{method_name}'
    method_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Processing {method_name}...")
    
    # Pre-extract all matrices to avoid repeated lookups
    matrices = {}
    for lag in lags:
        key = key_fmt.format(lag)
        if key in source:
            mu_hat = source[key]
            weights = np.abs(mu_hat)
            if method_name == 'SBTG':
                sig_key = f'sig_lag{lag}'
                if sig_key in source:
                    weights = weights * source[sig_key].astype(bool)
                
            matrices[lag] = weights
    
    if not matrices:
        continue
        
    for i, (src_type, tgt_type) in enumerate(pair_order):
        src_idx = type_indices_dict[src_type]
        tgt_idx = type_indices_dict[tgt_type]
        
        if not src_idx or not tgt_idx:
            continue
            
        # Collect all edge weights for this pair
        # We need to map local index -> neuron name
        edges_data = [] # List of (total_weight, [w_lag1, w_lag2...], label)
        
        # Iterate over all possible edges in this block
        for t_local, t_global in enumerate(tgt_idx):
            for s_local, s_global in enumerate(src_idx):
                edge_weights = []
                total_w = 0.0
                
                for lag in lags:
                    if lag in matrices:
                        w = matrices[lag][t_global, s_global]
                        edge_weights.append(w)
                        total_w += w
                    else:
                        edge_weights.append(0.0)
                
                # Filter out completely silent edges (threshold)
                if total_w > 1e-6:
                    label = f"{neuron_names[s_global]} → {neuron_names[t_global]}"
                    edges_data.append((total_w, edge_weights, label))
        
        if not edges_data:
            continue
            
        # Sort by total weight (descending)
        edges_data.sort(key=lambda x: x[0], reverse=True)
        
        K = 100
        if len(edges_data) > K:
             edges_data = edges_data[:K]
             display_limit = True
        else:
             display_limit = False
             
        # Extract for plotting
        plot_matrix = np.array([item[1] for item in edges_data])
        labels = [item[2] for item in edges_data]
        
        # Plot
        n_edges = len(labels)
        if n_edges == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(max(6, len(lags)*0.6), max(4, n_edges * 0.2)))
        
        # Dynamic vmax
        vmax = np.percentile(plot_matrix, 99) if np.max(plot_matrix) > 0 else 1.0
        if vmax < 1e-4: vmax = 1e-3
        
        im = ax.imshow(plot_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lags, fontsize=9)
        ax.set_yticks(range(n_edges))
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_xlabel('Lag')
        title = f'{method_name}: {pair_labels[i]} Edges'
        if display_limit:
            title += f' (Top {K} by Strength)'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5)
        cbar.set_label('|Weight|', fontsize=10)
        
        plt.tight_layout()
        pair_filename = f"pair_{src_type}_{tgt_type}.png"
        plt.savefig(method_dir / pair_filename, dpi=150, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------------------
        # Also generate Line Plot (Superimposed)
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(8, len(lags)*0.8), 6))
        
        # Create line segments: (x, y) points for each line
        segments = []
        for weight_series in plot_matrix:
            # Points are (lag_index, weight)
            # We map lag_index back to lag value or physical time
            pts = list(zip(range(len(lags)), weight_series))
            segments.append(pts)
            
        # Use LineCollection for efficiency and alpha handling
        lc = LineCollection(segments, colors='black', alpha=0.15, linewidths=1.0)
        ax.add_collection(lc)
        
        ax.autoscale()
        # Set x ticks same as heatmap
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([str(l) for l in lags], fontsize=9)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('|Weight|')
        
        title_lines = f'{method_name}: {pair_labels[i]} Lines'
        if display_limit:
            title_lines += f' (Top {K} Superimposed)'
        ax.set_title(title_lines, fontsize=12, fontweight='bold')
        
        # Add mean line in red
        mean_weights = np.mean(plot_matrix, axis=0)
        ax.plot(range(len(lags)), mean_weights, color='red', linewidth=2, label='Mean Strength')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        lines_filename = f"pair_{src_type}_{tgt_type}_lines.png"
        plt.savefig(method_dir / lines_filename, dpi=150, bbox_inches='tight')
        plt.close()

print(f"  ✓ Saved detailed heatmaps and line plots to {OUTPUT_DIR}/detailed_couplings/")

# =============================================================================
# FIGURE 9: SYNAPTIC & MONOAMINE BENCHMARKS
# =============================================================================

print(f"\n[9/9] Generating new benchmark figures ({STRUCTURAL_LABEL}/{FUNCTIONAL_LABEL} + {MONOAMINE_LABEL})...")

# Load baselines metadata
with open(OUTPUT_DIR / 'baselines_info.json') as f:
    baselines_info = json.load(f)
    
pi_cook = baselines_info['pi_cook']
pi_leifer = baselines_info['pi_leifer']
mono_densities = baselines_info['mono_densities']

# -----------------------------------------------------------------------------
# 1. Synaptic Benchmarks Figure (2x2)
# Top: AUROC (structural + functional benchmarks)
# Bottom: AUPRC (structural + functional benchmarks)
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (A) Structural benchmark AUROC
ax = axes[0, 0]
curve = df_cook[df_cook['method'] == 'SBTG'].sort_values('lag')
if len(curve) > 0:
    ax.plot(curve['time_s'], curve['auroc'], color='black', marker='o', lw=2.5, label='SBTG (Full)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
ax.set_title(f'{STRUCTURAL_LABEL} - AUROC', fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=11)
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

# (B) Functional benchmark AUROC
ax = axes[0, 1]
curve = df_leifer[df_leifer['method'] == 'SBTG'].sort_values('lag')
if len(curve) > 0:
    ax.plot(curve['time_s'], curve['auroc'], color='black', marker='o', lw=2.5, label='SBTG (Full)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
ax.set_title(f'{FUNCTIONAL_LABEL} - AUROC', fontsize=12, fontweight='bold')
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)

# (C) Structural benchmark AUPRC
ax = axes[1, 0]
curve = df_cook[df_cook['method'] == 'SBTG'].sort_values('lag')
if len(curve) > 0:
    ax.plot(curve['time_s'], curve['auprc'], color='black', marker='o', lw=2.5, label='SBTG (Full)')
ax.axhline(pi_cook, color='red', linestyle='--', alpha=0.7, label=f'Chance (π={pi_cook:.3f})')
ax.set_title(f'{STRUCTURAL_LABEL} - AUPRC', fontsize=12, fontweight='bold')
ax.set_ylabel('AUPRC', fontsize=11)
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

# (D) Functional benchmark AUPRC
ax = axes[1, 1]
curve = df_leifer[df_leifer['method'] == 'SBTG'].sort_values('lag')
if len(curve) > 0:
    ax.plot(curve['time_s'], curve['auprc'], color='black', marker='o', lw=2.5, label='SBTG (Full)')
ax.axhline(pi_leifer, color='red', linestyle='--', alpha=0.7, label=f'Chance (π={pi_leifer:.3f})')
ax.set_title(f'{FUNCTIONAL_LABEL} - AUPRC', fontsize=12, fontweight='bold')
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

plt.suptitle(f'Global Benchmarks: {STRUCTURAL_LABEL} + {FUNCTIONAL_LABEL}', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig_synaptic_benchmarks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: fig_synaptic_benchmarks.png")

# -----------------------------------------------------------------------------
# 2. Monoamine Benchmarks Figure (2x2: AUROC, placeholder, AUPRC, F1)
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Helper to plot monoamines
def plot_mono_curves(ax, metric_key, baseline_dict=None):
    mono_sbtg = df_mono[df_mono['method'] == 'SBTG']
    for trans in transmitters:
        df_trans = mono_sbtg[mono_sbtg['transmitter'] == trans].sort_values('lag')
        if len(df_trans) > 0:
            ax.plot(df_trans['time_s'], df_trans[metric_key],
                    color=trans_colors[trans], marker='s', lw=2.0, markersize=6,
                    label=trans_labels[trans])
            
            if baseline_dict and metric_key == 'auprc':
                pi = baseline_dict[trans]
                ax.axhline(pi, color=trans_colors[trans], linestyle=':', alpha=0.5, linewidth=1.5)

# (A') Monoamine AUROC
ax = axes[0, 0]
plot_mono_curves(ax, 'auroc')
ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
ax.set_title(f'{MONOAMINE_LABEL} - AUROC', fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=11)
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

# (B') Placeholder / Empty or combined
ax = axes[0, 1]
ax.axis('off')
ax.text(0.5, 0.5, f"{MONOAMINE_LABEL}\n(Details)", ha='center', va='center', fontsize=14, color='gray')

# (C') Monoamine AUPRC
ax = axes[1, 0]
plot_mono_curves(ax, 'auprc', baseline_dict=mono_densities)
ax.set_title(f'{MONOAMINE_LABEL} - AUPRC', fontsize=12, fontweight='bold')
ax.set_ylabel('AUPRC', fontsize=11)
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)
# (D') Monoamine F1
ax = axes[1, 1]
plot_mono_curves(ax, 'f1')
ax.set_title(f'{MONOAMINE_LABEL} - F1 Score', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xlabel('Lag (s)', fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle(f'{MONOAMINE_LABEL} Benchmarks', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / 'fig_monoamine_benchmarks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: fig_monoamine_benchmarks.png")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print(f"\nAll figures saved to: {OUTPUT_DIR}")
print(f"\nGenerated:")
print(f"  - 4 comprehensive comparison plots")
print(f"  - 4 cell-type heatmaps")
print(f"  - 1 cell-type comparison bar plot")
print(f"  - 4 state-dependent activity heatmaps")
print(f"  - 8 monoamine line plots (4 AUROC+AUPRC, 4 F1)")
print(f"  - 3 monoamine comparison heatmaps")
print(f"\nTotal: ~24 figures")
