#!/usr/bin/env python3
"""
Generate Phase-Specific Detailed Figures

This script iterates over the 4 phases (ON, OFF, NOTHING, SHOWING) from the
multilag separation analysis and generates detailed edge-level visualizations
(heatmaps + line plots) for each method.

Usage:
    python analysis/figures/generate_phase_specific_figures.py

Notes:
    If no explicit run directory is provided, the script selects the latest
    `results/multilag_separation/*/4period_analysis` directory.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

# Add pipeline to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from pipeline.utils.neuron_types import get_neuron_type

# Constants
OUTPUT_ROOT = PROJECT_ROOT / "merged_results/figures/phase_analysis"
PHASES = ['ON', 'OFF', 'NOTHING', 'SHOWING']

def find_latest_phases_root():
    """Find the latest multilag_separation run containing 4period_analysis."""
    base = PROJECT_ROOT / "results" / "multilag_separation"
    candidates = sorted(base.glob("*/4period_analysis"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No 4period_analysis found under {base}")
    return candidates[0]

PHASES_ROOT = find_latest_phases_root()

def process_phase(phase_name):
    print(f"\n{'='*60}")
    print(f"PROCESSING PHASE: {phase_name}")
    print(f"{'='*60}")
    
    phase_dir = PHASES_ROOT / phase_name
    output_dir = OUTPUT_ROOT / phase_name / "detailed_couplings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    result_file = phase_dir / "result.npz"
    baseline_file = phase_dir / "baselines.npz"
    
    if not result_file.exists():
        print(f"  Skipping {phase_name}: result.npz not found")
        return
        
    data = np.load(result_file, allow_pickle=True)
    data = dict(data)
    
    baselines = {}
    if baseline_file.exists():
        baselines = dict(np.load(baseline_file, allow_pickle=True))
    else:
        print("  Warning: baselines.npz not found")

    # Extract metadata
    neuron_names = list(data['neuron_names'])
    
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
    
    # Define method configs
    # Lags might vary per phase or be standard 1,2,3,5
    # Inspect data['lags']
    sbtg_lags = sorted(list(data['lags'])) if 'lags' in data else [1, 2, 3, 5]
    
    # Extract baseline lags
    baseline_lags = {'Pearson': [], 'VAR': [], 'Granger': []}
    for key in baselines.keys():
        for method in baseline_lags.keys():
            if key.startswith(method) and '_lag' in key:
                try:
                    lag = int(key.split('_lag')[-1])
                    baseline_lags[method].append(lag)
                except ValueError:
                    pass
    for method in baseline_lags:
        baseline_lags[method] = sorted(set(baseline_lags[method]))

    methods_config = {
        'SBTG': {'lags': sbtg_lags, 'data_source': data, 'key_fmt': 'mu_hat_lag{}'},
        'Pearson': {'lags': baseline_lags['Pearson'], 'data_source': baselines, 'key_fmt': 'Pearson_lag{}'},
        'VAR': {'lags': baseline_lags['VAR'], 'data_source': baselines, 'key_fmt': 'VAR_lag{}'},
        'Granger': {'lags': baseline_lags['Granger'], 'data_source': baselines, 'key_fmt': 'Granger_lag{}'},
    }

    # Generate Figures
    for method_name, config in methods_config.items():
        lags = config['lags']
        source = config['data_source']
        key_fmt = config['key_fmt']
        
        if not lags:
            continue
        
        # Create method directory
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Generating figures for {method_name}...")
        
        # Pre-extract all matrices
        matrices = {}
        for lag in lags:
            key = key_fmt.format(lag)
            if key in source:
                mu_hat = source[key]
                weights = np.abs(mu_hat)
                # Apply SBTG masking
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
                
            # Collect edge weights
            edges_data = [] 
            
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
                    
                    if total_w > 1e-6:
                        label = f"{neuron_names[s_global]} → {neuron_names[t_global]}"
                        edges_data.append((total_w, edge_weights, label))
            
            if not edges_data:
                continue
                
            # Sort
            edges_data.sort(key=lambda x: x[0], reverse=True)
            
            # Limit K
            K = 100
            display_limit = False
            if len(edges_data) > K:
                 edges_data = edges_data[:K]
                 display_limit = True
                 
            plot_matrix = np.array([item[1] for item in edges_data])
            labels = [item[2] for item in edges_data]
            n_edges = len(labels)
            
            # 1. Heatmap
            fig, ax = plt.subplots(figsize=(max(6, len(lags)*0.6), max(4, n_edges * 0.2)))
            vmax = np.percentile(plot_matrix, 99) if np.max(plot_matrix) > 0 else 1.0
            if vmax < 1e-4: vmax = 1e-3
            
            im = ax.imshow(plot_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
            ax.set_xticks(range(len(lags)))
            ax.set_xticklabels(lags, fontsize=9)
            ax.set_yticks(range(n_edges))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Lag')
            title = f'{method_name} ({phase_name}): {pair_labels[i]} Edges'
            if display_limit: title += f' (Top {K})'
            ax.set_title(title, fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.5, label='|Weight|')
            plt.tight_layout()
            plt.savefig(method_dir / f"pair_{src_type}_{tgt_type}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Line Plot
            fig, ax = plt.subplots(figsize=(max(8, len(lags)*0.8), 6))
            segments = []
            for weight_series in plot_matrix:
                pts = list(zip(range(len(lags)), weight_series))
                segments.append(pts)
            
            lc = LineCollection(segments, colors='black', alpha=0.15, linewidths=1.0)
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_xticks(range(len(lags)))
            ax.set_xticklabels([str(l) for l in lags], fontsize=9)
            ax.set_xlabel('Lag')
            ax.set_ylabel('|Weight|')
            title_lines = f'{method_name} ({phase_name}): {pair_labels[i]} Lines'
            if display_limit: title_lines += f' (Top {K})'
            ax.set_title(title_lines, fontsize=12, fontweight='bold')
            
            mean_weights = np.mean(plot_matrix, axis=0)
            ax.plot(range(len(lags)), mean_weights, color='red', linewidth=2, label='Mean Strength')
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(method_dir / f"pair_{src_type}_{tgt_type}_lines.png", dpi=150, bbox_inches='tight')
            plt.close()

    print(f"  ✓ Phase {phase_name} complete.")

def main():
    print("STARTING PHASE-SPECIFIC ANALYSIS")
    for phase in PHASES:
        process_phase(phase)
    print("\nALL PHASES COMPLETE")

if __name__ == "__main__":
    main()
