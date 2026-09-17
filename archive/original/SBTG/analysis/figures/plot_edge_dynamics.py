#!/usr/bin/env python3
"""
Visualize Edge Weight Dynamics Across Lags

For the top-K edges (ranked by peak absolute mean-transfer over all lags),
generates:
  1. A line-plot of individual edge strengths as a function of lag.
  2. A heatmap of signed edge strengths (excitatory vs inhibitory) showing
     how the strongest inferred connections evolve with time offset.

Figures are saved to the directory specified by output_path.

Usage:
    python analysis/figures/plot_edge_dynamics.py <path_to_results.npz> <output_dir> [top_k]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm
import pandas as pd
import os
import sys

def plot_edge_dynamics(npz_path, output_path, top_k=30):
    try:
        data = np.load(npz_path)
        neurons = data['neuron_names']
        
        # Identify available lags
        lags = sorted([int(k.split('lag')[1]) for k in data.keys() if k.startswith('mu_hat_lag')])
        print(f"Found lags: {lags}")
        
        # Dictionary to store max strength for ranking
        edge_max_strength = {} # (target_idx, source_idx) -> max_abs_val
        
        for lag in lags:
            mu_key = f'mu_hat_lag{lag}'
            sig_key = f'sig_lag{lag}'
            
            if mu_key not in data or sig_key not in data:
                continue
                
            mu = data[mu_key]
            sig = data[sig_key]
            
            # Mask insignificant edges for ranking purposes
            if sig.dtype == bool or np.max(sig) == 1:
                mu_masked = mu * sig
            else:
                mu_masked = np.where(sig > 0, mu, 0)
            
            # Update max strength
            rows, cols = np.where(np.abs(mu_masked) > 0)
            for r, c in zip(rows, cols):
                val = np.abs(mu_masked[r, c])
                if (r, c) not in edge_max_strength:
                    edge_max_strength[(r, c)] = 0.0
                edge_max_strength[(r, c)] = max(edge_max_strength[(r, c)], val)
        
        # Rankings based on max strength
        sorted_edges = sorted(edge_max_strength.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_edge_indices = [k for k, v in sorted_edges]
        
        if not top_edge_indices:
            print("No significant edges found.")
            return

        # Prepare matrix: Rows = Edges, Cols = Lags
        plot_data = np.zeros((len(top_edge_indices), len(lags)))
        edge_labels = []
        
        for i, (r, c) in enumerate(top_edge_indices):
            target = neurons[r]
            source = neurons[c]
            edge_labels.append(f"{source} -> {target}")
            
            for j, lag in enumerate(lags):
                mu = data[f'mu_hat_lag{lag}']
                sig = data[f'sig_lag{lag}']
                
                # Use masked value for plot
                if sig[r, c] > 0:
                    plot_data[i, j] = mu[r, c]
                else:
                    plot_data[i, j] = 0.0
        
        # Time labels
        time_labels = [f"{l*0.25:.2f}s" for l in lags]
        
        # Plotting
        plt.figure(figsize=(12, max(8, top_k * 0.3)))
        limit = np.max(np.abs(plot_data))
        if limit == 0: limit = 1.0 
        
        # SymLogNorm allows log scale for +/- values with a linear region around 0
        norm = SymLogNorm(linthresh=0.03, linscale=0.5, vmin=-limit, vmax=limit)
        
        sns.heatmap(plot_data, 
                    xticklabels=time_labels, 
                    yticklabels=edge_labels,
                    cmap="RdBu_r", 
                    norm=norm, # Apply log normalization
                    cbar_kws={'label': 'Functional Connectivity Strength (Log Scale)'})
        
        plt.title(f"Dynamic Functional Connectivity (Top {top_k} Edges) - Log Scale")
        plt.xlabel("Time Lag (seconds)")
        plt.ylabel("Connection (Source -> Target)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=300)
        print(f"Saved heatmap to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/figures/plot_edge_dynamics.py <path_to_results.npz> <output_png_path>")
    else:
        plot_edge_dynamics(sys.argv[1], sys.argv[2])
