#!/usr/bin/env python3
"""
Extract Top-K Strongest SBTG Edges

Loads an SBTG results .npz file, iterates over all available lags, and
extracts the strongest directed edges (by absolute mean-transfer magnitude)
that pass the FDR significance mask.  Prints a ranked table per lag and
optionally exports a combined CSV of all lags for downstream analysis.

Usage:
    python analysis/evaluation/extract_strongest_edges.py <path_to_results.npz> [top_k]
"""

import numpy as np
import sys
import pandas as pd
import os

def extract_strongest_edges(npz_path, top_k=15):
    try:
        data = np.load(npz_path)
        neurons = data['neuron_names']
        
        # Identify available lags from keys
        lags = sorted([int(k.split('lag')[1]) for k in data.keys() if k.startswith('mu_hat_lag')])
        
        print(f"Found lags: {lags}")
        print(f"Neurons: {len(neurons)}")
        print("-" * 60)
        
        all_lags_data = []
        
        for lag in lags:
            mu_key = f'mu_hat_lag{lag}'
            sig_key = f'sig_lag{lag}'
            
            if mu_key not in data or sig_key not in data:
                continue
                
            mu = data[mu_key]
            sig = data[sig_key]
            
            # Zero out insignificant edges using the FDR significance mask
            if sig.dtype == bool or np.max(sig) == 1:
                obs_mu = mu * sig
            else:
                obs_mu = np.where(sig > 0, mu, 0)
            
            # Get indices of top_k absolute values
            # Flatten
            flat_indices = np.argsort(np.abs(obs_mu).flatten())[::-1]
            top_indices = flat_indices[:top_k]
            
            print(f"Top {top_k} Strongest Edges for Lag {lag} (250ms * {lag} = {lag*0.25}s):")
            
            lag_data = []
            for idx in top_indices:
                row, col = np.unravel_index(idx, mu.shape)
                # mu[i, j] = mean transfer from neuron j -> neuron i (row=target, col=source)
                target = neurons[row]
                source = neurons[col]
                val = mu[row, col]
                
                if abs(val) < 1e-10: continue
                
                etype = "Excitatory" if val > 0 else "Inhibitory"
                lag_data.append({
                    "Lag": lag,
                    "Time_s": lag * 0.25,
                    "Source": source,
                    "Target": target,
                    "Strength": val,
                    "Type": etype
                })
            
            all_lags_data.extend(lag_data)
        
        # Ensure output directory exists
        output_dir = "results/summary"
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        df = pd.DataFrame(all_lags_data)
        csv_path = os.path.join(output_dir, "strongest_edges.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
        
        # Save to Markdown
        md_path = os.path.join(output_dir, "strongest_edges_report.md")
        with open(md_path, "w") as f:
            f.write("# Strongest Functional Connections by Lag\n\n")
            f.write(f"Generated from: `{npz_path}`\n\n")
            
            for lag in sorted(df['Lag'].unique()):
                lag_df = df[df['Lag'] == lag]
                f.write(f"## Lag {lag} ({lag*0.25}s)\n\n")
                f.write(lag_df.to_markdown(index=False))
                f.write("\n\n")
        print(f"Saved Markdown report to {md_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/evaluation/extract_strongest_edges.py <path_to_npz>")
    else:
        extract_strongest_edges(sys.argv[1])
