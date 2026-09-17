#!/usr/bin/env python3
"""
Evaluate External Baselines against Cook/Leifer ground truths.

Usage:
    python evaluate_external.py --dataset nacl
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.align import align_matrices
from pipeline.utils.io import load_structural_connectome
from pipeline.utils.leifer import load_leifer_atlas_data
from pipeline.utils.metrics import compute_weight_correlation

def compute_all_metrics(scores, ground_truth, density=0.15):
    """Compute AUROC, AUPRC, F1 (at density threshold), and correlation."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        from scipy.stats import pearsonr
        
        n = scores.shape[0]
        mask = ~np.eye(n, dtype=bool)  # Cook is directed, so exclude only diagonal
        
        y_score = np.abs(scores[mask])
        y_true = (ground_truth[mask] != 0).astype(int)
        
        # Only evaluate on valid numeric bounds
        valid = np.isfinite(y_score) & np.isfinite(y_true)
        if valid.sum() == 0:
            return {"AUROC": 0.5, "AUPRC": 0.0, "F1": 0.0, "Correlation": 0.0}
            
        y_score_v = y_score[valid]
        y_true_v = y_true[valid]
        
        if y_true_v.sum() == 0 or y_true_v.sum() == len(y_true_v):
            return {"AUROC": 0.5, "AUPRC": 0.0, "F1": 0.0, "Correlation": 0.0}
            
        auroc = roc_auc_score(y_true_v, y_score_v)
        auprc = average_precision_score(y_true_v, y_score_v)
        
        if y_score_v.std() > 0 and y_true_v.std() > 0:
            corr, _ = pearsonr(y_score_v, y_true_v)
        else:
            corr = 0.0
            
        # Top-K F1 with exact K edges (avoids >= threshold tie inflation).
        n_edges = int(density * len(y_score_v))
        if n_edges > 0 and n_edges < len(y_score_v):
            topk_idx = np.argpartition(y_score_v, -n_edges)[-n_edges:]
            y_pred = np.zeros_like(y_true_v, dtype=int)
            y_pred[topk_idx] = 1

            tp = (y_pred * y_true_v).sum()
            fp = (y_pred * (1 - y_true_v)).sum()
            fn = ((1 - y_pred) * y_true_v).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        else:
            f1 = 0.0
            
        return {"AUROC": float(auroc), "AUPRC": float(auprc), "F1": float(f1), "Correlation": float(corr)}
    except Exception as e:
        print(f"Metrics err: {e}")
        return {"AUROC": 0.5, "AUPRC": 0.0, "F1": 0.0, "Correlation": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nacl")
    args = parser.parse_args()
    
    results_file = PROJECT_ROOT / "merged_results" / "external_baselines" / f"external_results_{args.dataset}.npz"
    if not results_file.exists():
        print(f"Error: {results_file} not found. Run external_analysis.py first.")
        return
        
    data = np.load(results_file, allow_pickle=True)
    neuron_names = list(data['neuron_names'])
    reference_npz = PROJECT_ROOT / "merged_results" / "result_C_merged.npz"
    
    if len(neuron_names) > 0 and neuron_names[0] == "N0":
        if reference_npz.exists():
            ref_data = np.load(reference_npz, allow_pickle=True)
            if 'neuron_names' in ref_data and len(ref_data['neuron_names']) == len(neuron_names):
                neuron_names = list(ref_data['neuron_names'])
                print(f"Loaded {len(neuron_names)} actual neuron names from {reference_npz.name}.")
    
    # 1. Load Cook
    connectome_dir = PROJECT_ROOT / "results" / "intermediate" / "connectome"
    
    try:
        A_cook_full, cook_nodes, _ = load_structural_connectome(connectome_dir)
        A_cook, common_cook = align_matrices(A_cook_full, cook_nodes, neuron_names)
        has_cook = True
    except Exception as e:
        print(f"Could not load Cook connectome: {e}")
        has_cook = False
        
    # 2. Load Leifer
    try:
        A_leifer_full, _, leifer_nodes = load_leifer_atlas_data()
        A_leifer_full = np.nan_to_num(A_leifer_full, nan=0.0)
        A_leifer, common_leifer = align_matrices(A_leifer_full, leifer_nodes, neuron_names)
        has_leifer = True
    except Exception as e:
        print(f"Could not load Leifer connectome: {e}")
        has_leifer = False

    # Evaluate each method
    print("\n" + "="*50)
    print(f"EVALUATION: EXTERNAL BASELINES ({args.dataset.upper()})")
    print("="*50)
    
    results_list = []
    
    methods = [k for k in data.files if k != 'neuron_names']
    
    if reference_npz.exists():
        ref_data = np.load(reference_npz, allow_pickle=True)
        if 'mu_hat_lag1' in ref_data:
            data_dict = {k: data[k] for k in methods}
            data_dict['SBTG (Lag 1)'] = np.abs(ref_data['mu_hat_lag1']) 
            methods.append('SBTG (Lag 1)')
        else:
            data_dict = {k: data[k] for k in methods}
            print("No mu_hat_lag1 found in reference npz:", ref_data.files)
    else:
        data_dict = {k: data[k] for k in methods}

    for method in methods:
        adj = data_dict[method]
        
        row = {'Method': method}
        
        if has_cook:
            adj_cook, _ = align_matrices(adj, neuron_names, common_cook)
            metrics_cook = compute_all_metrics(adj_cook, A_cook)
            row.update({f'Cook {m}': v for m, v in metrics_cook.items()})
            
        if has_leifer:
            adj_leifer, _ = align_matrices(adj, neuron_names, common_leifer)
            metrics_leifer = compute_all_metrics(adj_leifer, A_leifer)
            row.update({f'Leifer {m}': v for m, v in metrics_leifer.items()})
            
        results_list.append(row)
        
    df = pd.DataFrame(results_list)
    print(df.to_markdown(index=False))
    
    # Save CSV
    out_csv = PROJECT_ROOT / "merged_results" / "external_baselines" / f"evaluation_{args.dataset}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()
