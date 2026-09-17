#!/usr/bin/env python3
"""
Generate clean phase-specific figures for paper-ready reporting.

This script maps internal phase codes (`NOTHING`, `ON`, `SHOWING`, `OFF`)
to display names (`Baseline`, `On`, `Steady`, `Off`) and renders standardized
comparison figures.

Usage:
    python analysis/generate_phase_paper_figures.py

Note:
    `RESULTS_DIR` is currently pinned to a specific run directory.
    Update that constant when reproducing figures from a different run.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.stimulus_periods import get_4period_mask, StimulusPeriod

# Constants
RESULTS_DIR = PROJECT_ROOT / "results/multilag_separation/20260123_112016/4period_analysis/comparison"
OUTPUT_DIR = RESULTS_DIR / "paper_figures"

# Mappings
# Keys match the CSV/Folder names. Values are the new Display Names.
PHASE_MAP = {
    'NOTHING': 'Baseline',
    'ON': 'On',
    'SHOWING': 'Steady',
    'OFF': 'Off'
}

# Colors matching the phases (using the keys from the map)
PERIOD_COLORS = {
    'Baseline': '#7f8c8d',  # Grey
    'On': '#e74c3c',        # Red
    'Steady': '#3498db',    # Blue
    'Off': '#9b59b6',       # Purple
}
SAMPLING_RATE = 4.0

def setup_style():
    """Set aesthetic/paper-ready style."""
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.fonttype'] = 'none'

def plot_clean_heatmaps():
    print("Generating clean heatmaps (Log Scale, Individual Legends)...")
    
    csv_keys = ['NOTHING', 'ON', 'SHOWING', 'OFF']
    data = {}
    
    for key in csv_keys:
        csv_path = RESULTS_DIR / f"celltype_by_lag_{key}.csv"
        if csv_path.exists():
            data[key] = pd.read_csv(csv_path, index_col=0)
        else:
            print(f"Warning: {csv_path} not found")
    
    if not data:
        return

    # Combined Figure: 1x4
    # Increase height to accommodate colorbars below
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    
    for i, key in enumerate(csv_keys):
        if key not in data:
            continue
            
        ax = axes[i]
        df = data[key]
        display_name = PHASE_MAP[key]
        
        # Ensure correct order
        df = df.reindex(pair_order)
        
        # Log Scale Transform
        # Add small epsilon to avoid log(0)
        log_data = np.log10(df + 1e-6)
        
        # Plot heatmap with INDIVIDUAL scalar mapping (auto vmin/vmax per plot)
        # Using LaTeX for log10 to ensure clean rendering across systems
        sns.heatmap(log_data, ax=ax, cmap='RdYlBu_r', 
                    annot=False,
                    cbar=True, 
                    cbar_kws={'orientation': 'horizontal', 'pad': 0.18, 'fraction': 0.05, 
                              'label': r'$\log_{10}(|W|)$'}, # TeX formatting
                    square=False)
        
        # Titles and Labels
        ax.set_title(display_name, fontsize=14, fontweight='bold', 
                     color=PERIOD_COLORS[display_name])
        
        ax.set_xlabel('Lag (frames)', fontsize=10)
        
        # Rename xticks from 'lag1' to '1'
        x_labels = [c.replace('lag', '') for c in df.columns]
        ax.set_xticklabels(x_labels, rotation=0)
        
        if i == 0:
            ax.set_ylabel('Cell-Type Pair', fontsize=12)
            ax.set_yticklabels(pair_order, rotation=0, fontsize=10)
        else:
            ax.set_ylabel('')
            
        # Add colored border
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            # spine.set_color(PERIOD_COLORS[display_name])  # Optional: color the border

    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_celltype_heatmaps_clean.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

def plot_clean_trace_example():
    print("Generating clean trace example...")
    
    fps = 4.0
    duration = 232.0
    n_frames = int(duration * fps)
    
    t = np.arange(n_frames) / fps
    mask = get_4period_mask(n_frames, fps=fps)
    
    # Dummy trace
    np.random.seed(42)
    trace = np.sin(t * 0.5) * 0.5 + np.sin(t * 0.1) * 0.3 + np.random.normal(0, 0.1, n_frames)
    
    # Plot - Small Size
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Map mask values to new labels
    # NOTHING=0, ON=1, SHOWING=2, OFF=3
    # Use StimulusPeriod constants
    
    period_to_val = {
        'Baseline': StimulusPeriod.NOTHING,
        'On': StimulusPeriod.ON,
        'Steady': StimulusPeriod.SHOWING,
        'Off': StimulusPeriod.OFF
    }
    
    # Plot periods
    ordered_display_names = ['Baseline', 'On', 'Steady', 'Off']
    
    for display_name in ordered_display_names:
        color = PERIOD_COLORS[display_name]
        val = period_to_val[display_name]
        
        is_period = (mask == val)
        if not is_period.any(): continue
        
        ax.fill_between(t, -2, 2, where=is_period, 
                        color=color, alpha=0.2, label=display_name, linewidth=0)
                        
    ax.plot(t, trace, color='black', lw=1.0, alpha=0.8)
    
    # Custom Legend below - using displayed names
    # Moved lower to avoid overlap with xlabel
    handles = [plt.Rectangle((0,0),1,1, color=PERIOD_COLORS[p], alpha=0.3) for p in ordered_display_names]
    ax.legend(handles, ordered_display_names, loc='upper center', bbox_to_anchor=(0.5, -0.45), 
              ncol=4, frameon=False, fontsize=10)
    
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([])
    # ax.set_xlabel('Time (s)', fontsize=10, labelpad=5) # Removed per request
    ax.set_title('Stimulus Period Segmentation', fontsize=11, fontweight='bold')
    
    # Skip region
    ax.fill_between(t, -2, 2, where=(mask == -1), color='gray', alpha=0.5, hatch='///', linewidth=0)
    
    sns.despine(left=True)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_trace_seg_clean.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

def plot_clean_edge_counts():
    print("Generating clean edge counts (Renamed)...")
    
    csv_path = RESULTS_DIR / "eval_all_periods.csv"
    if not csv_path.exists():
        print("eval_all_periods.csv not found")
        return
        
    df = pd.read_csv(csv_path)
    
    # Map 'period' column to new names
    # Current values in CSV: NOTHING, ON, SHOWING, OFF
    df['display_period'] = df['period'].map(PHASE_MAP)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ordered_display_names = ['Baseline', 'On', 'Steady', 'Off']
    
    pivot = df.pivot(index='lag', columns='display_period', values='n_fdr_edges')
    pivot = pivot[ordered_display_names] # Reorder
    
    pivot.plot(kind='bar', ax=ax, width=0.8, 
               color=[PERIOD_COLORS[p] for p in ordered_display_names], 
               edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('# Edges (FDR < 0.1)', fontsize=10)
    ax.set_xlabel('Lag (frames)', fontsize=10)
    ax.set_title('Network Density by Phase', fontsize=12, fontweight='bold')
    ax.legend(title=None, fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    sns.despine()
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_edge_counts_clean.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    setup_style()
    plot_clean_heatmaps()
    plot_clean_trace_example()
    plot_clean_edge_counts()
    print("\nAll clean figures generated in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
