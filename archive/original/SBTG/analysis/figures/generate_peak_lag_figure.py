#!/usr/bin/env python3
"""
Generate Peak Lag Figure for Paper
Converts the Peak Time Lags table into a professional bar chart.
Style adapted from pipeline/generate_phase_paper_figures.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.utils.display_names import FUNCTIONAL_LABEL, MONOAMINE_LABEL, STRUCTURAL_LABEL

OUTPUT_DIR = Path("merged_results/figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def setup_style():
    """Set aesthetic/paper-ready style matching generate_phase_paper_figures.py"""
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['svg.fonttype'] = 'none'

def main():
    setup_style()
    
    # Data from Table
    data = [
        {'Network': f'{MONOAMINE_LABEL}: Dopamine',   'Category': 'Monoamine',  'Peak': 0.75, 'Interp': 0.82},
        {'Network': f'{MONOAMINE_LABEL}: Serotonin',  'Category': 'Monoamine',  'Peak': 5.00, 'Interp': 5.00},
        {'Network': f'{MONOAMINE_LABEL}: Tyramine',   'Category': 'Monoamine',  'Peak': 0.50, 'Interp': 0.55},
        {'Network': f'{MONOAMINE_LABEL}: Octopamine', 'Category': 'Monoamine',  'Peak': 0.75, 'Interp': 0.87},
        {'Network': STRUCTURAL_LABEL,                'Category': 'Structural', 'Peak': 0.25, 'Interp': 0.25},
        {'Network': FUNCTIONAL_LABEL,                'Category': 'Functional', 'Peak': 2.00, 'Interp': 1.88},
    ]
    df = pd.DataFrame(data)
    
    # Clean up labels for plotting
    # Keep dataset labels canonical while shortening monoamine rows to transmitter suffix.
    df['Label'] = df['Network'].str.replace(f'{MONOAMINE_LABEL}: ', '', regex=False)
    
    # Sort by Interp Lag ascending
    df = df.sort_values('Interp', ascending=True)
    
    # Define Colors
    category_colors = {
        'Structural': '#e74c3c',
        'Functional': '#3498db',
        'Monoamine':  '#95a5a6' 
    }
    
    colors = []
    for _, row in df.iterrows():
        if row['Category'] == 'Structural':
            colors.append('#e74c3c') # Red
        elif row['Category'] == 'Functional':
            colors.append('#3498db') # Blue
        else:
            # Monoamines
            if 'Dopamine' in row['Label']: colors.append('#66a61e')
            elif 'Serotonin' in row['Label']: colors.append('#e6ab02')
            elif 'Tyramine' in row['Label']: colors.append('#666666')
            elif 'Octopamine' in row['Label']: colors.append('#a6761d')
            else: colors.append('#95a5a6')

    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Horizontal Bar Plot
    bars = ax.barh(df['Label'], df['Interp'], color=colors, alpha=0.8, height=0.6)
    
    for bar, val in zip(bars, df['Interp']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.2f}s", 
                va='center', fontsize=10, color='black')
        
    ax.set_xlabel('Time Lag (s)', fontsize=11, fontweight='bold')
    ax.set_title('Estimated Peak Matching Time Points', fontsize=12, fontweight='bold')
    
    # Add Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Despine
    sns.despine(left=True, bottom=False)
    
    # Remove y ticks lines but keep labels
    ax.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_peak_lags.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    main()
