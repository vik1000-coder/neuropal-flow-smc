#!/usr/bin/env python3
"""
Sankey Flow Diagrams of Inter-Neuron Connectivity

Generates interactive Sankey diagrams (via Plotly) showing the strongest
directed connections at selected time lags.  Each diagram displays
top-K edges with link width proportional to absolute mean-transfer
strength and colour indicating excitatory (red) vs inhibitory (blue)
coupling.  HTML and static image files are saved per lag.

Usage:
    python analysis/figures/plot_sankey.py <path_to_results.npz> <output_dir>
"""

import numpy as np
import plotly.graph_objects as go
import os
import sys

def generate_sankey(npz_path, output_dir, lags_to_plot=[1, 5, 10, 20], top_k=20):
    try:
        data = np.load(npz_path)
        neurons = data['neuron_names']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for lag in lags_to_plot:
            mu_key = f'mu_hat_lag{lag}'
            sig_key = f'sig_lag{lag}'
            
            if mu_key not in data:
                print(f"Lag {lag} data not found. Skipping.")
                continue
                
            mu = data[mu_key]
            sig = data[sig_key]
            
            # Mask data
            if sig.dtype == bool or np.max(sig) == 1:
                mu_masked = mu * sig
            else:
                mu_masked = np.where(sig > 0, mu, 0)
                
            # Get top K edges by absolute strength
            flat_indices = np.argsort(np.abs(mu_masked).flatten())[::-1]
            top_indices = flat_indices[:top_k]
            
            # Prepare Sankey Data
            sources = []
            targets = []
            values = []
            colors = []
            labels = []
            
            # Source and target layers are separated: neuron X appears as
            # "X (t)" on the left and "X (t+lag)" on the right to avoid loops.
            
            # Collect active neurons
            active_src_indices = set()
            active_tgt_indices = set()
            
            edge_data = []
            
            for idx in top_indices:
                r, c = np.unravel_index(idx, mu.shape)
                val = mu_masked[r, c]
                if abs(val) < 1e-10: continue
                
                src_name = neurons[c]
                tgt_name = neurons[r]
                
                edge_data.append({
                    'src': src_name,
                    'tgt': tgt_name,
                    'val': val
                })
                
            # Prepare Plotly Nodes
            # Source nodes will be indices 0..K, Target nodes K..M
            unique_src = sorted(list(set(d['src'] for d in edge_data)))
            unique_tgt = sorted(list(set(d['tgt'] for d in edge_data)))
            
            src_map = {name: i for i, name in enumerate(unique_src)}
            tgt_map = {name: i + len(unique_src) for i, name in enumerate(unique_tgt)}
            
            node_labels = [f"{name} (t)" for name in unique_src] + [f"{name} (t+{lag*0.25}s)" for name in unique_tgt]
            
            # Prepare Plotly Links
            link_src = []
            link_tgt = []
            link_val = []
            link_color = []
            
            for d in edge_data:
                link_src.append(src_map[d['src']])
                link_tgt.append(tgt_map[d['tgt']])
                link_val.append(abs(d['val']))
                
                # Color: Red for Excitatory, Blue for Inhibitory (with some transparency)
                if d['val'] > 0:
                    link_color.append("rgba(255, 0, 0, 0.6)") # Red
                else:
                    link_color.append("rgba(0, 0, 255, 0.6)") # Blue
            
            # Create Figure
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 20,
                  thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = node_labels,
                  color = "gray"
                ),
                link = dict(
                  source = link_src,
                  target = link_tgt,
                  value = link_val,
                  color = link_color
              ))])
            
            fig.update_layout(title_text=f"Information Flow at Lag {lag} ({lag*0.25}s) - Top {top_k} Edges", font_size=12)
            
            # Save HTML
            html_path = os.path.join(output_dir, f"sankey_lag{lag}.html")
            fig.write_html(html_path)
            print(f"Saved {html_path}")
            
            # Save PNG (requires kaleido)
            # fig.write_image(os.path.join(output_dir, f"sankey_lag{lag}.png"), scale=2)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def generate_multilag_sankey(npz_path, output_dir, lags=[1, 2, 3, 5], top_k=15):
    """
    Generate one Sankey view that fans out from source nodes at t=0 to
    lag-specific target columns (one column per lag).

    This layout preserves direct lag interpretation: each link represents a
    direct t -> t+lag connection, rather than chained transitions.
    """
    try:
        data = np.load(npz_path)
        neurons = data['neuron_names']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Collect edges for all requested lags
        all_edges = []
        
        for lag in lags:
            mu_key = f'mu_hat_lag{lag}'
            sig_key = f'sig_lag{lag}'
            
            if mu_key not in data: continue
            
            mu = data[mu_key]
            sig = data[sig_key]
            
            if sig.dtype == bool or np.max(sig) == 1:
                mu_masked = mu * sig
            else:
                mu_masked = np.where(sig > 0, mu, 0)
            
            # Top K for this lag
            flat_indices = np.argsort(np.abs(mu_masked).flatten())[::-1][:top_k]
            
            for idx in flat_indices:
                r, c = np.unravel_index(idx, mu.shape)
                val = mu_masked[r, c]
                if abs(val) < 1e-10: continue
                
                all_edges.append({
                    'src': neurons[c],
                    'tgt': neurons[r],
                    'val': val,
                    'lag': lag
                })
        
        # Build Nodes
        # We need a Source neuron at t=0, and Target neurons at each Lag
        # Node Name Format: "Neuron (t=0)", "Neuron (t=0.25)", etc.
        
        # To avoid clutter, only include neurons that are part of top edges
        active_sources = sorted(list(set(d['src'] for d in all_edges)))
        
        # Map (Neuron, Time) -> Node Index
        node_map = {}
        node_labels = []
        node_x = []
        node_y = []
        
        # Define X positions for columns
        # 0.05 for Source, then spread others
        x_positions = np.linspace(0.05, 0.95, len(lags) + 1)
        
        # 1. Source Nodes at t=0
        for i, name in enumerate(active_sources):
            node_idx = len(node_labels)
            node_map[(name, 0)] = node_idx
            node_labels.append(f"{name} (t=0)")
            # Fix X position at 0
            node_x.append(x_positions[0])
            # Y could be evenly distributed or auto
            # node_y.append((i + 0.5) / len(active_sources)) 
            # Letting Plotly handle Y is usually safer unless we want strict grid
            node_y.append(None) # Auto
            
        # 2. Target Nodes for each Lag
        for i, lag in enumerate(lags):
            # Find all targets for this lag
            targets_at_lag = sorted(list(set(d['tgt'] for d in all_edges if d['lag'] == lag)))
            
            for name in targets_at_lag:
                node_idx = len(node_labels)
                node_map[(name, lag)] = node_idx
                node_labels.append(f"{name} (Lag {lag})")
                node_x.append(x_positions[i+1])
                node_y.append(None) # Auto
                
        # Build Links
        link_src = []
        link_tgt = []
        link_val = []
        link_color = []
        link_label = []
        
        for d in all_edges:
            s_key = (d['src'], 0)
            t_key = (d['tgt'], d['lag'])
            
            if s_key in node_map and t_key in node_map:
                link_src.append(node_map[s_key])
                link_tgt.append(node_map[t_key])
                link_val.append(abs(d['val']))
                # Recolor: Red=Exc, Blue=Inh
                c = "rgba(255, 0, 0, 0.4)" if d['val'] > 0 else "rgba(0, 0, 255, 0.4)"
                link_color.append(c)
                link_label.append(f"Lag {d['lag']}")

        # Plot
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = node_labels,
              x = [x if x is not None else None for x in node_x], # Manual X assignment
              y = [y if y is not None else None for y in node_y],
              color = "darkgray"
            ),
            link = dict(
              source = link_src,
              target = link_tgt,
              value = link_val,
              color = link_color,
              label = link_label
          ))])
        
        title = f"Multi-Lag Information Flow (Lags {lags}) - Fan Out from t=0"
        fig.update_layout(title_text=title, font_size=10)
        
        out_path = os.path.join(output_dir, "multilag_sankey.html")
        fig.write_html(out_path)
        print(f"Saved {out_path}")

    except Exception as e:
        print(f"Error in multi-lag: {e}")
        import traceback
        traceback.print_exc()


def generate_time_expanded_sankey(npz_path, output_dir, time_steps=[0, 1, 2, 3, 5], top_k=20, mode='combined'):
    """
    Generate a Time-Expanded Sankey Diagram.
    mode: 'combined', 'chain', 'jump'
    """
    try:
        data = np.load(npz_path)
        neurons = data['neuron_names']
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        matrices = {}
        for lag in [1, 2, 3, 5]: 
            mu_key = f'mu_hat_lag{lag}'
            sig_key = f'sig_lag{lag}'
            if mu_key in data:
                mu = data[mu_key]
                sig = data[sig_key]
                if sig.dtype == bool or np.max(sig) == 1:
                    matrices[lag] = mu * sig
                else:
                    matrices[lag] = np.where(sig > 0, mu, 0)

        # 2. Identify Active Neurons (to reduce clutter)
        active_neurons = set()
        for lag, mat in matrices.items():
            flat_indices = np.argsort(np.abs(mat).flatten())[::-1][:top_k]
            for idx in flat_indices:
                r, c = np.unravel_index(idx, mat.shape)
                if abs(mat[r, c]) > 1e-10:
                    active_neurons.add(neurons[r])
                    active_neurons.add(neurons[c])
        
        active_neurons = sorted(list(active_neurons))
        node_indices = {} 
        plotly_labels = []
        plotly_x = []
        plotly_y = []
        
        x_step = 1.0 / (len(time_steps) + 0.5)
        
        for t_idx, t in enumerate(time_steps):
            for name in active_neurons:
                global_idx = len(plotly_labels)
                node_indices[(name, t)] = global_idx
                plotly_labels.append(f"{name} (t={t*0.25:.2f}s)")
                plotly_x.append(0.05 + t_idx * x_step)
                plotly_y.append(None) 
                
        link_src = []
        link_tgt = []
        link_val = []
        link_color = []
        link_label = []
        
        def add_links(src_t, tgt_t, lag):
            if lag not in matrices: return
            mat = matrices[lag]
            
            # Re-filter for top K *at this specific lag step* to ensure we respect "top 20 for each"
            # The previous active_neurons block ensured the *nodes* exist.
            # Now we ensure we only draw the top K edges.
            
            flat_indices = np.argsort(np.abs(mat).flatten())[::-1][:top_k]
            top_edges_indices = set(flat_indices)

            for idx in flat_indices:
                r, c = np.unravel_index(idx, mat.shape)
                val = mat[r, c]
                if abs(val) < 1e-10: continue
                
                src_name = neurons[c]
                tgt_name = neurons[r]
                
                # Only add if nodes are active (they should be, given logic above)
                if (src_name, src_t) in node_indices and (tgt_name, tgt_t) in node_indices:
                    link_src.append(node_indices[(src_name, src_t)])
                    link_tgt.append(node_indices[(tgt_name, tgt_t)])
                    link_val.append(abs(val))
                    
                    c_rgba = "rgba(255, 0, 0, 0.4)" if val > 0 else "rgba(0, 0, 255, 0.4)"
                    link_color.append(c_rgba)
                    link_label.append(f"Lag {lag}")

        # Add Links based on mode
        
        # Chain: t -> t+1
        if mode in ['combined', 'chain']:
            for i in range(len(time_steps) - 1):
                t_curr = time_steps[i]
                t_next = time_steps[i+1]
                diff = t_next - t_curr
                if diff in matrices:
                    add_links(t_curr, t_next, diff)
        
        # Jump: 0 -> t (where t > 1)
        if mode in ['combined', 'jump']:
            for t_target in time_steps[1:]:
                lag = t_target - 0
                if lag in matrices and lag > 1: 
                     add_links(0, t_target, lag)

        # Plot
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = plotly_labels,
              x = plotly_x,
              y = plotly_y,
              color = "darkgray"
            ),
            link = dict(
              source = link_src,
              target = link_tgt,
              value = link_val,
              color = link_color,
              label = link_label
          ))])
        
        title = f"Information Flow ({mode.capitalize()}) - Top {top_k} Edges"
        fig.update_layout(title_text=title, font_size=10)
        
        filename = f"sankey_{mode}.html"
        out_path = os.path.join(output_dir, filename)
        fig.write_html(out_path)
        print(f"Saved {out_path}")

    except Exception as e:
        print(f"Error in time-expanded: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/figures/plot_sankey.py <path_to_results.npz> <output_dir>")
    else:
        # Generate individual lags
        generate_sankey(sys.argv[1], sys.argv[2])
        # Generate fan-out
        generate_multilag_sankey(sys.argv[1], sys.argv[2], lags=[1, 2, 3, 5], top_k=20)
        # Generate time-expanded split
        generate_time_expanded_sankey(sys.argv[1], sys.argv[2], time_steps=[0, 1, 2, 3, 5], top_k=20, mode='chain')
        generate_time_expanded_sankey(sys.argv[1], sys.argv[2], time_steps=[0, 1, 2, 3, 5], top_k=20, mode='jump')
