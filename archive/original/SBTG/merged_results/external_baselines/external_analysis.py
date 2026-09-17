#!/usr/bin/env python3
"""
Run External Deep Learning Baselines for Functional Connectome Inference

Adapts C. elegans calcium imaging data for three external architectures:
1. NRI  (Neural Relational Inference)   — Kipf et al., 2018
2. NetFormer                            — Chen et al., 2024
3. LINT (Low-Rank Inference)            — Dubreuil / Valente et al., 2022

Each method is trained on sliding-window data and produces an NxN interaction
matrix.  Saved to ``merged_results/external_baselines/external_results_<ds>.npz``.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name="nacl", window_size=10):
    """Load calcium imaging segments and create sliding windows.

    Returns
    -------
    X_train : ndarray, shape [B, N, T]
    nodes   : list of neuron name strings
    """
    data_dir = PROJECT_ROOT / "results" / "intermediate" / "datasets" / dataset_name
    segments_file = data_dir / "X_segments.npy"
    if not segments_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {segments_file}. Run 01_prepare_data.py first."
        )

    X_segments = np.load(segments_file, allow_pickle=True)
    X_list = [np.asarray(X_segments[i], dtype=np.float32)
              for i in range(X_segments.shape[0])]

    windows = []
    for x in X_list:
        T_total, N = x.shape
        for start in range(T_total - window_size + 1):
            windows.append(x[start : start + window_size, :])

    if len(windows) == 0:
        raise ValueError("Not enough timepoints to generate windows.")

    X_train = np.stack(windows).transpose((0, 2, 1))  # [B, N, T]

    names_file = data_dir / "neuron_names.json"
    if names_file.exists():
        with open(names_file, "r") as f:
            nodes = json.load(f)
    else:
        nodes = [f"N{i}" for i in range(X_train.shape[1])]

    return X_train, nodes


# =============================================================================
# 1. NRI — Neural Relational Inference  (Kipf et al., 2018)
# =============================================================================

def run_nri(X, epochs=50, batch_size=64):
    """Train NRI encoder + decoder and extract inferred edge probabilities.

    Uses the full encoder-decoder architecture with Gumbel-Softmax edge
    sampling, reconstruction loss (MSE on predicted next timestep), and
    KL divergence against a uniform prior over edge types.

    Parameters
    ----------
    X : ndarray, shape [B, N, T]
    epochs, batch_size : int

    Returns
    -------
    adj_matrix : ndarray, shape [N, N]   (edge probabilities)
    """
    print("\n--- Running NRI ---")
    nri_dir = str(BASE_DIR / "nri")
    sys.path.insert(0, nri_dir)
    try:
        from modules import MLPEncoder, MLPDecoder
        from utils import encode_onehot
    except ImportError as e:
        print(f"  Skipping NRI: {e}")
        sys.path.remove(nri_dir)
        return None

    B, N, T = X.shape
    edge_types = 2
    n_edges = N * (N - 1)
    n_hid = 128 if N > 50 else 256
    eff_bs = min(batch_size, max(4, 2048 // max(1, n_edges)))

    off_diag = np.ones([N, N]) - np.eye(N)
    rel_rec = torch.FloatTensor(
        np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    )
    rel_send = torch.FloatTensor(
        np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    )

    encoder = MLPEncoder(n_in=T, n_hid=n_hid, n_out=edge_types)
    decoder = MLPDecoder(
        n_in_node=1,
        edge_types=edge_types,
        msg_hid=n_hid,
        msg_out=n_hid,
        n_hid=n_hid,
        skip_first=True,
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=5e-4)

    data = torch.FloatTensor(X)                     # [B, N, T]
    data_4d = data.unsqueeze(-1)                     # [B, N, T, 1]
    target_4d = data_4d[:, :, 1:, :]                 # [B, N, T-1, 1]

    encoder.train()
    decoder.train()
    print(f"  Training encoder+decoder for {epochs} epochs "
          f"(B={B}, N={N}, T={T}, bs={eff_bs}) ...")

    for ep in range(epochs):
        perm = torch.randperm(B)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, B, eff_bs):
            idx = perm[start : start + eff_bs]
            batch_4d = data_4d[idx]
            batch_tgt = target_4d[idx]

            optimizer.zero_grad()

            logits = encoder(batch_4d, rel_rec, rel_send)
            edges = F.gumbel_softmax(logits, tau=0.5, hard=False, dim=-1)
            output = decoder(batch_4d, edges, rel_rec, rel_send, pred_steps=1)

            loss_recon = F.mse_loss(output, batch_tgt)

            prob = F.softmax(logits, dim=-1)
            log_prior = np.log(1.0 / edge_types)
            loss_kl = (prob * (torch.log(prob + 1e-16) - log_prior)).sum(-1).mean()

            loss = loss_recon + 0.01 * loss_kl
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={ep_loss / n_batches:.4f}")

    # --- extract edge probabilities ---
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, B, eff_bs):
            logits = encoder(data_4d[start : start + eff_bs],
                             rel_rec, rel_send)
            chunks.append(torch.softmax(logits, dim=-1)[:, :, 1])

    edge_probs = torch.cat(chunks, dim=0).mean(dim=0)

    adj_matrix = np.zeros((N, N))
    rows, cols = np.where(off_diag)
    for k in range(len(rows)):
        adj_matrix[rows[k], cols[k]] = edge_probs[k].item()

    sys.path.remove(nri_dir)
    return adj_matrix


# =============================================================================
# 2. NetFormer  (Chen et al., 2024)
# =============================================================================

def run_netformer(X, epochs=50, batch_size=64):
    """Train NetFormer on next-step prediction; return learned attention.

    NetFormer infers neuron-to-neuron connectivity through its attention
    mechanism.  We train with MSE loss on one-step-ahead prediction and
    then average the attention matrices over the dataset.

    Parameters
    ----------
    X : ndarray, shape [B, N, T]
    epochs, batch_size : int

    Returns
    -------
    adj_matrix : ndarray, shape [N, N]
    """
    print("\n--- Running NetFormer ---")
    nf_dir = str(BASE_DIR / "NetFormer")
    sys.path.insert(0, nf_dir)
    try:
        from NetFormer.models import NetFormer_sim
    except ImportError as e:
        print(f"  Skipping NetFormer: {e}")
        sys.path.remove(nf_dir)
        return None

    B, N, T = X.shape
    predict_window = 1

    model = NetFormer_sim(
        neuron_num=N,
        window_size=T,
        predict_window_size=predict_window,
        learning_rate=1e-4,
        attention_activation="none",
        dim_E=min(64, N),
    )

    data = torch.FloatTensor(X)
    x_in = data[:, :, : T - predict_window]       # [B, N, T-1]
    y_tgt = data[:, :, T - predict_window :]        # [B, N, 1]

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    print(f"  Training for {epochs} epochs (B={B}, N={N}, T={T}) ...")

    for ep in range(epochs):
        perm = torch.randperm(B)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, B, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            y_hat, _ = model(x_in[idx])
            loss = F.mse_loss(y_hat, y_tgt[idx])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1

        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={ep_loss / n_batches:.4f}")

    # --- extract attention ---
    model.eval()
    attn_sum = torch.zeros(N, N)
    n_samples = 0
    with torch.no_grad():
        for start in range(0, B, batch_size):
            _, attn = model(x_in[start : start + batch_size])
            attn_sum += attn.sum(dim=0).cpu()
            n_samples += attn.shape[0]

    adj_matrix = (attn_sum / n_samples).abs().numpy()
    np.fill_diagonal(adj_matrix, 0)

    sys.path.remove(nf_dir)
    return adj_matrix


# =============================================================================
# 3. LINT — Low-Rank Inference from Neural Trajectories
# =============================================================================

def run_lint(X, epochs=100, rank=5, batch_size=64):
    """Train a low-rank RNN on next-step prediction; return J = m n^T / N.

    Parameters
    ----------
    X : ndarray, shape [B, N, T]
    epochs : int
    rank   : int – rank of the recurrent weight factorisation
    batch_size : int

    Returns
    -------
    adj_matrix : ndarray, shape [N, N]
    """
    print(f"\n--- Running LINT (rank={rank}) ---")
    lint_dir = str(BASE_DIR / "lowrank_inference")
    sys.path.insert(0, lint_dir)
    try:
        from low_rank_rnns.modules import LowRankRNN
    except ImportError as e:
        print(f"  Skipping LINT: {e}")
        sys.path.remove(lint_dir)
        return None

    B, N, T = X.shape

    try:
        X_t = X.transpose(0, 2, 1)                        # [B, T, N]
        _inp = torch.FloatTensor(X_t[:, :-1, :])          # [B, T-1, N]
        _tgt = torch.FloatTensor(X_t[:, 1:, :])           # [B, T-1, N]

        rnn = LowRankRNN(
            input_size=N, hidden_size=N, output_size=N,
            noise_std=0.01, alpha=0.2, rank=rank,
            train_wi=True, train_wo=True, train_wrec=True, train_h0=True,
            non_linearity=torch.tanh, output_non_linearity=torch.tanh,
        )

        optimizer = optim.Adam(rnn.parameters(), lr=1e-3)

        rnn.train()
        print(f"  Training for {epochs} epochs (B={B}, N={N}) ...")
        for ep in range(epochs):
            perm = torch.randperm(B)
            ep_loss, n_batches = 0.0, 0

            for start in range(0, B, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                out = rnn(_inp[idx])
                loss = F.mse_loss(out, _tgt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_batches += 1

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"    Epoch {ep+1}/{epochs}  loss={ep_loss / n_batches:.4f}")

        m = rnn.m.detach().numpy()
        n = rnn.n.detach().numpy()
        adj_matrix = np.abs(m @ n.T) / N

    except Exception as e:
        print(f"  LINT failed: {e}")
        import traceback
        traceback.print_exc()
        adj_matrix = None

    sys.path.remove(lint_dir)
    return adj_matrix


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run external deep-learning baselines for connectome inference"
    )
    parser.add_argument("--dataset", type=str, default="nacl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lint_rank", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=10)
    args = parser.parse_args()

    try:
        X_train, nodes = load_dataset(args.dataset, window_size=args.window_size)
        print(f"Loaded {args.dataset}: {X_train.shape[0]} windows, "
              f"{X_train.shape[1]} neurons, window={X_train.shape[2]}")
    except Exception as e:
        print(f"Dataset load failed: {e}")
        print("Falling back to dummy data for testing ...")
        X_train = np.random.randn(200, 20, 10).astype(np.float32)
        nodes = [f"N{i}" for i in range(20)]

    results = {}

    nri_adj = run_nri(X_train, epochs=args.epochs, batch_size=args.batch_size)
    if nri_adj is not None:
        results["NRI"] = nri_adj

    nf_adj = run_netformer(X_train, epochs=args.epochs,
                           batch_size=args.batch_size)
    if nf_adj is not None:
        results["NetFormer"] = nf_adj

    lint_adj = run_lint(X_train, epochs=args.epochs * 2,
                        rank=args.lint_rank, batch_size=args.batch_size)
    if lint_adj is not None:
        results["LINT"] = lint_adj

    out_dir = PROJECT_ROOT / "merged_results" / "external_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"external_results_{args.dataset}.npz"

    np.savez(out_file, neuron_names=np.array(nodes), **results)
    print(f"\nSaved predictions to {out_file}")


if __name__ == "__main__":
    main()
