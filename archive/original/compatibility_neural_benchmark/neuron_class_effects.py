"""Descriptive class-to-class summaries of the frozen NeuroPAL atlas.

No fitting, reference optimization, or edge-level independence assumptions.
The primary class labels come from the actual Cook SI6 workbook. Uncertainty
resamples whole worm contributions, conditional on the existing fitted models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("sensory", "interneuron", "motor")
PRIMARY = "progressive_bridge_smc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_ledger(root: Path) -> int:
    entries = (root / "checksums.sha256").read_text().splitlines()
    for line in entries:
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected:
            raise ValueError(f"Checksum mismatch: {root / relative}")
    return len(entries)


def align_classes(frame: pd.DataFrame, neurons: np.ndarray) -> pd.DataFrame:
    if frame.neuron.duplicated().any():
        raise ValueError("Duplicate neuron classification")
    if set(frame.neuron) != set(neurons.tolist()):
        raise ValueError("Classification must match the exact atlas vocabulary")
    if not set(frame.primary_class).issubset(CLASSES):
        raise ValueError("Unexpected primary class")
    out = frame.set_index("neuron").loc[neurons.tolist()].reset_index()
    flags = out.mixed_or_disputed.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Mixed-role flags must be true/false")
    out["mixed_or_disputed"] = flags.eq("true")
    return out


def edge_mask(labels, source_class, target_class, included, sources):
    """Return [target, source] mask, excluding same-class-coordinate self edges."""
    labels = np.asarray(labels)
    keep_source = included & sources & (labels == source_class)
    keep_target = included & (labels == target_class)
    mask = keep_target[:, None] & keep_source[None, :]
    np.fill_diagonal(mask, False)
    return mask


def strong_sources(atlas, method, context, threshold=0.8):
    valid = atlas[f"valid_fraction__{method}__{context}"]
    keep = np.all(valid >= threshold, axis=0)
    if method == PRIMARY:
        keep &= np.all(
            atlas[f"genealogy_valid_fraction_0_10__{method}__{context}"] >= threshold,
            axis=0,
        )
    return keep


def block_values(matrix, mask):
    """Average pair effects; no sum that rewards larger classes."""
    if not mask.any():
        shape = matrix.shape[:-2]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    selected = matrix[..., mask]
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite effect in eligible block")
    return selected.mean(axis=-1), np.abs(selected).mean(axis=-1)


def bootstrap_weights(n_worms, repetitions, seed):
    draws = np.random.default_rng(seed).integers(0, n_worms, (repetitions, n_worms))
    return np.stack([np.bincount(row, minlength=n_worms) for row in draws]) / n_worms


def intervals(worm_values, weights):
    """Input [lag,worm,horizon]; descriptive pointwise 95% intervals."""
    flat = worm_values.transpose(1, 0, 2).reshape(worm_values.shape[1], -1)
    sample_means = weights @ flat
    bounds = np.quantile(sample_means, [0.025, 0.975], axis=0)
    return bounds.reshape(2, worm_values.shape[0], worm_values.shape[2])


def analyze(atlas_dir: Path, classification_path: Path, output_dir: Path,
            repetitions: int = 2000, seed: int = 20260831):
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "class_effects.csv").exists():
        raise FileExistsError("Use a fresh output directory; existing results are immutable")
    verified = verify_ledger(atlas_dir)
    classification = pd.read_csv(classification_path)
    rows, masks, denominators, worm_arrays = [], [], [], {}
    max_reconstruction_error = 0.0
    max_group_reconstruction_error = 0.0
    with np.load(atlas_dir / "atlas_matrices.npz", allow_pickle=False) as atlas, np.load(
        atlas_dir / "worm_matrices.npz", allow_pickle=False
    ) as worms:
        neurons = atlas["neurons"]
        if str(atlas["orientation"].item()) != "target_row_source_column":
            raise ValueError("Unexpected matrix orientation")
        for key in ["neurons", "worm_ids", "source_lag_frames", "horizon_frames"]:
            if not np.array_equal(atlas[key], worms[key]):
                raise ValueError(f"Atlas/worm metadata mismatch: {key}")
        if str(worms["primary_method"].item()) != PRIMARY:
            raise ValueError("Worm arrays must represent progressive bridge SMC")
        labels_frame = align_classes(classification, neurons)
        labels = labels_frame.primary_class.to_numpy()
        methods, channels, contexts = (atlas[k].tolist() for k in ["methods", "channels", "contexts"])
        lags, horizons = atlas["source_lag_frames"], atlas["horizon_frames"]
        weights = bootstrap_weights(len(worms["worm_ids"]), repetitions, seed)
        worm_arrays.update({key: worms[key] for key in ["neurons", "worm_ids", "source_lag_frames", "horizon_frames"]})
        roster_masks = {
            "cook_primary": np.ones(len(neurons), bool),
            "exclude_mixed_disputed": ~labels_frame.mixed_or_disputed.to_numpy(),
        }
        for roster, included in roster_masks.items():
            for src in CLASSES:
                for tgt in CLASSES:
                    pair_mask = edge_mask(labels, src, tgt, included, np.ones(len(neurons), bool))
                    denominators.append(dict(roster=roster, source_class=src, target_class=tgt,
                                             source_classes=int(np.sum(included & (labels == src))),
                                             target_classes=int(np.sum(included & (labels == tgt))),
                                             possible_directed_pairs=int(pair_mask.sum())))
        for method in methods:
            for context in contexts:
                own_support = strong_sources(atlas, method, context)
                policies = {"all_estimated": np.ones(len(neurons), bool),
                            "strong_common_lags": own_support}
                for policy, source_keep in policies.items():
                    for n, keep in zip(neurons, source_keep):
                        masks.append(dict(method=method, context=context, support_policy=policy,
                                          neuron=str(n), source_included=bool(keep)))
                for channel in channels:
                    is_signed = channel != "endpoint_wasserstein1" or context.endswith("_minus_baseline")
                    mean_matrix = atlas[f"mean_normalized__{method}__{channel}__{context}"].astype(float)
                    worm_matrix = None
                    if method == PRIMARY:
                        worm_matrix = worms[f"normalized__{channel}__{context}"].astype(float)
                        error = float(np.max(np.abs(worm_matrix.mean(axis=1) - mean_matrix)))
                        max_reconstruction_error = max(max_reconstruction_error, error)
                        if error > 2e-6:
                            raise ValueError("Worm means do not reconstruct canonical matrix")
                    for roster, included in roster_masks.items():
                        for policy, source_keep in policies.items():
                            pooled_sum = np.zeros(mean_matrix.shape[:2])
                            total_pairs = 0
                            for src in CLASSES:
                                for tgt in CLASSES:
                                    mask = edge_mask(labels, src, tgt, included, source_keep)
                                    n_pairs = int(mask.sum())
                                    signed, magnitude = block_values(mean_matrix, mask)
                                    if n_pairs:
                                        pooled_sum += signed * n_pairs
                                    total_pairs += n_pairs
                                    ci = np.full((2, len(lags), len(horizons)), np.nan)
                                    worm_magnitude = np.full((len(lags), len(horizons)), np.nan)
                                    worm_mag_ci = ci.copy()
                                    sign_fraction = np.full_like(signed, np.nan)
                                    if worm_matrix is not None and n_pairs:
                                        worm_signed, worm_abs = block_values(worm_matrix, mask)
                                        ci = intervals(worm_signed, weights)
                                        worm_mag_ci = intervals(worm_abs, weights)
                                        worm_magnitude = worm_abs.mean(axis=1)
                                        if is_signed:
                                            sign_fraction = np.maximum((worm_signed > 0).mean(axis=1), (worm_signed < 0).mean(axis=1))
                                        key = f"{roster}__{policy}__{channel}__{context}__{src}_to_{tgt}"
                                        worm_arrays[f"signed__{key}"] = worm_signed.astype(np.float32)
                                        worm_arrays[f"mean_absolute__{key}"] = worm_abs.astype(np.float32)
                                    for li, lag in enumerate(lags):
                                        for hi, horizon in enumerate(horizons):
                                            rows.append(dict(
                                                method=method, roster=roster, support_policy=policy,
                                                channel=channel, effect_is_signed=is_signed, context=context, source_class=src,
                                                target_class=tgt, source_lag_frames=int(lag),
                                                horizon_frames=int(horizon), lag_seconds=float(atlas["source_lag_seconds"][li]),
                                                horizon_seconds=float(atlas["horizon_seconds"][hi]),
                                                n_sources=int(np.sum(included & source_keep & (labels == src))),
                                                n_targets=int(np.sum(included & (labels == tgt))), n_pairs=n_pairs,
                                                signed_mean=float(signed[li, hi]),
                                                mean_absolute_pooled_edge=float(magnitude[li, hi]),
                                                signed_ci_low=float(ci[0, li, hi]), signed_ci_high=float(ci[1, li, hi]),
                                                mean_absolute_within_worm=float(worm_magnitude[li, hi]),
                                                within_worm_abs_ci_low=float(worm_mag_ci[0, li, hi]),
                                                within_worm_abs_ci_high=float(worm_mag_ci[1, li, hi]),
                                                worm_sign_fraction=float(sign_fraction[li, hi]),
                                                n_worms=len(worms["worm_ids"]),
                                                uncertainty_status="descriptive_worm_bootstrap" if method == PRIMARY and n_pairs else "not_available",
                                                status="estimated" if n_pairs else "no_eligible_pairs",
                                            ))
                            all_mask = included[:, None] & (included & source_keep)[None, :]
                            np.fill_diagonal(all_mask, False)
                            assert int(all_mask.sum()) == total_pairs
                            if total_pairs:
                                reconstructed = pooled_sum / total_pairs
                                direct = mean_matrix[..., all_mask].mean(axis=-1)
                                max_group_reconstruction_error = max(max_group_reconstruction_error, float(np.max(np.abs(reconstructed-direct))))
        labels_frame.to_csv(output_dir / "aligned_classification.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "class_effects.csv", index=False)
    pd.DataFrame(rows).to_parquet(output_dir / "class_effects.parquet", index=False)
    pd.DataFrame(denominators).to_csv(output_dir / "class_denominators.csv", index=False)
    pd.DataFrame(masks).to_csv(output_dir / "source_support_masks.csv", index=False)
    np.savez_compressed(output_dir / "class_worm_effects.npz", **worm_arrays)
    metadata = dict(
        schema_version="neuron_class_effects_v1", status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        primary_class_counts=labels_frame.primary_class.value_counts().to_dict(),
        excluded_mixed_disputed=labels_frame.loc[labels_frame.mixed_or_disputed, "neuron"].tolist(),
        row_count=len(rows), canonical_ledger_verified_entries=verified,
        source_hashes={str(p): sha256(p) for p in [atlas_dir / "atlas_matrices.npz", atlas_dir / "worm_matrices.npz", atlas_dir / "checksums.sha256", classification_path, Path(__file__)]},
        bootstrap=dict(repetitions=repetitions, seed=seed, unit="held-out worm", interval="pointwise percentile 95%; conditional fitted-model sensitivity, no multiplicity correction"),
        max_worm_mean_reconstruction_error=max_reconstruction_error,
        max_pair_weighted_group_reconstruction_error=max_group_reconstruction_error,
        definitions={
            "orientation": "target rows, source columns; group names source_to_target",
            "self_edges": "excluded; pooled class-coordinate diagonal, not individual cell self-synapses",
            "signed_mean": "equal-weight mean over directed class-coordinate pairs of canonical worm-mean normalized effects; within-state W1 is unsigned, not directional (see effect_is_signed)",
            "mean_absolute_pooled_edge": "mean over pairs of absolute canonical worm-mean effects; not abs(signed_mean)",
            "mean_absolute_within_worm": "absolute per-worm pair effects averaged over pairs then worms; includes heterogeneity and Monte Carlo noise",
            "normalization": "inherited event-wise response / max(abs(achieved_source_gap),0.10); no second normalization",
            "strong_common_lags": "source validity >=0.8 at ALL four lags in selected context; progressive additionally genealogy-valid(f=0.10) >=0.8 at all lags; source sets fixed across lags/horizons, but can differ by context/method",
            "direct_uncertainty": "point estimates only; canonical worm archive contains progressive only; no borrowed intervals",
            "claim_boundary": "post-hoc descriptive fitted-law effects, not class causal influence or validated physical delays; no new significance claims",
        },
    )
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.atlas_dir, args.classification, args.output_dir, args.bootstrap_repetitions), indent=2))


if __name__ == "__main__":
    main()
