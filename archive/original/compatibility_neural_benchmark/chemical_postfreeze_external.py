from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from compatibility_neural_benchmark.fair_atlas_analysis import source_bootstrap_ci
from compatibility_neural_benchmark.postfreeze_external_analysis import (
    binary_metrics,
    load_references,
    source_macro_metrics,
)


def evaluate(
    matrix_archive: Path, release: Path, n_boot: int, sbtg_head_run: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    with np.load(matrix_archive, allow_pickle=False) as data:
        chemicals = data["chemical_names"].astype(str)
        neurons = data["neurons"].astype(str)
        delays = data["delay_frames"].astype(int)
        delay_seconds = data["delay_seconds"].astype(float)
        horizons = data["horizon_frames"].astype(int)
        horizon_seconds = data["horizon_seconds"].astype(float)
        schema_version = str(data["stimulus_schema_version"].item())
        schema_fingerprint = str(data["stimulus_schema_fingerprint"].item())
        matrix_keys = [key for key in data.files if key.endswith("_onset_minus_quiet")]
        matrices = {key.removesuffix("_onset_minus_quiet"): data[key] for key in matrix_keys}
    references, networks = load_references(release, neurons.tolist())
    reference_rows = []
    network_rows = []
    relationship_rows = []
    for method_index, (method, array) in enumerate(matrices.items()):
        expected = (len(chemicals), len(delays), len(horizons), len(neurons), len(neurons))
        if array.shape != expected:
            raise RuntimeError(f"matrix shape mismatch for {method}: {array.shape} != {expected}")
        for chemical, chemical_name in enumerate(chemicals):
            for delay, delay_frame in enumerate(delays):
                for horizon, horizon_frame in enumerate(horizons):
                    matrix = array[chemical, delay, horizon]
                    for reference_index, (reference, item) in enumerate(references.items()):
                        reference_rows.append(
                            {
                                "method": method,
                                "chemical": chemical_name,
                                "delay_frames": int(delay_frame),
                                "delay_seconds": float(delay_seconds[delay]),
                                "horizon_frames": int(horizon_frame),
                                "horizon_seconds": float(horizon_seconds[horizon]),
                                "reference": reference,
                                **binary_metrics(matrix, item["labels"], item["mask"]),
                                **source_macro_metrics(matrix, item["labels"], item["mask"]),
                                **source_bootstrap_ci(
                                    matrix,
                                    item["labels"],
                                    item["mask"],
                                    n_boot=n_boot,
                                    seed=(
                                        20_260_828
                                        + 10_007 * method_index
                                        + 1009 * chemical
                                        + 101 * delay
                                        + 17 * horizon
                                        + reference_index
                                    ),
                                ),
                            }
                        )
                    off = ~np.eye(len(neurons), dtype=bool)
                    for network, labels in networks.items():
                        eligible_sources = labels.any(axis=0)
                        for scope, mask in (
                            ("all_pairs_legacy", off),
                            ("eligible_sources", off & eligible_sources[None]),
                        ):
                            network_rows.append(
                                {
                                    "method": method,
                                    "chemical": chemical_name,
                                    "delay_frames": int(delay_frame),
                                    "delay_seconds": float(delay_seconds[delay]),
                                    "horizon_frames": int(horizon_frame),
                                    "horizon_seconds": float(horizon_seconds[horizon]),
                                    "network": network,
                                    "scope": scope,
                                    "n_eligible_sources": int(eligible_sources.sum()),
                                    **binary_metrics(matrix, labels, mask),
                                    **source_macro_metrics(matrix, labels, mask),
                                }
                            )
    sbtg_files: list[str] = []
    if sbtg_head_run is not None:
        paths = sorted((sbtg_head_run / "fold_lag").glob("sbtg_feature_bilinear__f*__lag1__s*.npz"))
        if not paths:
            raise RuntimeError(f"no corrected head-only SBTG lag-1 outputs in {sbtg_head_run}")
        sbtg_matrices = []
        folds = set()
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {"stimulus_schema_fingerprint", "data_lineage", "cohort_mode"}
                if required.difference(data.files):
                    raise RuntimeError(f"SBTG archive lacks corrected lineage: {path}")
                if str(data["stimulus_schema_fingerprint"].item()) != schema_fingerprint:
                    raise RuntimeError("SBTG and lag-matrix stimulus schemas differ")
                if not np.array_equal(data["neurons"].astype(str), neurons):
                    raise RuntimeError("SBTG and lag-matrix neuron orders differ")
                if "no tail pseudo-pairing; no donor-trace copying" not in str(data["data_lineage"].item()):
                    raise RuntimeError("SBTG lineage is not corrected head-only/no-donor")
                folds.add(int(data["fold"]))
                sbtg_matrices.append(data["mu_hat"].astype(float))
                sbtg_files.append(str(path))
        if folds != set(range(5)):
            raise RuntimeError(f"corrected SBTG does not cover five folds: {sorted(folds)}")
        sbtg_matrix = np.mean(sbtg_matrices, axis=0)
        for reference_index, (reference, item) in enumerate(references.items()):
            reference_rows.append(
                {
                    "method": "sbtg_head_corrected_lag1",
                    "chemical": "not_stimulus_specific",
                    "delay_frames": 1,
                    "delay_seconds": 0.25,
                    "horizon_frames": np.nan,
                    "horizon_seconds": np.nan,
                    "reference": reference,
                    **binary_metrics(sbtg_matrix, item["labels"], item["mask"]),
                    **source_macro_metrics(sbtg_matrix, item["labels"], item["mask"]),
                    **source_bootstrap_ci(
                        sbtg_matrix,
                        item["labels"],
                        item["mask"],
                        n_boot=n_boot,
                        seed=20_261_828 + reference_index,
                    ),
                }
            )
        off = ~np.eye(len(neurons), dtype=bool)
        for network, labels in networks.items():
            eligible_sources = labels.any(axis=0)
            for scope, mask in (
                ("all_pairs_legacy", off),
                ("eligible_sources", off & eligible_sources[None]),
            ):
                network_rows.append(
                    {
                        "method": "sbtg_head_corrected_lag1",
                        "chemical": "not_stimulus_specific",
                        "delay_frames": 1,
                        "delay_seconds": 0.25,
                        "horizon_frames": np.nan,
                        "horizon_seconds": np.nan,
                        "network": network,
                        "scope": scope,
                        "n_eligible_sources": int(eligible_sources.sum()),
                        **binary_metrics(sbtg_matrix, labels, mask),
                        **source_macro_metrics(sbtg_matrix, labels, mask),
                    }
                )
        for method, array in matrices.items():
            for chemical, chemical_name in enumerate(chemicals):
                for delay, delay_frame in enumerate(delays):
                    for horizon, horizon_frame in enumerate(horizons):
                        left = array[chemical, delay, horizon][off]
                        right = sbtg_matrix[off]
                        relationship_rows.append(
                            {
                                "method": method,
                                "chemical": chemical_name,
                                "delay_frames": int(delay_frame),
                                "horizon_frames": int(horizon_frame),
                                "horizon_seconds": float(horizon_seconds[horizon]),
                                "comparator": "sbtg_head_corrected_lag1",
                                "pearson": float(pearsonr(left, right).statistic),
                                "spearman": float(spearmanr(left, right).statistic),
                                "sign_agreement": float(np.mean(np.sign(left) == np.sign(right))),
                            }
                        )
    metadata = {
        "matrix_archive": str(matrix_archive),
        "published_release": str(release),
        "methods": sorted(matrices),
        "chemicals": chemicals.tolist(),
        "neurons": neurons.tolist(),
        "source_bootstrap_repeats": n_boot,
        "stimulus_schema_version": schema_version,
        "stimulus_schema_fingerprint": schema_fingerprint,
        "corrected_sbtg_head_run": str(sbtg_head_run) if sbtg_head_run else None,
        "corrected_sbtg_files": sbtg_files,
    }
    return (
        pd.DataFrame(reference_rows),
        pd.DataFrame(network_rows),
        pd.DataFrame(relationship_rows),
        metadata,
    )


def write_report(
    output: Path,
    references: pd.DataFrame,
    networks: pd.DataFrame,
    relationships: pd.DataFrame,
) -> None:
    primary_refs = ["randi_wild_type", "cook_struct_54", "cook_chem_54", "cook_gap_54"]
    primary = references[references.reference.isin(primary_refs)].sort_values(
        ["method", "chemical", "horizon_seconds", "reference"]
    )
    lines = [
        "# Post-freeze chemical lag-matrix correspondence",
        "",
        "## Scope",
        "",
        "These correspondence checks were run only after the predictive encoding gate and lag matrices were frozen. No Cook, Randi, SBTG, receptor, or neuromodulator target entered training, model selection, or sampler tuning.",
        "",
        "| Method | Chemical | Horizon (s) | Reference | AUROC | AUPRC | Source-macro AUROC |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.method} | {row.chemical} | {row.horizon_seconds:g} | {row.reference} | "
            f"{row.auroc:.3f} | {row.auprc:.3f} | {row.macro_source_auroc:.3f} |"
        )
    neuromod = networks[
        networks.network.isin(["monoamine_all", "neuropeptide_all", "neuromodulator_union"])
        & (networks.scope == "eligible_sources")
    ]
    lines.extend(
        [
            "",
            "## Neuromodulator receptor-edge correspondence",
            "",
            "| Method | Chemical | Horizon (s) | Network | AUROC | AUPRC |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in neuromod.sort_values(
        ["method", "chemical", "horizon_seconds", "network"]
    ).itertuples():
        lines.append(
            f"| {row.method} | {row.chemical} | {row.horizon_seconds:g} | {row.network} | "
            f"{row.auroc:.3f} | {row.auprc:.3f} |"
        )
    lines.extend(
        [
            "",
            "Cook is structural connectivity, Randi is a perturbational activity atlas, and the neuromodulator maps record transmitter/receptor-compatible edges. None is ground truth for passive calcium dynamics. AUROC/AUPRC therefore measure descriptive correspondence, not causal validity; a best horizon is not a physical transmission or receptor delay.",
            "",
            "SBTG-published 80 is intentionally absent from the primary table because its local released lineage pseudo-pairs head/tail recordings and copies donor traces. It may be shown only as a separately labeled historical artifact. When `sbtg_head_corrected_lag1` appears above, it was rebuilt from raw head-only recordings with no donor copying on the same 54-neuron order; it is static rather than chemical/onset-specific.",
            "",
        ]
    )
    if len(relationships):
        lines.extend(
            [
                "## Relationship to corrected head-only SBTG lag 1",
                "",
                "| Method | Chemical | Horizon (s) | Spearman | Pearson | Sign agreement |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in relationships.itertuples():
            lines.append(
                f"| {row.method} | {row.chemical} | {row.horizon_seconds:g} | "
                f"{row.spearman:.3f} | {row.pearson:.3f} | {row.sign_agreement:.3f} |"
            )
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-archive", type=Path, required=True)
    parser.add_argument("--published-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bootstrap", type=int, default=1000)
    parser.add_argument("--sbtg-head-run", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references, networks, relationships, metadata = evaluate(
        args.matrix_archive.resolve(),
        args.published_release.resolve(),
        args.source_bootstrap,
        args.sbtg_head_run.resolve() if args.sbtg_head_run else None,
    )
    references.to_csv(output / "randi_cook_correspondence.csv", index=False)
    networks.to_csv(output / "neuromodulator_correspondence.csv", index=False)
    relationships.to_csv(output / "sbtg_head_relationships.csv", index=False)
    write_report(output, references, networks, relationships)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                **metadata,
                "selection_role": "post-freeze descriptive correspondence only",
                "claim_boundary": "reference existence/correspondence is not passive activity, causality, receptor action, or physical delay",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
