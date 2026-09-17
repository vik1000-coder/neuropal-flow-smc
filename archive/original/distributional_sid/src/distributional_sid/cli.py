from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import (
    GaussianRunConfig,
    MomentBlindRunConfig,
    freeze_source_and_environment,
    package_confirmation_run,
    package_development_run,
    package_e4_confirmation,
    run_e2,
    run_e0,
    run_gaussian_seed,
    run_moment_blind_seed,
    run_support_motion_seed,
    write_frozen_config,
)
from .core import atomic_csv, atomic_json


def _force_thread_limits() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def command_development(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = GaussianRunConfig(
        n_train=args.n_train,
        n_bank=args.n_bank,
        n_folds=args.folds,
        history_dim=args.history_dim,
        response_dim=args.response_dim,
        query_count=args.queries,
        random_width=args.width,
        feature_ridge=args.feature_ridge,
        riesz_ridge=args.riesz_ridge,
        qmc_power=args.qmc_power,
        stage="development",
    )
    seeds = list(range(args.seed_start, args.seed_end + 1))
    inits = list(range(args.initializations))
    payload = {
        "command": "development",
        "config": asdict(config),
        "seeds": seeds,
        "initializations": inits,
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    for seed in seeds:
        for method_init in inits:
            print(f"E1 seed={seed} init={method_init}", flush=True)
            run_gaussian_seed(run_dir, seed, method_init, config, save_scores=not args.no_scores)
    print("E2 controlled perturbations", flush=True)
    _, gate = run_e2(run_dir, stage="development", qmc_power=args.e2_qmc_power)
    validation = package_development_run(run_dir, expected_cases=len(seeds) * len(inits))
    print(gate.to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_e0(args: argparse.Namespace) -> int:
    _force_thread_limits()
    payload = run_e0(Path(args.run_dir).resolve())
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["passed"] else 2


def command_tune(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    widths = [32, 64, 128]
    feature_ridges = [1e-4, 1e-3]
    riesz_ridges = [1e-4, 1e-3, 1e-2]
    payload = {
        "command": "tune",
        "stage": "development",
        "seeds": [1, 2],
        "initializations": [0],
        "widths": widths,
        "feature_ridges": feature_ridges,
        "riesz_ridges": riesz_ridges,
        "n_train": args.n_train,
        "n_bank": args.n_bank,
        "folds": args.folds,
        "history_dim": args.history_dim,
        "response_dim": args.response_dim,
        "queries": args.queries,
        "qmc_power": args.qmc_power,
        "selection_rule": "mean_orth_nrmse + .25*abs(slope-1) + .25*alpha_nrmse + .10*min(feature_derivative_nrmse,10) + probe_q95",
    }
    write_frozen_config(run_dir, payload)
    rows: list[dict[str, object]] = []
    for width in widths:
        for feature_ridge in feature_ridges:
            for riesz_ridge in riesz_ridges:
                combo_id = f"w{width}_fm{feature_ridge:.0e}_ra{riesz_ridge:.0e}".replace("-", "m")
                combo_dir = run_dir / "grid" / combo_id
                config = GaussianRunConfig(
                    n_train=args.n_train,
                    n_bank=args.n_bank,
                    n_folds=args.folds,
                    history_dim=args.history_dim,
                    response_dim=args.response_dim,
                    query_count=args.queries,
                    random_width=width,
                    feature_ridge=feature_ridge,
                    riesz_ridge=riesz_ridge,
                    qmc_power=args.qmc_power,
                    stage="development_tuning",
                )
                metric_frames = []
                nuisance_frames = []
                for seed in (1, 2):
                    print(f"tune {combo_id} seed={seed}", flush=True)
                    metrics, nuisances = run_gaussian_seed(
                        combo_dir, seed, 0, config, save_scores=False
                    )
                    metric_frames.append(metrics)
                    nuisance_frames.append(nuisances)
                metrics = pd.concat(metric_frames, ignore_index=True)
                nuisances = pd.concat(nuisance_frames, ignore_index=True)
                orth = metrics[
                    (metrics.method == "ORTH")
                    & metrics.metric_name.isin(["nrmse", "calibration_slope"])
                ]
                means = orth.groupby("metric_name").metric_value.mean()
                orth_nrmse = float(means["nrmse"])
                slope = float(means["calibration_slope"])
                alpha_nrmse = float(nuisances.alpha_nrmse.mean())
                derivative_nrmse = float(nuisances.feature_derivative_nrmse.mean())
                probe_q95 = float(nuisances.probe_q95_abs.mean())
                objective = (
                    orth_nrmse
                    + 0.25 * abs(slope - 1.0)
                    + 0.25 * alpha_nrmse
                    + 0.10 * min(derivative_nrmse, 10.0)
                    + probe_q95
                )
                rows.append(
                    {
                        "combo_id": combo_id,
                        "random_width": width,
                        "feature_ridge": feature_ridge,
                        "riesz_ridge": riesz_ridge,
                        "mean_orth_nrmse": orth_nrmse,
                        "mean_orth_calibration_slope": slope,
                        "mean_alpha_nrmse": alpha_nrmse,
                        "mean_feature_derivative_nrmse": derivative_nrmse,
                        "mean_probe_q95_abs": probe_q95,
                        "selection_objective": objective,
                    }
                )
    frame = pd.DataFrame(rows).sort_values("selection_objective").reset_index(drop=True)
    frame["selected"] = False
    frame.loc[0, "selected"] = True
    atomic_csv(run_dir / "tuning_results.csv", frame)
    selected = frame.iloc[0].to_dict()
    atomic_json(run_dir / "selected_config.json", selected)
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(selected, indent=2), flush=True)
    return 0


def command_confirm_e1(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = GaussianRunConfig(
        n_train=8000,
        n_bank=1024,
        n_folds=5,
        history_dim=8,
        response_dim=4,
        query_count=64,
        random_width=32,
        feature_ridge=1e-3,
        riesz_ridge=1e-2,
        qmc_power=15,
        stage="confirmation",
    )
    seeds = list(range(1001, 1031))
    initializations = 3
    payload = {
        "command": "confirm-e1",
        "config": asdict(config),
        "seeds": seeds,
        "initializations": list(range(initializations)),
        "development_selection_artifact": "distributional_sid/runs/development_tuning_20260714/selected_config.json",
        "score_storage": "initialization-averaged ORTH contributions per DGP seed",
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    for seed in seeds:
        marker = run_dir / "completed_seeds" / f"seed_{seed}.json"
        if marker.exists():
            print(f"E1 confirmation seed={seed} complete", flush=True)
            continue
        temporary_scores = []
        for method_init in range(initializations):
            print(f"E1 confirmation seed={seed} init={method_init}", flush=True)
            run_gaussian_seed(
                run_dir,
                seed,
                method_init,
                config,
                save_scores=True,
                score_methods=("ORTH",),
            )
            temporary_scores.append(
                run_dir / "scores" / f"e1_gaussian__seed_{seed}__init_{method_init}.npz"
            )
        arrays = [np.load(path) for path in temporary_scores]
        orth = np.mean(np.stack([item["orth"] for item in arrays], axis=0), axis=0)
        truth = arrays[0]["truth"]
        from .core import atomic_npz

        atomic_npz(
            run_dir / "scores_seed_average" / f"e1_gaussian__seed_{seed}.npz",
            orth=orth.astype(np.float32),
            truth=truth.astype(np.float64),
            dgp_seed=np.array([seed], dtype=np.int64),
            initialization_count=np.array([initializations], dtype=np.int16),
        )
        for item in arrays:
            item.close()
        for path in temporary_scores:
            path.unlink()
        atomic_json(marker, {"dgp_seed": seed, "initializations": initializations, "passed": True})
    validation = package_confirmation_run(run_dir, expected_seeds=len(seeds), initializations=initializations)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_develop_e4(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = MomentBlindRunConfig(stage="development")
    payload = {
        "command": "develop-e4",
        "config": asdict(config),
        "families": ["finite_difference_mixture", "continuous_legendre_tilt"],
        "seeds": [3, 4],
        "initializations": [0],
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    metrics = []
    nuisances = []
    for family in payload["families"]:
        for seed in payload["seeds"]:
            print(f"E4 development family={family} seed={seed}", flush=True)
            metric, nuisance = run_moment_blind_seed(
                run_dir, family, seed, 0, config, save_scores=True
            )
            metrics.append(metric)
            nuisances.append(nuisance)
    metric_frame = pd.concat(metrics, ignore_index=True)
    nuisance_frame = pd.concat(nuisances, ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", metric_frame)
    atomic_csv(run_dir / "nuisance_metrics.csv", nuisance_frame)
    passed = bool(
        np.all(metric_frame.fit_status == "ok")
        and np.all(np.isfinite(metric_frame.metric_value))
        and len(list((run_dir / "scores").glob("*.npz"))) == 4
    )
    atomic_json(run_dir / "validation.json", {"passed": passed, "case_count": 4})
    print(metric_frame[metric_frame.metric_name.isin(["nrmse", "calibration_slope", "any_false_call_bonferroni"])].to_string(index=False), flush=True)
    return 0 if passed else 2


def command_confirm_e4(args: argparse.Namespace) -> int:
    _force_thread_limits()
    from scipy.stats import norm
    from .core import atomic_npz

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = MomentBlindRunConfig(
        stage="confirmation",
        frequency_scales=(1.0, 2.0, 4.0, 8.0),
    )
    families = ["finite_difference_mixture", "continuous_legendre_tilt"]
    seeds = list(range(1001, 1031))
    initializations = 3
    payload = {
        "command": "confirm-e4",
        "config": asdict(config),
        "families": families,
        "seeds": seeds,
        "initializations": list(range(initializations)),
        "familywise_rule": "two-sided Bonferroni over 2Q characteristic coordinates and separately over four moment coordinates",
        "development_amendment": "distributional_sid/E4_FREQUENCY_BANK_AMENDMENT_20260714.md",
        "selected_frequency_bank": [1.0, 2.0, 4.0, 8.0],
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    detection_rows = []
    detection_path = run_dir / "seed_detection.csv"
    if detection_path.exists():
        detection_rows = pd.read_csv(detection_path).to_dict("records")
    completed = {(row["dgp_family"], int(row["dgp_seed"])) for row in detection_rows}
    for family in families:
        for seed in seeds:
            if (family, seed) in completed:
                print(f"E4 confirmation family={family} seed={seed} complete", flush=True)
                continue
            temporary_scores = []
            for method_init in range(initializations):
                print(f"E4 confirmation family={family} seed={seed} init={method_init}", flush=True)
                run_moment_blind_seed(
                    run_dir, family, seed, method_init, config, save_scores=True
                )
                temporary_scores.append(
                    run_dir / "scores" / f"e4_{family}__seed_{seed}__init_{method_init}.npz"
                )
            arrays = [np.load(path) for path in temporary_scores]
            orth = np.mean(np.stack([item["orth"] for item in arrays], axis=0), axis=0)
            moment = np.mean(np.stack([item["moment_orth"] for item in arrays], axis=0), axis=0)
            truth = arrays[0]["truth"]
            orth_estimate = np.mean(orth, axis=0)
            orth_se = np.std(orth, axis=0, ddof=1) / np.sqrt(orth.shape[0])
            moment_estimate = np.mean(moment, axis=0)
            moment_se = np.std(moment, axis=0, ddof=1) / np.sqrt(moment.shape[0])
            orth_threshold = norm.ppf(1.0 - 0.05 / (2.0 * orth.shape[1]))
            moment_threshold = norm.ppf(1.0 - 0.05 / (2.0 * moment.shape[1]))
            detection_rows.append(
                {
                    "dgp_family": family,
                    "dgp_seed": seed,
                    "characteristic_detected": bool(np.any(np.abs(orth_estimate / np.maximum(orth_se, 1e-12)) > orth_threshold)),
                    "moment_false_call": bool(np.any(np.abs(moment_estimate / np.maximum(moment_se, 1e-12)) > moment_threshold)),
                    "characteristic_max_abs_z": float(np.max(np.abs(orth_estimate / np.maximum(orth_se, 1e-12)))),
                    "moment_max_abs_z": float(np.max(np.abs(moment_estimate / np.maximum(moment_se, 1e-12)))),
                }
            )
            atomic_npz(
                run_dir / "scores_seed_average" / f"e4_{family}__seed_{seed}.npz",
                orth=orth.astype(np.float32),
                moment_orth=moment.astype(np.float32),
                truth=truth.astype(np.float64),
                dgp_seed=np.array([seed], dtype=np.int64),
                initialization_count=np.array([initializations], dtype=np.int16),
            )
            for item in arrays:
                item.close()
            for path in temporary_scores:
                path.unlink()
            atomic_csv(detection_path, pd.DataFrame(detection_rows).sort_values(["dgp_family", "dgp_seed"]))
    validation = package_e4_confirmation(run_dir, expected_seeds=len(seeds), initializations=initializations)
    print(pd.read_csv(run_dir / "detection_summary.csv").to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_tune_e4(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = ["finite_difference_mixture", "continuous_legendre_tilt"]
    banks = {
        "default_0p25_2": (0.25, 0.5, 1.0, 2.0),
        "wide_0p5_4": (0.5, 1.0, 2.0, 4.0),
        "high_1_8": (1.0, 2.0, 4.0, 8.0),
    }
    payload = {
        "command": "tune-e4",
        "stage": "development_frequency_audit",
        "families": families,
        "banks": banks,
        "seeds": [3, 4],
        "n_train": [8000, 32000],
        "amplitude_fraction": [0.5, 0.8],
        "initializations": [0],
        "selection_policy": "frequency bank may be selected on medium-amplitude n=8000 development cells; amplitude and sample size are reported as scientific axes, not tuning choices",
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    rows = []
    for family in families:
        for bank_name, scales in banks.items():
            for amplitude in (0.5, 0.8):
                for n_train in (8000, 32000):
                    combo = f"{family}__{bank_name}__a{amplitude:g}__n{n_train}"
                    combo_dir = run_dir / "grid" / combo
                    config = MomentBlindRunConfig(
                        n_train=n_train,
                        query_count=128,
                        random_width=32,
                        feature_ridge=1e-3,
                        riesz_ridge=1e-2,
                        qmc_power=17,
                        amplitude_fraction=amplitude,
                        frequency_scales=scales,
                        stage="development_frequency_audit",
                    )
                    metric_frames = []
                    nuisance_frames = []
                    for seed in (3, 4):
                        print(f"E4 audit {combo} seed={seed}", flush=True)
                        metric, nuisance = run_moment_blind_seed(
                            combo_dir, family, seed, 0, config, save_scores=False
                        )
                        metric_frames.append(metric)
                        nuisance_frames.append(nuisance)
                    metrics = pd.concat(metric_frames, ignore_index=True)
                    nuisances = pd.concat(nuisance_frames, ignore_index=True)
                    orth = metrics[(metrics.method == "ORTH") & metrics.metric_name.isin(["nrmse", "calibration_slope"])]
                    method_means = (
                        metrics[
                            metrics.metric_name.eq("nrmse")
                            & metrics.method.isin(["PLUG", "RIESZ", "ORTH", "OR_M", "OR_R"])
                        ]
                        .groupby("method")
                        .metric_value.mean()
                    )
                    means = orth.groupby("metric_name").metric_value.mean()
                    moment_fpr = metrics[
                        (metrics.method == "MOMENT_ORTH_1_TO_4")
                        & (metrics.metric_name == "any_false_call_bonferroni")
                    ].metric_value.mean()
                    rows.append(
                        {
                            "family": family,
                            "bank": bank_name,
                            "frequency_scales": ",".join(map(str, scales)),
                            "amplitude_fraction": amplitude,
                            "n_train": n_train,
                            "orth_nrmse": float(means["nrmse"]),
                            "orth_calibration_slope": float(means["calibration_slope"]),
                            "plugin_nrmse": float(method_means["PLUG"]),
                            "riesz_nrmse": float(method_means["RIESZ"]),
                            "oracle_m_nrmse": float(method_means["OR_M"]),
                            "oracle_riesz_weight_nrmse": float(method_means["OR_R"]),
                            "moment_familywise_false_call": float(moment_fpr),
                            "feature_derivative_nrmse": float(nuisances.feature_derivative_nrmse.mean()),
                            "alpha_nrmse": float(nuisances.alpha_nrmse.mean()),
                            "probe_q95_abs": float(nuisances.probe_q95_abs.mean()),
                        }
                    )
    frame = pd.DataFrame(rows)
    eligible = frame[(frame.amplitude_fraction == 0.5) & (frame.n_train == 8000)]
    bank_scores = eligible.groupby("bank").agg(
        mean_orth_nrmse=("orth_nrmse", "mean"),
        worst_orth_nrmse=("orth_nrmse", "max"),
        mean_calibration_error=("orth_calibration_slope", lambda value: float(np.mean(np.abs(value - 1.0)))),
    )
    bank_scores["selection_score"] = bank_scores.mean_orth_nrmse + 0.25 * bank_scores.mean_calibration_error
    selected_bank = str(bank_scores.selection_score.idxmin())
    frame["selected_frequency_bank"] = frame.bank.eq(selected_bank)
    atomic_csv(run_dir / "frequency_audit.csv", frame)
    atomic_csv(run_dir / "bank_selection.csv", bank_scores.reset_index())
    atomic_json(run_dir / "selected_bank.json", {"selected_bank": selected_bank, "scales": list(banks[selected_bank])})
    print(frame.to_string(index=False), flush=True)
    print(bank_scores.to_string(), flush=True)
    return 0


def command_confirm_e3(args: argparse.Namespace) -> int:
    _force_thread_limits()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        {"cell_id": "d4_n8000", "history_dim": 4, "n_train": 8000, "width": 32},
        {"cell_id": "d16_n8000", "history_dim": 16, "n_train": 8000, "width": 64},
        {"cell_id": "d32_n32000", "history_dim": 32, "n_train": 32000, "width": 64},
    ]
    seeds = list(range(1001, 1031))
    initializations = 3
    payload = {
        "command": "confirm-e3",
        "stage": "confirmation_partial_iid",
        "cells": cells,
        "seeds": seeds,
        "initializations": list(range(initializations)),
        "fixed": {"response_dim": 4, "query_count": 64, "folds": 5, "feature_ridge": 1e-3, "riesz_ridge": 1e-2},
        "omission": "dependent trajectory cells and full observation-score matrices are not included in this local pass",
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    for cell in cells:
        cell_dir = run_dir / "cells" / cell["cell_id"]
        config = GaussianRunConfig(
            n_train=cell["n_train"],
            n_bank=1024,
            n_folds=5,
            history_dim=cell["history_dim"],
            response_dim=4,
            query_count=64,
            random_width=cell["width"],
            feature_ridge=1e-3,
            riesz_ridge=1e-2,
            qmc_power=15,
            stage="confirmation_partial_iid",
        )
        for seed in seeds:
            for method_init in range(initializations):
                print(f"E3 {cell['cell_id']} seed={seed} init={method_init}", flush=True)
                run_gaussian_seed(cell_dir, seed, method_init, config, save_scores=False)
    metric_frames = []
    nuisance_frames = []
    for cell in cells:
        cell_dir = run_dir / "cells" / cell["cell_id"]
        for path in sorted((cell_dir / "cases").glob("*.csv")):
            frame = pd.read_csv(path)
            frame["experiment_id"] = "E3"
            frame["cell_id"] = cell["cell_id"]
            metric_frames.append(frame)
        for path in sorted((cell_dir / "nuisance").glob("*.csv")):
            frame = pd.read_csv(path)
            frame["experiment_id"] = "E3"
            frame["cell_id"] = cell["cell_id"]
            nuisance_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    nuisances = pd.concat(nuisance_frames, ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", metrics)
    metrics.to_parquet(run_dir / "seed_level.parquet", index=False)
    atomic_csv(run_dir / "nuisance_metrics.csv", nuisances)
    nuisances.to_parquet(run_dir / "nuisance_metrics.parquet", index=False)
    group_columns = [
        "experiment_id", "cell_id", "dgp_seed", "method", "target_name", "metric_name"
    ]
    averaged = metrics.groupby(group_columns, as_index=False).agg(
        metric_value=("metric_value", "mean"),
        initialization_sd=("metric_value", "std"),
        initialization_count=("method_init", "nunique"),
    )
    atomic_csv(run_dir / "seed_level_init_averaged.csv", averaged)
    rng = np.random.default_rng(94411)
    summary_rows = []
    for keys, frame in averaged.groupby(["cell_id", "method", "target_name", "metric_name"], sort=True):
        values = frame.metric_value.to_numpy(float)
        draws = np.mean(values[rng.integers(0, values.size, size=(2000, values.size))], axis=1)
        summary_rows.append(
            {
                **dict(zip(["cell_id", "method", "target_name", "metric_name"], keys, strict=True)),
                "mean": float(np.mean(values)),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_dgp_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(run_dir / "summary.csv", summary)
    expected_cases = len(cells) * len(seeds) * initializations
    validation = {
        "expected_case_files": expected_cases,
        "case_files": len(metric_frames),
        "nuisance_files": len(nuisance_frames),
        "dgp_seeds_per_cell": int(averaged.groupby("cell_id").dgp_seed.nunique().min()),
        "all_initialization_counts_complete": bool(averaged.initialization_count.eq(initializations).all()),
        "failed_metric_rows": int(np.sum(metrics.fit_status != "ok")),
        "nonfinite_metric_values": int(np.sum(~np.isfinite(metrics.metric_value))),
        "observation_score_storage_complete": False,
        "passed": bool(
            len(metric_frames) == expected_cases
            and len(nuisance_frames) == expected_cases
            and averaged.groupby("cell_id").dgp_seed.nunique().min() == len(seeds)
            and averaged.initialization_count.eq(initializations).all()
            and np.all(metrics.fit_status == "ok")
            and np.all(np.isfinite(metrics.metric_value))
        ),
    }
    atomic_json(run_dir / "validation.json", validation)
    atomic_json(
        run_dir / "package_manifest.json",
        {
            "source_hash": __import__("distributional_sid.core", fromlist=["sha256_tree"]).sha256_tree(Path(__file__).resolve().parents[2]),
            "frozen_config_sha256": __import__("distributional_sid.core", fromlist=["sha256_file"]).sha256_file(run_dir / "frozen_config.json"),
            "validation_sha256": __import__("distributional_sid.core", fromlist=["sha256_file"]).sha256_file(run_dir / "validation.json"),
        },
    )
    print(summary[(summary.method == "ORTH") & summary.metric_name.isin(["nrmse", "calibration_slope", "coverage_fraction"])].to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def command_confirm_e5(args: argparse.Namespace) -> int:
    _force_thread_limits()
    from .core import sha256_file, sha256_tree

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]
    seeds = list(range(1001, 1031))
    initializations = 3
    payload = {
        "command": "confirm-e5",
        "stage": "confirmation",
        "sigmas": sigmas,
        "seeds": seeds,
        "initializations": list(range(initializations)),
        "fixed": {"n_train": 8000, "query_count": 64, "folds": 5, "width": 32, "feature_ridge": 1e-3, "riesz_ridge": 1e-2},
        "score_storage": "metrics, nuisance diagnostics, query banks, splits, and coefficients; full observation scores omitted for local disk safety",
    }
    write_frozen_config(run_dir, payload)
    freeze_source_and_environment(run_dir)
    for sigma in sigmas:
        for seed in seeds:
            for method_init in range(initializations):
                print(f"E5 sigma={sigma:g} seed={seed} init={method_init}", flush=True)
                run_support_motion_seed(run_dir, sigma, seed, method_init)
    case_files = sorted((run_dir / "cases").glob("*.csv"))
    nuisance_files = sorted((run_dir / "nuisance").glob("*.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
    nuisances = pd.concat([pd.read_csv(path) for path in nuisance_files], ignore_index=True)
    atomic_csv(run_dir / "seed_level.csv", metrics)
    metrics.to_parquet(run_dir / "seed_level.parquet", index=False)
    atomic_csv(run_dir / "nuisance_metrics.csv", nuisances)
    nuisances.to_parquet(run_dir / "nuisance_metrics.parquet", index=False)
    averaged = metrics.groupby(
        ["cell_id", "dgp_seed", "method", "target_name", "metric_name"], as_index=False
    ).agg(
        metric_value=("metric_value", "mean"),
        initialization_sd=("metric_value", "std"),
        initialization_count=("method_init", "nunique"),
    )
    atomic_csv(run_dir / "seed_level_init_averaged.csv", averaged)
    rng = np.random.default_rng(95519)
    summary_rows = []
    for keys, frame in averaged.groupby(["cell_id", "method", "metric_name"], sort=True):
        values = frame.metric_value.to_numpy(float)
        draws = np.mean(values[rng.integers(0, values.size, size=(2000, values.size))], axis=1)
        summary_rows.append(
            {
                **dict(zip(["cell_id", "method", "metric_name"], keys, strict=True)),
                "mean": float(np.mean(values)),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_dgp_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(run_dir / "summary.csv", summary)
    likelihood_summary = nuisances.groupby("sigma", as_index=False).agg(
        likelihood_score_available=("likelihood_score_available", "mean"),
        likelihood_score_rms=("likelihood_score_rms", "mean"),
        likelihood_score_q99=("likelihood_score_q99", "mean"),
        feature_derivative_nrmse=("feature_derivative_nrmse", "mean"),
        alpha_nrmse=("alpha_nrmse", "mean"),
        probe_q95_abs=("probe_q95_abs", "mean"),
    )
    atomic_csv(run_dir / "likelihood_score_diagnostic.csv", likelihood_summary)
    expected_cases = len(sigmas) * len(seeds) * initializations
    validation = {
        "expected_case_files": expected_cases,
        "case_files": len(case_files),
        "nuisance_files": len(nuisance_files),
        "all_initialization_counts_complete": bool(averaged.initialization_count.eq(initializations).all()),
        "failed_metric_rows": int(np.sum(metrics.fit_status != "ok")),
        "nonfinite_metric_values": int(np.sum(~np.isfinite(metrics.metric_value))),
        "sigma_zero_likelihood_explicitly_unavailable": bool(
            nuisances.loc[nuisances.sigma == 0, "likelihood_score_available"].eq(0).all()
        ),
        "observation_score_storage_complete": False,
        "passed": bool(
            len(case_files) == expected_cases
            and len(nuisance_files) == expected_cases
            and averaged.initialization_count.eq(initializations).all()
            and np.all(metrics.fit_status == "ok")
            and np.all(np.isfinite(metrics.metric_value))
            and nuisances.loc[nuisances.sigma == 0, "likelihood_score_available"].eq(0).all()
        ),
    }
    atomic_json(run_dir / "validation.json", validation)
    atomic_json(
        run_dir / "package_manifest.json",
        {
            "source_hash": sha256_tree(Path(__file__).resolve().parents[2]),
            "frozen_config_sha256": sha256_file(run_dir / "frozen_config.json"),
            "validation_sha256": sha256_file(run_dir / "validation.json"),
            "summary_sha256": sha256_file(run_dir / "summary.csv"),
            "likelihood_diagnostic_sha256": sha256_file(run_dir / "likelihood_score_diagnostic.csv"),
        },
    )
    print(summary[(summary.method == "ORTH") & summary.metric_name.isin(["nrmse", "calibration_slope", "coverage_fraction"])].to_string(index=False), flush=True)
    print(likelihood_summary.to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distributional-sid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    development = subparsers.add_parser("development")
    development.add_argument("--run-dir", required=True)
    development.add_argument("--seed-start", type=int, default=1)
    development.add_argument("--seed-end", type=int, default=2)
    development.add_argument("--initializations", type=int, default=3)
    development.add_argument("--n-train", type=int, default=8000)
    development.add_argument("--n-bank", type=int, default=1024)
    development.add_argument("--folds", type=int, default=5)
    development.add_argument("--history-dim", type=int, default=8)
    development.add_argument("--response-dim", type=int, default=4)
    development.add_argument("--queries", type=int, default=64)
    development.add_argument("--width", type=int, default=64)
    development.add_argument("--feature-ridge", type=float, default=1e-3)
    development.add_argument("--riesz-ridge", type=float, default=1e-3)
    development.add_argument("--qmc-power", type=int, default=15)
    development.add_argument("--e2-qmc-power", type=int, default=18)
    development.add_argument("--no-scores", action="store_true")
    development.set_defaults(func=command_development)

    e0 = subparsers.add_parser("e0")
    e0.add_argument("--run-dir", required=True)
    e0.set_defaults(func=command_e0)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--run-dir", required=True)
    tune.add_argument("--n-train", type=int, default=8000)
    tune.add_argument("--n-bank", type=int, default=1024)
    tune.add_argument("--folds", type=int, default=5)
    tune.add_argument("--history-dim", type=int, default=8)
    tune.add_argument("--response-dim", type=int, default=4)
    tune.add_argument("--queries", type=int, default=64)
    tune.add_argument("--qmc-power", type=int, default=15)
    tune.set_defaults(func=command_tune)

    confirm_e1 = subparsers.add_parser("confirm-e1")
    confirm_e1.add_argument("--run-dir", required=True)
    confirm_e1.set_defaults(func=command_confirm_e1)

    develop_e4 = subparsers.add_parser("develop-e4")
    develop_e4.add_argument("--run-dir", required=True)
    develop_e4.set_defaults(func=command_develop_e4)

    confirm_e4 = subparsers.add_parser("confirm-e4")
    confirm_e4.add_argument("--run-dir", required=True)
    confirm_e4.set_defaults(func=command_confirm_e4)

    tune_e4 = subparsers.add_parser("tune-e4")
    tune_e4.add_argument("--run-dir", required=True)
    tune_e4.set_defaults(func=command_tune_e4)

    confirm_e3 = subparsers.add_parser("confirm-e3")
    confirm_e3.add_argument("--run-dir", required=True)
    confirm_e3.set_defaults(func=command_confirm_e3)

    confirm_e5 = subparsers.add_parser("confirm-e5")
    confirm_e5.add_argument("--run-dir", required=True)
    confirm_e5.set_defaults(func=command_confirm_e5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
