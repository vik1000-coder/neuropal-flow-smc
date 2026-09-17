from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def run(ceiling_dir: Path) -> None:
    metrics = pd.read_csv(ceiling_dir / "per_fold_metrics.csv")
    selection = json.loads((ceiling_dir / "winner_selection.json").read_text())
    lag = int(selection["lag"])
    candidate = str(selection["model"])
    rows: list[dict] = []
    for split, folds in (("screen", (0, 1, 2)), ("confirmation", (3, 4)), ("all", range(5))):
        subset = metrics[(metrics.lag == lag) & metrics.fold.isin(folds)]
        base = subset[subset.model == "self_ridge"].set_index("fold")
        model = subset[subset.model == candidate].set_index("fold")
        common = sorted(set(base.index) & set(model.index))
        for fold in common:
            rows.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "lag_frames": lag,
                    "candidate": candidate,
                    "energy_improvement_vs_self": float(
                        base.loc[fold, "energy"] - model.loc[fold, "energy"]
                    ),
                    "balanced_energy_improvement_vs_self": float(
                        base.loc[fold, "energy__stim_balanced"]
                        - model.loc[fold, "energy__stim_balanced"]
                    ),
                    "rmse_improvement_vs_self": float(
                        base.loc[fold, "rmse"] - model.loc[fold, "rmse"]
                    ),
                    "crps_improvement_vs_self": float(
                        base.loc[fold, "crps"] - model.loc[fold, "crps"]
                    ),
                    "target_spearman_gain_vs_self": float(
                        model.loc[fold, "target_spearman"]
                        - base.loc[fold, "target_spearman"]
                    ),
                    "candidate_innovation_spearman": float(
                        model.loc[fold, "innovation_spearman"]
                    ),
                }
            )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(ceiling_dir / "paired_self_baseline_contrasts.csv", index=False)
    confirm = contrasts[contrasts.split == "confirmation"]
    energy = float(confirm.energy_improvement_vs_self.mean())
    balanced = float(confirm.balanced_energy_improvement_vs_self.mean())
    rmse = float(confirm.rmse_improvement_vs_self.mean())
    innovation = float(confirm.candidate_innovation_spearman.mean())
    gate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_history_frames": lag,
        "selected_history_seconds": lag / 4.0,
        "best_offdiagonal_candidate": candidate,
        "conditional_distribution_winner": "self_ridge",
        "confirmation_folds": [3, 4],
        "mean_energy_improvement_vs_self": energy,
        "mean_balanced_energy_improvement_vs_self": balanced,
        "mean_rmse_improvement_vs_self": rmse,
        "mean_candidate_innovation_spearman": innovation,
        "both_confirmation_folds_improve_energy": bool(
            (confirm.energy_improvement_vs_self > 0).all()
        ),
        "both_confirmation_folds_improve_rmse": bool(
            (confirm.rmse_improvement_vs_self > 0).all()
        ),
        "promotion_rule": (
            "positive confirmation energy and RMSE improvement over self-history, "
            "with positive off-diagonal future-innovation correlation"
        ),
        "passes_promotion_gate": bool(energy > 0 and rmse > 0 and innovation > 0),
        "interpretation": (
            "The off-diagonal candidate contains weak innovation-ranking signal but "
            "does not improve held-out probabilistic prediction over self history."
        ),
        "external_references_consulted": False,
    }
    (ceiling_dir / "promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    validation = json.loads((ceiling_dir / "validation.json").read_text())
    validation["promotion_gate_recorded"] = True
    validation["passes_offdiagonal_promotion_gate"] = gate["passes_promotion_gate"]
    (ceiling_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    paths = sorted(
        path
        for path in ceiling_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (ceiling_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceiling-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.ceiling_dir.resolve())


if __name__ == "__main__":
    main()
