r"""Assemble a Markdown report from a run directory (Section 27).

Reads ``fit_metadata.json``, ``diagnostics.json``, ``readouts.parquet`` and emits
``report.md`` with the required sections and caveats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CAVEATS = """\
## 14. Limitations and caveats

- These are **predictive distributional effects, not direct structural synapses**.
- Slow gain/tail/covariance effects are **candidate** neuromodulatory signatures, not
  molecular identification.
- Behavioral covariates and calcium kinetics can explain some slow effects.
- Bands cover the projection onto the chosen feature/statistic family; unmodeled
  channels may remain.
- Long-memory and informative sampling are known scope boundaries.
- Predictive directedness must not be read structurally without additional assumptions.
"""


def _read_json(p):
    return json.load(open(p)) if Path(p).exists() else {}


def build_report(run_dir: str | Path) -> str:
    run = Path(run_dir)
    meta = _read_json(run / "fit_metadata.json")
    diag = _read_json(run / "diagnostics.json")
    readouts_path = run / "readouts.parquet"
    lines = ["# sid_neuromod analysis report", ""]

    lines += ["## 1. Dataset summary", ""]
    for k in ("dataset_id", "n_timepoints", "n_neurons", "prediction_horizon_s",
              "target_mode", "model_class"):
        if k in meta:
            lines.append(f"- **{k}**: {meta[k]}")
    lines.append("")

    lines += ["## 2-3. Splits, model & feature configuration", ""]
    for k in ("timescales_s", "ridge_lambda", "hac_lags", "train_interval",
              "calibration_interval", "test_interval"):
        if k in meta:
            lines.append(f"- **{k}**: {meta[k]}")
    lines.append("")

    lines += ["## 4. Calibration diagnostics", ""]
    for k in ("pit_mean", "pit_variance", "pit_ks_stat", "pit_ks_pvalue",
              "invalid_variance_fraction"):
        if k in diag:
            lines.append(f"- **{k}**: {diag[k]:.4g}")
    lines.append("")

    lines += ["## 5. Held-out predictive performance", ""]
    for k in ("heldout_nll", "baseline_var_heldout_nll", "baseline_ridge_r2"):
        if k in diag:
            lines.append(f"- **{k}**: {diag[k]:.4g}")
    lines.append("")

    if readouts_path.exists():
        r = pd.read_parquet(readouts_path)
        lines += ["## 6-8. Functional connectome & timescale decomposition", ""]
        lines.append(f"- readout rows: {len(r)}")
        if "significant" in r:
            lines.append(f"- significant readouts: {int(r['significant'].sum())}")
        gain = r[r.channel == "gain_log_variance"] if "channel" in r else r.iloc[:0]
        if len(gain):
            top = gain.reindex(gain.estimate.abs().sort_values(ascending=False).index).head(5)
            lines += ["", "### 9. Top candidate slow gain effects", "",
                      "| target | source | timescale_s | estimate | z | sig |",
                      "|---|---|---:|---:|---:|:--:|"]
            for _, row in top.iterrows():
                lines.append(
                    f"| {row.get('target_id','')} | {row.get('source_id','')} | "
                    f"{row.get('timescale_s',''):g} | {row['estimate']:.4g} | "
                    f"{row['z_score']:.2f} | {'*' if row['significant'] else ''} |")
        lines.append("")

    nmi_path = run / "neuromod_candidates.parquet"
    if nmi_path.exists():
        nmi = pd.read_parquet(nmi_path)
        lines += ["## 9. Candidate neuromodulatory factors (by NMI)", "",
                  "| source | target | NMI | sig gain |", "|---|---|---:|---:|"]
        for _, row in nmi.head(8).iterrows():
            lines.append(f"| {row.source_id} | {row.target_id} | {row.nmi:.3f} | "
                         f"{int(row.n_significant_gain)} |")
        lines.append("")

    lines += ["## 15. Machine-readable artifact inventory", ""]
    for f in sorted(run.glob("*")):
        if f.is_file():
            lines.append(f"- `{f.name}`")
    lines.append("")
    lines.append(CAVEATS)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a Markdown report for a run dir")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    text = build_report(args.run_dir)
    out = Path(args.run_dir) / "report.md"
    out.write_text(text)
    print(f"wrote {out}")
    return text


if __name__ == "__main__":
    main()
