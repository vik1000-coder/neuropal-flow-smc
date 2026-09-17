"""Verify the clarity-first figure delivery; --seal records a reviewed delivery."""
from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, OUT, OLD, sha256, relative


def verify_hash(path, expected):
    assert path.is_file() and sha256(path) == expected, f"Changed hashed file: {path}"


def verify_ledger(path):
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        verify_hash((path.parent / name.lstrip("*")).resolve(), expected)
        count += 1
    return count


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)


def check_values():
    original = pd.read_csv(OLD / "data/f01_example_recording_display_z.csv")
    plotted = pd.read_csv(OUT / "data/data_recording.csv")
    close(plotted, original[plotted.columns])
    stats = pd.read_csv(OLD / "data/f09_awc_observed_onset_statistics.csv")
    for chemical in ("butanone", "pentanedione", "nacl"):
        saved = stats[stats.chemical.eq(chemical)].sort_values("horizon_seconds")
        new = pd.read_csv(OUT / f"data/observed_awc_{chemical}.csv")
        columns = ["horizon_seconds", "mean_onset_minus_quiet_z", "ci95_low", "ci95_high", "bh_q_value"]
        close(new[columns], saved[columns])
    original = pd.read_csv(OLD / "data/f08_distributional_screen_inventory.csv")
    for path in (OUT / "data").glob("modeled_onset_*.csv"):
        new = pd.read_csv(path)
        saved = original[original.candidate_id.eq(new.candidate_id.iloc[0])]
        columns = ["source_lag_frames", "horizon_frames", "screen_mean_normalized", "targeted_n128_mean_normalized"]
        close(new[columns], saved[columns])
        assert saved.context.iloc[0] == new.context.iloc[0]
    controls = pd.read_csv(OLD / "data/f08b_plotted_sampler_control_evidence.csv")
    for stem, metric in [("prediction_mean_sampler_excess", "endpoint_mean"),
                         ("prediction_w1_sampler_excess", "endpoint_wasserstein1")]:
        new = pd.read_csv(OUT / f"data/{stem}.csv")
        saved = controls[controls.source_neuron.eq("RIP") & controls.target_neuron.eq("URB") & controls.metric.eq(metric)]
        assert len(new) == len(saved) == 1
        close(new[["plotted_estimate", "plotted_ci_low", "plotted_ci_high"]],
              saved[["sampling_excess_mean", "sampling_excess_ci_2_5", "sampling_excess_ci_97_5"]])
        close(new.sampling_excess_joint_max_t_p, saved.sampling_excess_joint_max_t_p)
    peptide = pd.read_csv(OUT / "data/lag_neuropeptides.csv")
    close(peptide.auroc, [.581, .576, .567, .546])
    close(peptide.source_to_cut_seconds, [.25, 1, 2, 4])
    tyramine = pd.read_csv(OUT / "data/lag_tyramine.csv")
    assert tyramine.auroc.isna().all() and tyramine.n_common_supported_sources.eq(0).all()
    for path in (OUT / "data").glob("lag_*.csv"):
        data = pd.read_csv(path)
        if "source_to_cut_seconds" not in data or "horizon_frames" not in data:
            continue
        close(data.source_to_cut_seconds, data.lag_frames / 4)
        close(data.forecast_horizon_seconds, data.horizon_frames / 4)
        assert data.n_common_supported_sources.nunique() == 1
    return "Passed exact saved recording, onset, sampler-excess, lag timing and missingness checks; component manifests retain additional native-archive checks."


def main(seal=False):
    checks = {
        "unchanged_original_bundle_files": verify_ledger(OLD / "SHA256SUMS"),
        "unchanged_prototype_files": verify_ledger(ROOT / "results/figure_redesign_prototypes_20260831/SHA256SUMS"),
    }
    manifests = [json.loads(path.read_text()) for path in sorted(OUT.glob("*_manifest.json"))]
    assert len(manifests) == 3
    figures = []
    verified_hashes = 0
    for manifest in manifests:
        assert manifest["no_new_fits_model_samples_or_inference"] is True
        for name, expected in {**manifest["source_hashes"], **manifest["output_hashes"]}.items():
            verify_hash(ROOT / name, expected)
            verified_hashes += 1
        for key in ("common_renderer", "code"):
            if key in manifest:
                verify_hash(ROOT / manifest[key]["path"], manifest[key]["sha256"])
        figures += manifest["figures"]
    stems = [figure["stem"] for figure in figures]
    assert len(stems) == len(set(stems)) == 20
    assert {p.stem for p in (OUT / "figures").glob("*.png")} == set(stems)
    assert {p.stem for p in (OUT / "figures").glob("*.svg")} == set(stems)
    for figure in figures:
        assert figure["layout"]["all_text_inside_canvas"]
        assert len(figure["caption_lines"]) <= 2
        with Image.open(OUT / "figures" / f"{figure['stem']}.png") as picture:
            assert picture.size == (2640, 1540)
        with Image.open(OUT / "qa" / f"{figure['stem']}_1100px.png") as picture:
            assert picture.size[0] == 1100
        svg = ET.parse(OUT / "figures" / f"{figure['stem']}.svg")
        assert svg.findall(".//{http://www.w3.org/2000/svg}text")
    checks.update(new_figures=20, source_and_output_hashes=verified_hashes,
                  png_svg_qa_sizes_and_editable_text="passed", values=check_values())

    index = json.loads((OUT / "INDEX.json").read_text())
    assert len(index) == 22
    assert {r["stem"] for r in index if r["origin"] == "new figure"} == set(stems)
    docs = list(OUT.rglob("*.md")) + [ROOT / "FIGURE_REDESIGN_PLAN.md", ROOT / "FIGURE_STORY.md"]
    links = 0
    for doc in docs:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme:
                continue
            path = (doc.parent / unquote(parsed.path)).resolve()
            if seal and path in (OUT / "SHA256SUMS", OUT / "VALIDATION.json"):
                continue
            assert path.exists(), f"Broken document link: {doc} -> {target}"
            links += 1
    checks["local_document_links"] = links
    for script in Path(__file__).parent.glob("*.py"):
        compile(script.read_text(), str(script), "exec")
    checks["source_scripts_compile"] = "passed"
    checks["review_completed_for_all_groups"] = all(m["status"].startswith("complete") for m in manifests)
    if seal:
        assert checks["review_completed_for_all_groups"], "Review figures before sealing."
        (OUT / "VALIDATION.json").write_text(json.dumps({
            "status": "passed", "checks": checks,
            "scope": "Saved-value, provenance and display validation; no new statistical inference or empirical model validation.",
        }, indent=2) + "\n")
        files = sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
                       + list(Path(__file__).parent.glob("*.py")))
        (OUT / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {os.path.relpath(path, OUT)}\n" for path in files))
    if (OUT / "SHA256SUMS").exists():
        checks["new_bundle_ledger_files"] = verify_ledger(OUT / "SHA256SUMS")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    main(parser.parse_args().seal)
