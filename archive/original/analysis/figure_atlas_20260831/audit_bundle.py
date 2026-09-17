"""Validate the static figure delivery without fitting, sampling, or testing effects.

Default: read-only checks, including a delivered ledger when present.
--seal: explicitly record this audit and reseal only the figure-delivery files,
after inspecting intended changes. Frozen source ledgers must already pass.
"""
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

from common import ROOT, OUT, sha256

EXPECTED_STEMS = [
    "f01_data_and_stimuli", "f02_sampling_and_effect_definitions",
    "f03_cook_randi_correspondence", "f04_bentley_lag_alignment",
    "f05_weight_and_cohort_sensitivity", "f06_existing_family_and_candidate_evidence",
    "f07_top_mean_candidate_lag_profiles", "f08a_distributional_screen_to_n128",
    "f08b_matched_sampling_and_quiet_controls", "f09_observed_onset_vs_quiet",
    "f10_class_estimates_and_support", "f11_predictive_distribution_audit",
    "f12_baseline_mean_evidence_map",
]
ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
LIVING_GUIDES = [ROOT / "DASHBOARD.md", ROOT / "FIGURES.md", ATLAS / "DASHBOARD_GUIDE.md"]
INDEX_DOCS = [ROOT / "README.md", ROOT / "docs/current/README.md", ROOT / "results/README.md",
              ROOT / "EXPERIMENT_INDEX.md", ATLAS / "README.md"]


def verify_hash(path: Path, expected: str):
    assert path.is_file(), f"Missing hashed file: {path}"
    actual = sha256(path)
    assert actual == expected, f"Hash mismatch: {path}; expected {expected}, got {actual}"


def verify_ledger(path: Path):
    entries = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        verify_hash((path.parent / name.lstrip("*")).resolve(), expected)
        entries += 1
    return entries


def heading_ids(path):
    ids = set()
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", path.read_text(), flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        ids.add(slug if count == 0 else f"{slug}-{count}")
        counts[slug] = count + 1
    return ids


def verify_links(documents, prospective_ledger=False):
    count = 0
    for doc in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("//"):
                continue
            path = (doc.parent / unquote(parsed.path)).resolve() if parsed.path else doc
            if prospective_ledger and path in {OUT / "SHA256SUMS", OUT / "VALIDATION.json"}:
                count += 1
                continue
            assert path.exists(), f"Broken local link: {doc} → {target}"
            if parsed.fragment and path.suffix == ".md":
                assert unquote(parsed.fragment) in heading_ids(path), f"Missing heading: {doc} → {target}"
            count += 1
    return count


def component_hashes():
    checked = {}
    for path in sorted(OUT.glob("*manifest.json")):
        meta = json.loads(path.read_text())
        n = 0
        for key in ["source_hashes", "build_provenance", "outputs", "output_sha256", "code_sha256"]:
            hashes = meta.get(key, {})
            if not isinstance(hashes, dict):
                continue
            for name, expected in hashes.items():
                if isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected):
                    verify_hash(ROOT / name, expected)
                    n += 1
        for entry in meta.get("input_inventory", []):
            verify_hash(ROOT / entry["path"], entry["sha256"])
            n += 1
        checked[path.name] = n
    assert len(checked) == 5, checked
    return checked


def numeric_checks():
    expected_rows = {
        "f01_stimulus_schedules.csv": 51,
        "f01_example_recording_display_z.csv": 960,
        "f03_fixed_lag_reference_rows.csv": 16,
        "f03_reported_source_bootstrap_rows.csv": 4,
        "f04_common_source_w1_profiles_audit_rounded.csv": 16,
        "f04_progressive_network_lagmax_rows.csv": 14,
        "f05_historical80_saved_bootstrap_rows.csv": 4,
        "f06_full_qualifying_mean_edges.csv": 102,
        "f06_full_qualifying_mean_cells.csv": 492,
        "f06_displayed_mean_candidates.csv": 6,
        "f07_plotted_lag_profiles.csv": 16,
        "f08_distributional_screen_inventory.csv": 6,
        "f08_primary_screen_inventory.csv": 6,
        "f08_all_sampler_control_evidence.csv": 24,
        "f08b_plotted_sampler_control_evidence.csv": 6,
        "f09_awc_observed_onset_statistics.csv": 18,
        "f09_awc_observed_worm_responses.csv": 306,
        "f10_class_descriptive_blocks.csv": 18,
        "f11_five_primary_saved_comparisons.csv": 5,
        "f11_descriptive_tail_calibration.csv": 3,
        "f12_baseline_mean_evidence_map.csv": 2916,
    }
    for name, count in expected_rows.items():
        frame = pd.read_csv(OUT / "data" / name)
        assert len(frame) == count, (name, len(frame), count)
        numeric = frame.select_dtypes(include="number").to_numpy()
        assert not np.isinf(numeric).any(), f"Infinite value in {name}"
    schedules = pd.read_csv(OUT / "data/f01_stimulus_schedules.csv")
    assert schedules.groupby("worm_id").size().eq(3).all()
    assert schedules.groupby("worm_id").chemical.nunique().eq(3).all()
    controls = pd.read_csv(OUT / "data/f08_all_sampler_control_evidence.csv")
    assert controls.evidence_label.eq("exceeds_sampling_controls_only").sum() == 4
    positive_quiet = (controls.support_pass & controls.selection_eligible &
                      controls.temporal_specificity_excess_mean.gt(0) &
                      controls.temporal_specificity_ci_2_5.gt(0) &
                      controls.temporal_specificity_joint_max_t_p.le(.05))
    assert not positive_quiet.any()
    rip = controls[(controls.source_neuron == "RIP") & (controls.target_neuron == "URB") &
                   (controls.metric == "endpoint_wasserstein1")]
    assert len(rip) == 1 and rip.iloc[0].support_pass and rip.iloc[0].selection_eligible
    assert rip.iloc[0].sampling_excess_mean > 0
    assert rip.iloc[0].sampling_excess_joint_max_t_p < .05
    assert rip.iloc[0].temporal_specificity_joint_max_t_p >= .05
    classes = pd.read_csv(OUT / "data/f10_class_descriptive_blocks.csv")
    counts = classes.groupby("support_policy").n_pairs.sum().to_dict()
    assert counts == {"all_estimated": 2862, "strong_common_lags": 1484}, counts
    return {"row_counts": expected_rows, "finite_numeric_values_or_explicit_missing": True,
            "support_sensitive_class_denominators": counts,
            "positive_sampler_only_outcomes": 4, "rip_urb_w1_sampling_control_pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    inputs = component_hashes()
    ledgers = [ATLAS / "checksums.sha256", ATLAS / "canonical/checksums.sha256",
               ATLAS / "gui/checksums.sha256", ATLAS / "postfreeze_external/checksums.sha256",
               ATLAS / "complete_family_evidence_20260830/checksums.sha256",
               ATLAS / "sampling_null_controls_combined8_n128_20260830/analysis/checksums.sha256",
               ROOT / "results/neuron_class_effects_20260831/SHA256SUMS",
               ROOT / "results/distribution_structure_20260831/SHA256SUMS"]
    frozen = {str(p.relative_to(ROOT)): verify_ledger(p) for p in ledgers}
    sizes = {}
    for stem in EXPECTED_STEMS:
        with Image.open(OUT / "figures" / f"{stem}.png") as im:
            sizes[stem] = list(im.size)
            assert im.width >= 1800 and im.height >= 900, (stem, im.size)
            im.verify()
        svg = ET.parse(OUT / "figures" / f"{stem}.svg")
        assert len(svg.findall(".//{http://www.w3.org/2000/svg}text")) > 0, stem
    assert len(list((OUT / "figures").glob("*.png"))) == 13
    assert len(list((OUT / "figures").glob("*.svg"))) == 13
    numeric = numeric_checks()
    documents = sorted(OUT.glob("*.md")) + LIVING_GUIDES + INDEX_DOCS
    links = verify_links(documents, prospective_ledger=args.seal)
    nav = json.loads((OUT / "navigation_hash_update.json").read_text())
    for row in nav["updates"]:
        verify_hash(ROOT / row["path"], row["after"])
    report = {
        "status": "pass_with_scientific_caveats", "new_inferential_tests": False,
        "new_models_or_samples": False, "png_svg_pairs": 13,
        "component_manifest_hash_entries": inputs, "existing_ledger_entries": frozen,
        "numeric_checks": numeric, "local_markdown_links_checked": links,
        "image_dimensions": sizes, "editable_svg_xml": "pass",
        "visual_qa": "Final PNGs inspected by figure owners; primary agent reviewed all figures. See VALIDATION.md.",
        "repository_tests": {"collected": 307, "passed": 305, "skipped": 2, "warnings": 1,
                             "seconds": 29.17, "details": "TEST_RESULTS.md"},
        "original_scientific_outputs_changed": False,
        "navigation_ledger_updates": "navigation_hash_update.json",
    }
    if args.seal:
        (OUT / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
        paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        paths += sorted(Path(__file__).parent.glob("*.py")) + LIVING_GUIDES
        lines = [f"{sha256(p)}  {Path(os.path.relpath(p, OUT))}" for p in paths]
        (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    delivery = verify_ledger(OUT / "SHA256SUMS") if (OUT / "SHA256SUMS").exists() else None
    print(json.dumps({**report, "delivery_checksum_entries": delivery}, indent=2))


if __name__ == "__main__":
    main()
