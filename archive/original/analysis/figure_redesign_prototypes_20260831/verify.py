"""Check two figure prototypes without altering frozen scientific results.

Default: read-only verification. --seal records checks and hashes of this bundle.
Visual review must be completed separately after every rendering change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figure_redesign_prototypes_20260831"
CODE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(path, expected):
    assert path.is_file() and digest(path) == expected, f"Hash mismatch: {path}"


def ledger_check(path):
    lines = path.read_text().splitlines()
    for line in lines:
        expected, name = line.split(maxsplit=1)
        checked_hash((path.parent / name.lstrip("*")).resolve(), expected)
    return len(lines)


def main(seal=False):
    sampling = json.loads((OUT / "sampling_manifest.json").read_text())
    reference = json.loads((OUT / "reference_manifest.json").read_text())
    checks = {}
    hashes = {**sampling["source_hashes"], **sampling["output_hashes"],
              **reference["output_sha256"],
              reference["code"]["path"]: reference["code"]["sha256"]}
    for entry in reference["input_inventory"]:
        hashes[entry["path"]] = entry["sha256"]
        hashes[entry["verified_against"]] = entry["ledger_sha256"]
    for name, expected in hashes.items():
        checked_hash(ROOT / name, expected)
    checks["source_and_output_hashes"] = len(hashes)
    checks["unchanged_original_delivery_files"] = ledger_check(
        ROOT / "results/figure_atlas_20260831/SHA256SUMS")

    def rows(path):
        with path.open(newline="") as handle:
            return {row["reference"]: row for row in csv.DictReader(handle)}

    old = rows(ROOT / "results/figure_atlas_20260831/data/f03_reported_source_bootstrap_rows.csv")
    new = rows(OUT / "reference_values.csv")
    assert len(new) == 4 and set(old) == set(new)
    for name, row in new.items():
        assert row["auroc_difference"] == old[name]["progressive_minus_published"]
        for bound in ("reported_ci_low", "reported_ci_high"):
            assert row[bound] == old[name][bound]
        difference = float(row["progressive_auroc"]) - float(row["sbtg_published_auroc"])
        assert abs(difference - float(row["auroc_difference"])) < 1e-14
    checks["exact_saved_differences"] = 4
    checks["exact_saved_interval_endpoints"] = 8
    assert sampling["semantic_checks"]["history_frames"] == 80
    assert sampling["semantic_checks"]["fps"] == 4
    assert sampling["semantic_checks"]["source_window_frames"] == 4
    assert sampling["all_curves_are_schematic"]

    for stem, size in [("sampling_diagram", (2750, 1540)),
                       ("reference_comparison", (2640, 1540))]:
        with Image.open(OUT / f"{stem}.png") as picture:
            assert picture.size == size
        svg = ET.parse(OUT / f"{stem}.svg")
        assert svg.findall(".//{http://www.w3.org/2000/svg}text")
    checks["png_dimensions_and_editable_svg_text"] = "passed"
    for script in CODE.glob("*.py"):
        compile(script.read_text(), str(script), "exec")
    checks["scripts_compile"] = "passed"

    documents = list(OUT.glob("*.md")) + [ROOT / "FIGURE_REDESIGN_PLAN.md"]
    link_count = 0
    for doc in documents:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            parsed = urlsplit(link.strip("<>"))
            if parsed.scheme:
                continue
            path = (doc.parent / unquote(parsed.path)).resolve()
            if seal and path in (OUT / "SHA256SUMS", OUT / "VALIDATION.json"):
                continue
            assert path.exists(), f"Missing local link: {doc} -> {link}"
            link_count += 1
    checks["local_document_links"] = link_count
    checks["visual_review_recorded"] = (
        sampling["status"].startswith("complete") and reference["status"].startswith("complete"))
    if seal:
        assert checks["visual_review_recorded"], "Record visual review before sealing."
        (OUT / "VALIDATION.json").write_text(json.dumps({
            "status": "passed", "checks": checks,
            "scope": "Display and saved-value checks only; no new inference or empirical validation.",
        }, indent=2) + "\n")
        files = sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
                       + list(CODE.glob("*.py")))
        import os
        (OUT / "SHA256SUMS").write_text("".join(
            f"{digest(path)}  {os.path.relpath(path, OUT)}\n" for path in files))
    if (OUT / "SHA256SUMS").exists():
        checks["prototype_ledger_files"] = ledger_check(OUT / "SHA256SUMS")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    main(parser.parse_args().seal)
