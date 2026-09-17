"""Build a plain Markdown reading order for the standalone figure exports."""
from pathlib import Path
import json

from common import OUT

# The order is authored for explanation, not computed by ranking result magnitude.
MAIN = [
    ("data_recording", "The recordings", "Three observed channels and the actual stimulus schedule."),
    ("sampling_diagram", "The sampling method", "Shared history, two repaired branches, one target contrast."),
    ("reference_comparison", "Cook and Randi", "A single AUROC comparison with the saved uncertainty intervals."),
    ("lag_neuropeptides", "Neuropeptide lag alignment", "Reference alignment is distinct from identifying a delay."),
    ("prediction_mean_sampler_excess", "Mean effect and sampling controls", "One RIP→URB mean prediction compared with its matched sampling controls."),
    ("prediction_mean_lag_profile", "Mean effects across source lags", "An effect can be supported without resolving a preferred lag."),
    ("prediction_mean_versus_spread_illustration", "Mean versus spread", "An explicitly illustrative explanation, not fitted distributions."),
    ("prediction_w1_sampler_excess", "Distributional effect and sampling controls", "A W1 contrast may include a mean shift; it is not evidence of gain by itself."),
    ("prediction_spread_consistency", "A selected spread prediction", "ASH→OLQ in the screen and particle rerun, with saved pointwise intervals."),
    ("observed_awc_butanone", "Observed stimulus response", "AWC activity after butanone onset, compared with quiet time."),
]
COMPANIONS = [
    ("observed_awc_pentanedione", "Observed AWC: pentanedione", "Same scale and averaging windows as the butanone view."),
    ("observed_awc_nacl", "Observed AWC: NaCl", "All windows retained, including non-passing tests."),
    ("modeled_onset_ase_ask", "Modeled onset contrast: ASE→ASK", "Butanone-event W1 contrast; selected and exploratory."),
    ("modeled_onset_afd_ury", "Modeled onset contrast: AFD→URY", "Pentanedione-event log-SD contrast; selected and exploratory."),
    ("modeled_onset_awa_rib", "Modeled onset contrast: AWA→RIB", "Pooled-event SD contrast; selected and exploratory."),
    ("modeled_onset_aim_aib", "Modeled onset contrast: AIM→AIB", "Pooled-event log-SD contrast; selected and exploratory."),
    ("lag_dopamine", "Dopamine lag alignment", "Two supported source classes; no corrected alignment evidence."),
    ("lag_serotonin", "Serotonin lag alignment", "One supported source class; no corrected alignment evidence."),
    ("lag_octopamine", "Octopamine lag alignment", "One supported source class; no corrected alignment evidence."),
    ("lag_tyramine", "Tyramine support limitation", "Not evaluable is not a zero effect or an absent reference pathway."),
    ("lag_monoamines", "All-monoamine lag alignment", "The aggregate reference on its fixed supported-source set."),
    ("lag_union", "Peptide/monoamine union lag alignment", "Overlaps the peptide result; not independent replication."),
]


def entry(section, index, row):
    stem, title, point = row
    prefix = "../figure_redesign_prototypes_20260831"
    if stem in ("sampling_diagram", "reference_comparison"):
        png, svg = f"{prefix}/{stem}.png", f"{prefix}/{stem}.svg"
        caption = f"{prefix}/{'SAMPLING' if stem == 'sampling_diagram' else 'REFERENCE'}_CAPTION.md"
        data = f"{prefix}/{'sampling_illustrations' if stem == 'sampling_diagram' else 'reference_values'}.csv"
        origin = "preserved prototype"
    else:
        png, svg = f"figures/{stem}.png", f"figures/{stem}.svg"
        caption, data = f"captions/{stem}.md", f"data/{stem}.csv"
        origin = "new figure"
    return dict(section=section, number=index, stem=stem, title=title, point=point,
                png=png, svg=svg, caption=caption, data=data, origin=origin)


def main():
    entries = [entry("Main reading sequence", i, row) for i, row in enumerate(MAIN, 1)]
    entries += [entry("Companion figures", i, row) for i, row in enumerate(COMPANIONS, len(MAIN) + 1)]
    for row in entries:
        for key in ("png", "svg", "caption", "data"):
            assert (OUT / row[key]).is_file(), f"Missing indexed {key}: {row[key]}"
    readme = [
        "# NeuroPAL figures: clarity-first set", "",
        "**20 new figures**, plus the two preserved prototypes: 22 views in one indexed collection. "
        "Prepared 31 August 2026 from saved results; no training, model sampling, or statistical tests were rerun.", "",
        "[Open the full visual gallery](GALLERY.md). Each view has one main question, a short caption, "
        "an editable SVG, and plotted-data CSV. Detailed definitions stay in linked captions.", "",
        "## Main reading sequence", "",
        "These ten views cover the data, method, external references, lags, and prediction evidence. "
        "The remaining views retain all named pathways and selected onset follow-ups.", "",
        "| View | Topic | Files |", "| --- | --- | --- |",
    ]
    gallery = ["# NeuroPAL figure gallery", "", "[Figure index and reproduction](README.md)", "",
               "One figure at a time. Main reading sequence first; companion cases follow. "
               "The original sampling and Cook/Randi prototypes are included by reference, unchanged.", ""]
    for section in ("Main reading sequence", "Companion figures"):
        if section == "Companion figures":
            readme += ["", "## Companion figures", "", "| View | Topic | Files |", "| --- | --- | --- |"]
        gallery += [f"## {section}", ""]
        for row in [r for r in entries if r["section"] == section]:
            links = f"[PNG]({row['png']}) · [SVG]({row['svg']}) · [Caption]({row['caption']}) · [Data]({row['data']})"
            readme.append(f"| {row['number']} | {row['title']} | {links} |")
            gallery += [f"### {row['number']}. {row['title']}", "", f"![{row['title']}]({row['png']})", "",
                        row["point"], "", links, ""]
    readme += ["", "## What the evidence labels mean", "",
        "Saved corrected tests accompany claims of statistical support. Pointwise intervals are labeled as such; "
        "they do not by themselves establish differences between lags or runs. Missing matched controls are stated. "
        "Model-based effects are not physical interventions, reference compatibility is not active signaling, and "
        "distributional distance is not automatically a variance or gain effect.", "",
        "[All seven pathway decisions](captions/lag_evidence_summary.md) · "
        "[Original complete evidence record](../figure_atlas_20260831/README.md) · "
        "[Redesign plan](../../FIGURE_REDESIGN_PLAN.md)", "",
        "## Validation and reproduction", "",
        "All new images were inspected at normal viewing size (1100 pixels wide), with a separate-agent review. "
        "Source hashes, plotted values, timing, labels, and editable SVG text were checked. This verifies the "
        "figure delivery, not independent biological validity or reader comprehension.", "",
        "[Validation record](VALIDATION.json) · [Checksums](SHA256SUMS) · "
        "[Data/onset provenance](data_onset_manifest.json) · [Lag provenance](lag_manifest.json) · "
        "[Prediction provenance](prediction_manifest.json)", "",
        "From `/Users/vik/Developer/new_sbtg_neuro`:", "", "```sh",
        ".venv/bin/python analysis/figure_story_20260831/data_onset_figures.py",
        ".venv/bin/python analysis/figure_story_20260831/lag_figures.py",
        ".venv/bin/python analysis/figure_story_20260831/prediction_figures.py",
        ".venv/bin/python analysis/figure_story_20260831/index_figures.py",
        ".venv/bin/python analysis/figure_story_20260831/verify.py", "```", "",
        "Rendering resets review metadata. After intended changes, inspect the new exports and record the review "
        "before using `verify.py --seal`. A changed delivery must not silently replace the old checksums. "
        "The original 13-figure technical bundle and the two earlier prototypes remain intact.", "",
    ]
    (OUT / "README.md").write_text("\n".join(readme))
    (OUT / "GALLERY.md").write_text("\n".join(gallery))
    (OUT / "INDEX.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Indexed {len(entries)} views: 20 new, 2 preserved prototypes.")


if __name__ == "__main__":
    main()
