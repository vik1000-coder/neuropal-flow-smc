"""Scientific and API regression tests for the historical-80 dashboard."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from dashboard_v2_80.server import (
    AtlasData,
    CANONICAL,
    PROGRESSIVE,
    ROOT,
    make_handler,
    safe_json,
)


@pytest.fixture(scope="module")
def data():
    return AtlasData()


@pytest.mark.parametrize(
    "source,target,channel,context",
    [
        ("FLP", "ADE", "endpoint_mean", "state_average"),
        ("RIP", "URB", "endpoint_wasserstein1", "baseline"),
        ("ADL", "SMB", "endpoint_log_sd", "onset_minus_baseline"),
        ("PHA", "AVA", "cumulative_mean", "butanone_onset"),
    ],
)
def test_edge_arrays_use_exact_target_row_source_column_orientation(
    data, source, target, channel, context
):
    result = data.edge(
        dict(source=source, target=target, channel=channel, context=context)
    )
    source_index, target_index = data.neurons.index(source), data.neurons.index(target)
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
        for short, prefix in (
            ("mean", "mean_normalized"),
            ("low", "ci_low_normalized"),
            ("high", "ci_high_normalized"),
        ):
            expected = archive[
                f"{prefix}__{PROGRESSIVE}__{channel}__{context}"
            ][:, :, target_index, source_index]
            np.testing.assert_array_equal(result["series"][PROGRESSIVE][short], expected)
    np.testing.assert_allclose(
        result["worms"].mean(axis=1),
        result["series"][PROGRESSIVE]["mean"],
        atol=3e-8,
        rtol=1e-6,
    )


def test_primary_scope_is_one_seed_eighty_classes_and_twenty_traces(data):
    meta = data.meta()
    assert len(meta["neurons"]) == 80
    assert len(meta["worm_ids"]) == 20
    assert meta["methods"] == [{"id": PROGRESSIVE, "label": "Best progressive SMC"}]
    assert meta["sampling"] == {
        "particles": 64,
        "repair_branch_factor": 4,
        "future_branch_factor": 2,
        "integration_steps": 20,
    }
    assert meta["seed_choice"]["selected_generator_seed"] == 1701
    assert "two-seed" in meta["seed_choice"]["two_seed_note"]


def test_support_diagnostics_match_saved_arrays_and_are_not_inference(data):
    result = data.edge(
        {"source": "FLP", "target": "ADE", "context": "state_average"}
    )
    source_index = data.neurons.index("FLP")
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
        expected_valid = archive[
            f"valid_fraction__{PROGRESSIVE}__state_average"
        ][:, source_index]
        expected_genealogy = archive[
            f"genealogy_valid_fraction_0_10__{PROGRESSIVE}__state_average"
        ][:, source_index]
    np.testing.assert_array_equal(
        result["series"][PROGRESSIVE]["valid_fraction"], expected_valid
    )
    np.testing.assert_array_equal(
        result["series"][PROGRESSIVE]["genealogy_fraction"], expected_genealogy
    )
    assert len(result["support"]) == 4
    assert not result["cell_evidence"] and result["edge_evidence"] is None
    assert data.controls({})["available"] is False
    assert data.prediction({})["available"] is False


def test_all_target_slice_preserves_orientation(data):
    result = data.targets(
        {
            "source": "FLP",
            "target": "ADE",
            "channel": "endpoint_mean",
            "context": "state_average",
            "lag": "1",
            "horizon": "1",
        }
    )
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
        expected = archive[
            f"mean_normalized__{PROGRESSIVE}__endpoint_mean__state_average"
        ][0, 0, :, data.neurons.index("FLP")]
    assert len(result["neurons"]) == 80
    np.testing.assert_array_equal(result["mean"], expected)


def test_observed_band_is_actual_between_trace_quantiles(data):
    result = data.signals({"source": "FLP", "target": "ADE", "chemical": "nacl"})
    values = np.stack([row["values"] for row in result["rows"]])
    np.testing.assert_allclose(result["median"], np.nanmedian(values, axis=0))
    np.testing.assert_allclose(result["lower"], np.nanquantile(values, 0.1, axis=0))
    np.testing.assert_allclose(result["upper"], np.nanquantile(values, 0.9, axis=0))
    assert len(result["rows"]) == 20
    assert "pseudo-paired" in result["note"] and "donor-imputed" in result["note"]
    assert safe_json(np.array([np.nan, np.inf, -np.inf, 0.2])) == [
        None,
        None,
        None,
        0.2,
    ]


def test_reference_view_contains_single_seed_published_and_two_seed(data):
    result = data.references()
    rows = result["comparisons"]
    references = {
        "randi_wild_type",
        "cook_struct_80",
        "cook_chem_80",
        "cook_gap_80",
    }
    names = {row["reference_or_network"] for row in rows}
    assert references.issubset(names)
    for reference in references:
        subset = [row for row in rows if row["reference_or_network"] == reference]
        assert {row["method"] for row in subset} == {
            PROGRESSIVE,
            "sbtg_published",
            "two_seed_sensitivity",
        }
        single = next(
            row
            for row in subset
            if row["method"] == PROGRESSIVE
            and row["timing_label"] == "0.25 s lag · 0.25 s forecast"
        )
        published = next(row for row in subset if row["method"] == "sbtg_published")
        two_seed = next(row for row in subset if row["method"] == "two_seed_sensitivity")
        assert single["auroc"] > published["auroc"]
        assert two_seed["auroc"] > single["auroc"]
        interval = result["primary_two_seed_delta_intervals"][reference]
        assert interval["difference"] == pytest.approx(
            two_seed["auroc"] - published["auroc"], abs=2e-6
        )
        single_interval = result["primary_single_seed_delta_intervals"][reference]
        assert single_interval["difference"] == pytest.approx(
            single["auroc"] - published["auroc"], abs=2e-6
        )

    bentley_networks = {
        "monoamine_all",
        "monoamine_dopamine",
        "monoamine_serotonin",
        "monoamine_tyramine",
        "monoamine_octopamine",
        "neuropeptide_all",
        "neuromodulator_union",
    }
    assert bentley_networks.issubset(names)
    assert bentley_networks == set(result["bentley_lagmax"])
    for network in bentley_networks:
        subset = [row for row in rows if row["reference_or_network"] == network]
        assert {row["method"] for row in subset} == {
            PROGRESSIVE,
            "sbtg_published",
        }
        assert {row["channel"] for row in subset if row["method"] == PROGRESSIVE} == {
            "endpoint_mean",
            "endpoint_wasserstein1",
        }
        lagmax = result["bentley_lagmax"][network]
        assert {row["method"] for row in lagmax} == {
            PROGRESSIVE,
            "sbtg_published",
        }
    tyramine = result["bentley_lagmax"]["monoamine_tyramine"]
    progressive_tyramine = next(
        row
        for row in tyramine
        if row["method"] == PROGRESSIVE and row["channel"] == "endpoint_mean"
    )
    assert not progressive_tyramine["evaluable"]


def test_latex_report_and_pair_level_bentley_table_are_complete():
    report = ROOT / "results/sbtg80_optimized_full_atlas_20260902/latex_report"
    assert (report / "main.tex").is_file()
    assert (report / "historical_80_neuron_atlas_report.pdf").read_bytes().startswith(
        b"%PDF"
    )
    edges = pd.read_csv(report / "bentley_positive_neuron_relationships.csv")
    expected = {
        "monoamine_all": 112,
        "monoamine_dopamine": 46,
        "monoamine_serotonin": 22,
        "monoamine_tyramine": 33,
        "monoamine_octopamine": 11,
        "neuropeptide_all": 755,
        "neuromodulator_union": 841,
    }
    assert edges.groupby("network").size().to_dict() == expected
    assert len(edges) == 1820
    assert edges.select_dtypes("number").notna().all().all()
    assert set(edges.source_recording_origin) <= {"head", "tail"}
    assert set(edges.target_recording_origin) <= {"head", "tail"}


def test_lag1_archives_match_frozen_optimization_samples_exactly(data):
    current_root = ROOT / "results/sbtg80_optimized_full_atlas_20260902/responses/progressive_bridge_smc"
    frozen_root = ROOT / "results/sbtg80_flow_optimization_20260901/sampling/flow_lr6e4__N64__b4__f2/responses/progressive_bridge_smc"
    for fold in range(5):
        name = f"flow_lr6e4__progressive_bridge_smc__ell1__N64__f{fold}__s1701.npz"
        with np.load(current_root / name, allow_pickle=False) as current, np.load(
            frozen_root / name, allow_pickle=False
        ) as prior:
            assert int(current["progressive_branch_factor"]) == 4
            assert int(current["progressive_future_branch_factor"]) == 2
            for channel in (
                "endpoint_mean",
                "cumulative_mean",
                "peak_mean",
                "event_probability",
                "endpoint_sd",
                "endpoint_log_sd",
                "endpoint_wasserstein1",
            ):
                np.testing.assert_array_equal(
                    current[f"response_{channel}"][..., 0, :],
                    prior[f"response_{channel}"][..., 0, :],
                )


def test_readonly_api_validation_and_json_safety(data):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(root + "/api/edge?source=FLP&target=ADE") as response:
            body = response.read()
            assert response.headers["Cache-Control"] == "no-cache"
            assert json.loads(body)["selection"]["source"] == "FLP"
            assert b"NaN" not in body and b"Infinity" not in body
        for route, expected in (
            ("/api/edge?source=unknown", 400),
            ("/api/edge?source=ADE&source=FLP", 400),
            ("/api/targets?lag=3", 400),
            ("/server.py", 404),
            ("/../README.md", 404),
        ):
            with pytest.raises(HTTPError) as exc:
                urlopen(root + route)
            assert exc.value.code == expected
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(root + "/api/meta", headers={"Host": "untrusted.example"}))
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
