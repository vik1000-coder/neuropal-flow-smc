"""Scientific and API regression tests for the new presentation layer."""
import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from dashboard_v2.server import AtlasData, CANONICAL, DIRECT, PROGRESSIVE, make_handler, safe_json


@pytest.fixture(scope="module")
def data():
    return AtlasData()


@pytest.mark.parametrize("source,target,channel,context", [
    ("FLP", "ADE", "endpoint_mean", "baseline"),
    ("RIP", "URB", "endpoint_wasserstein1", "baseline"),
    ("ADL", "SMB", "endpoint_log_sd", "onset_minus_baseline"),
    ("AWC", "AVA", "cumulative_mean", "butanone_onset"),
])
def test_all_pair_values_and_intervals_use_exact_canonical_orientation(data, source, target, channel, context):
    result = data.edge(dict(source=source, target=target, channel=channel, context=context))
    s, t = data.neurons.index(source), data.neurons.index(target)
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as z:
        for method in (PROGRESSIVE, DIRECT):
            for short, prefix in (("mean", "mean_normalized"), ("low", "ci_low_normalized"), ("high", "ci_high_normalized")):
                expected = z[f"{prefix}__{method}__{channel}__{context}"][:, :, t, s]
                np.testing.assert_array_equal(result["series"][method][short], expected)
    np.testing.assert_allclose(result["worms"].mean(axis=1), result["series"][PROGRESSIVE]["mean"], atol=3e-8, rtol=1e-6)


def test_direct_does_not_borrow_progressive_individuals_or_tests(data):
    result = data.edge({"method": DIRECT})
    assert result["worms"] is None
    assert not result["cell_evidence"] and not result["calibration"]
    assert result["edge_evidence"] is None
    assert result["worm_values_unavailable_reason"]


def test_uncertainty_never_becomes_a_significance_test(data):
    w1 = data.edge({"source": "RIP", "target": "URB", "channel": "endpoint_wasserstein1"})
    assert w1["signed"] is False
    assert not w1["cell_evidence"]
    onset = data.edge({"context": "onset_minus_baseline"})
    assert not onset["cell_evidence"] and onset["edge_evidence"] is None
    diagonal = data.edge({"source": "ADE", "target": "ADE"})
    assert diagonal["self_pair"] and not diagonal["cell_evidence"]


def test_control_excess_is_paired_at_worm_level_and_matches_every_saved_row(data):
    for row in data.calibration.to_dict("records"):
        result = data.controls({"id": row["candidate_id"], "metric": row["metric"]})
        observed = np.asarray(result["observed"], dtype=float)
        np.testing.assert_allclose(result["sampling_excess"], np.abs(observed) - result["sampling"], atol=3e-8)
        np.testing.assert_allclose(result["quiet_excess"], np.abs(observed) - result["quiet"], atol=3e-8)
        assert np.mean(result["sampling_excess"]) == pytest.approx(row["sampling_excess_mean"], abs=3e-8)
        assert np.mean(result["quiet_excess"]) == pytest.approx(row["temporal_specificity_excess_mean"], abs=3e-8)


def test_prediction_draws_use_saved_endpoint_and_exact_inverse_scale(data):
    result = data.prediction({"source": "FLP", "target": "ADE", "lag": "1", "horizon": "8", "worm": "0", "chemical": "butanone"})
    assert result["available"] and len(result["low"]) == len(result["high"]) == 256
    assert result["readout_frame"] == result["cut_frame"] + 8
    target = data.neurons.index("ADE")
    assert result["actual"] == data.cohort().traces[0][result["readout_frame"], target]
    from dashboard_v2.server import ROOT
    with np.load(ROOT / result["archive"], allow_pickle=False) as z:
        li = z["worm_indices"].tolist().index(0)
        event = z["chemical_name_by_worm_event"][li].astype(str).tolist().index("butanone")
        ti = z["selected_target_indices"].tolist().index(target)
        si = z["selected_source_indices"].tolist().index(data.neurons.index("FLP"))
        raw = z["observed_endpoint_samples"][li, 0, event, 0, :, 0, :, ti, si]
        scaler = data.scaler(int(z["fold"]))
        expected = raw * scaler.scale[target] + scaler.mean[target]
    np.testing.assert_array_equal(result["low"], expected[0])
    np.testing.assert_array_equal(result["high"], expected[1])


@pytest.mark.parametrize("params", [{"lag":"4"}, {"context":"onset"}, {"method":DIRECT}, {"target":"ADF"}])
def test_unarchived_predictions_are_explicitly_unavailable(data, params):
    assert data.prediction(params)["available"] is False


def test_observed_band_is_actual_between_worm_quantiles(data):
    result = data.signals({"source":"FLP", "target":"ADE", "chemical":"nacl"})
    values = np.stack([r["values"] for r in result["rows"]])
    np.testing.assert_allclose(result["median"], np.nanmedian(values, axis=0))
    np.testing.assert_allclose(result["lower"], np.nanquantile(values, .1, axis=0))
    np.testing.assert_allclose(result["upper"], np.nanquantile(values, .9, axis=0))
    assert len(result["rows"]) == 17
    assert all(r["duration"] == 10 for r in result["rows"])
    assert "not a confidence interval" in result["note"]
    assert np.isnan(data.cohort().traces[0][:4]).all()
    assert safe_json(np.array([np.nan, np.inf, -np.inf, .2])) == [None, None, None, .2]


def test_sources_and_exact_support_gates_are_preserved(data):
    result = data.edge({})
    support = result["support"][0]
    assert support["source_lag_frames"] == 1
    assert support["support_qualified"] and support["genealogy_strong_gate_pass"]
    assert len(data.verified) == 6 and sum(x["entries"] for x in data.verified) >= 114


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
        for route, expected in [("/api/edge?source=unknown",400),("/api/edge?source=ADE&source=FLP",400),("/api/targets?lag=3",400),("/server.py",404),("/../README.md",404)]:
            with pytest.raises(HTTPError) as exc:
                urlopen(root+route)
            assert exc.value.code == expected
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(root+"/api/meta",headers={"Host":"untrusted.example"}))
        assert exc.value.code == 403
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
