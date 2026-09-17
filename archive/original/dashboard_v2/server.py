"""Small local API over frozen atlas files. No training, sampling, or writes.

Run from the repository root: .venv/bin/python -m dashboard_v2.server
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from types import SimpleNamespace
import threading
from urllib.parse import parse_qs, urlsplit
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
PORTABLE = WEB / "data"
IS_PORTABLE = (PORTABLE / "bundle_manifest.json").exists()
ATLAS = ROOT / "results/neural_prediction_atlas_20260829"
CANONICAL = PORTABLE if IS_PORTABLE else ATLAS / "canonical"
EVIDENCE = PORTABLE if IS_PORTABLE else ATLAS / "complete_family_evidence_20260830"
NULLS = PORTABLE if IS_PORTABLE else ATLAS / "sampling_null_controls_combined8_n128_20260830/analysis"
SAMPLES = PORTABLE if IS_PORTABLE else NULLS.parent / "raw"
PROGRESSIVE = "progressive_bridge_smc"
DIRECT = "direct_importance"
CHANNELS = {
    "endpoint_mean": ("Mean activity", "Difference in mean target activity at the forecast endpoint."),
    "cumulative_mean": ("Average activity over time", "Difference in time-averaged target activity; not a sum or integral."),
    "peak_mean": ("Peak activity", "Difference in the average pathwise maximum over the forecast window."),
    "event_probability": ("Event probability", "Difference in crossing probability, normalized by achieved source displacement."),
    "endpoint_sd": ("Activity spread (SD)", "Difference in endpoint standard deviation; not variance or gain modulation."),
    "endpoint_log_sd": ("Log activity spread", "Difference in log endpoint SD, normalized before pooling; not a percentage change."),
    "endpoint_wasserstein1": ("Distribution distance (W1)", "Distance between target distributions. Positive values alone do not establish an effect beyond sampling noise."),
}
CONTEXTS = {
    "baseline": "Before stimulus", "onset": "At stimulus onset",
    "active": "During stimulus", "offset": "At stimulus offset", "recovery": "After stimulus",
    "state_average": "Average across all five phases", "onset_minus_baseline": "Onset minus baseline",
    "butanone_onset": "Butanone onset", "pentanedione_onset": "Pentanedione onset", "nacl_onset": "NaCl onset",
    "butanone_onset_minus_baseline": "Butanone onset minus baseline",
    "pentanedione_onset_minus_baseline": "Pentanedione onset minus baseline",
    "nacl_onset_minus_baseline": "NaCl onset minus baseline",
}
CI_NOTE = ("Shading is the saved pointwise 95% percentile interval for the mean across 17 worms "
           "(256 bootstrap resamples; model seeds averaged within worm). Conditional on fitted models; "
           "not a prediction interval, simultaneous band, or test corrected for dashboard exploration.")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_ledger(directory):
    ledger = directory / "checksums.sha256"
    checked = 0
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        path = (directory / filename.lstrip("*")).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Input ledger points outside this repository")
        if sha256(path) != digest:
            raise ValueError(f"Input checksum mismatch: {path.relative_to(ROOT)}")
        checked += 1
    return {"ledger": str(ledger.relative_to(ROOT)), "entries": checked, "sha256": sha256(ledger)}


@lru_cache(maxsize=96)
def atlas_array(key):
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as data:
        value = data[key]
    value.flags.writeable = False
    return value


@lru_cache(maxsize=6)
def worm_array(channel, context):
    with np.load(CANONICAL / "worm_matrices.npz", allow_pickle=False) as data:
        value = data[f"normalized__{channel}__{context}"]
    value.flags.writeable = False
    return value


class AtlasData:
    def __init__(self, verify=True):
        if verify and IS_PORTABLE:
            manifest = json.loads((PORTABLE / "bundle_manifest.json").read_text())
            for entry in manifest["files"]:
                path = (WEB / entry["path"]).resolve()
                if not path.is_relative_to(WEB) or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Portable input checksum mismatch: {entry['path']}")
            self.verified = [{"ledger": "data/bundle_manifest.json", "entries": len(manifest["files"]), "sha256": sha256(PORTABLE / "bundle_manifest.json")}]
        else:
            self.verified = [verify_ledger(p) for p in (CANONICAL, EVIDENCE, NULLS, SAMPLES, ATLAS / "gui")] if verify else []
        if verify and not IS_PORTABLE:
            manifest = json.loads((WEB / "input_manifest.json").read_text())
            for entry in manifest["inputs"]:
                path = (ROOT / entry["path"]).resolve()
                if not path.is_relative_to(ROOT) or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Observed-data or implementation input changed: {entry['path']}")
            self.verified.append({"ledger": "dashboard_v2/input_manifest.json", "entries": len(manifest["inputs"]), "sha256": sha256(WEB / "input_manifest.json")})
        with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as z:
            self.neurons = z["neurons"].astype(str).tolist()
            self.worms = z["worm_ids"].astype(str).tolist()
            self.lags = z["source_lag_frames"].tolist()
            self.horizons = z["horizon_frames"].tolist()
            assert str(z["orientation"]) == "target_row_source_column"
        self.cell_evidence = pd.read_csv(EVIDENCE / "cell_evidence.csv") if IS_PORTABLE else pd.read_parquet(EVIDENCE / "cell_evidence.parquet")
        self.edge_evidence = pd.read_csv(EVIDENCE / "edge_evidence.csv")
        self.calibration = pd.read_csv(NULLS / "sham_calibration.csv")
        with np.load(NULLS / "null_inference_arrays.npz", allow_pickle=False) as z:
            self.control_arrays = {k: z[k] for k in ("candidate_metric_id", "worm_ids", "observed_worm_value", "sampling_null_worm_magnitude", "quiet_pseudo_worm_magnitude")}
            assert z["worm_ids"].astype(str).tolist() == self.worms
        self.support = pd.read_csv(CANONICAL / "support_cells.csv") if IS_PORTABLE else pd.read_parquet(CANONICAL / "support_cells.parquet")
        self.cohort_lock = threading.Lock()
        self._cohort = None
        self._references = None

    def meta(self):
        shortcuts = self.calibration[self.calibration.metric.eq("endpoint_mean")]
        return {
            "title": "Neural atlas · effects & uncertainty", "version": "2.0", "snapshot": "2026-08-31",
            "neurons": self.neurons, "worm_ids": self.worms, "fps": 4,
            "lags": self.lags, "horizons": self.horizons,
            "channels": [{"id": k, "label": v[0], "description": v[1]} for k, v in CHANNELS.items()],
            "contexts": [{"id": k, "label": v} for k, v in CONTEXTS.items()],
            "methods": [{"id": PROGRESSIVE, "label": "Progressive SMC"}, {"id": DIRECT, "label": "Direct importance"}],
            "shortcuts": shortcuts[["candidate_id", "source_neuron", "target_neuron", "source_lag_frames", "horizon_frames", "context", "support_pass"]].to_dict("records"),
            "ci_note": CI_NOTE, "verified_inputs": self.verified,
            "portable": IS_PORTABLE, "original_explorer_available": not IS_PORTABLE,
            "scope": "17 worms · 54 pooled head-neuron classes · binary stimulus conditioning",
            "discovery_counts": {"baseline_mean_edges": 102, "resolved_lag_edges": 0, "both_control_gates": 0},
        }

    def selection(self, params):
        s, t = params.get("source", "FLP"), params.get("target", "ADE")
        channel, context = params.get("channel", "endpoint_mean"), params.get("context", "baseline")
        method = params.get("method", PROGRESSIVE)
        if s not in self.neurons or t not in self.neurons or channel not in CHANNELS or context not in CONTEXTS or method not in (PROGRESSIVE, DIRECT):
            raise ValueError("Unknown source, target, outcome, context or method")
        return self.neurons.index(s), self.neurons.index(t), channel, context, method

    def edge(self, params):
        s, t, channel, context, method = self.selection(params)
        values = {}
        for m in (PROGRESSIVE, DIRECT):
            suffix = f"{m}__{channel}__{context}"
            values[m] = {short: atlas_array(f"{prefix}__{suffix}")[:, :, t, s]
                         for short, prefix in (("mean", "mean_normalized"), ("low", "ci_low_normalized"), ("high", "ci_high_normalized"))}
            values[m]["valid_fraction"] = atlas_array(f"valid_fraction__{m}__{context}")[:, s]
            if m == PROGRESSIVE:
                values[m]["genealogy_fraction"] = atlas_array(f"genealogy_valid_fraction_0_10__{m}__{context}")[:, s]
        # Exact context binding is essential: onset-minus-baseline has no active-minus-baseline evidence.
        e = self.cell_evidence
        e = e[e.source_index.eq(s) & e.target_index.eq(t) & e.channel.eq(channel) & e.context.eq(context)]
        edge = self.edge_evidence
        edge = edge[edge.source_index.eq(s) & edge.target_index.eq(t) & edge.channel.eq(channel) & edge.context.eq(context)]
        keep = ["source_lag_frames", "horizon_frames", "support_eligible", "joint_primary_cell_max_t_p_value", "joint_primary_lag_contrast_max_t_p_value", "evidence_label"]
        cal = self.calibration
        cal = cal[cal.source_index.eq(s) & cal.target_index.eq(t) & cal.metric.eq(channel) & cal.context.eq(context)]
        support = self.support
        support = support[support.source_index.eq(s) & support.method.eq(method) & support.context.eq(context)]
        return {
            "selection": {"source": self.neurons[s], "target": self.neurons[t], "channel": channel, "context": context, "method": method},
            "series": values, "worms": worm_array(channel, context)[:, :, :, t, s] if method == PROGRESSIVE else None,
            "worm_values_method": PROGRESSIVE,
            "worm_values_unavailable_reason": None if method == PROGRESSIVE else "Individual-worm arrays were archived for progressive SMC only. The direct estimate and interval are available.",
            "cell_evidence": e[keep].to_dict("records") if method == PROGRESSIVE else [],
            "edge_evidence": edge.iloc[0].to_dict() if len(edge) and method == PROGRESSIVE else None,
            "calibration": cal.to_dict("records") if method == PROGRESSIVE else [],
            "support": support[["source_lag_frames", "support_qualified", "genealogy_strong_gate_pass", "valid_fraction", "genealogy_valid_fraction_0_10"]].to_dict("records"),
            "self_pair": s == t, "ci_note": CI_NOTE,
            "signed": channel != "endpoint_wasserstein1" or context.endswith("minus_baseline"),
        }

    def targets(self, params):
        s, _, channel, context, method = self.selection(params)
        lag, horizon = int(params.get("lag", 1)), int(params.get("horizon", 8))
        if lag not in self.lags or horizon not in self.horizons:
            raise ValueError("Unsupported timing")
        li, hi = self.lags.index(lag), self.horizons.index(horizon)
        suffix = f"{method}__{channel}__{context}"
        values = {k: atlas_array(f"{p}__{suffix}")[li, hi, :, s] for k, p in (("mean", "mean_normalized"), ("low", "ci_low_normalized"), ("high", "ci_high_normalized"))}
        return {"neurons": self.neurons, **values, "note": "Alphabetical targets, not a significance ranking. Intervals are pointwise."}

    def controls(self, params):
        cid, metric = params.get("id"), params.get("metric", "endpoint_mean")
        matches = self.calibration[self.calibration.candidate_id.eq(cid) & self.calibration.metric.eq(metric)]
        if len(matches) != 1:
            raise ValueError("No exact archived calibration for this coordinate")
        ids = self.control_arrays["candidate_metric_id"].astype(str).tolist()
        ix = ids.index(f"{cid}:{metric}")
        observed = self.control_arrays["observed_worm_value"][ix]
        sampling = self.control_arrays["sampling_null_worm_magnitude"][ix]
        quiet = self.control_arrays["quiet_pseudo_worm_magnitude"][ix]
        return {"summary": matches.iloc[0].to_dict(), "worms": self.worms,
                "observed": observed, "sampling": sampling, "quiet": quiet,
                "sampling_excess": np.abs(observed) - sampling,
                "quiet_excess": np.abs(observed) - quiet,
                "note": "Separate selected N128 rerun. Excess = absolute effect within each worm minus that worm's control magnitude; not the absolute pooled effect."}

    def cohort(self):
        with self.cohort_lock:
            if self._cohort is None:
                if IS_PORTABLE:
                    with np.load(PORTABLE / "observed_cohort.npz", allow_pickle=False) as z:
                        lengths = z["lengths"].astype(int)
                        traces = tuple(z["traces"][i, :length].copy() for i, length in enumerate(lengths))
                        intervals, chemicals = z["event_intervals_seconds"], z["chemical_name_by_worm_event"].astype(str)
                        schedules = tuple(SimpleNamespace(event_intervals_seconds=tuple(map(tuple, intervals[i].tolist())), chemical_name_by_event=tuple(chemicals[i].tolist())) for i in range(len(lengths)))
                        cohort = SimpleNamespace(traces=traces, neurons=z["neurons"].astype(str).tolist(), worm_ids=z["worm_ids"].astype(str).tolist(), fps=float(z["fps"]), stimulus_schedules=schedules)
                else:
                    from conditional_neural_benchmark.data import load_cohort
                    cohort = load_cohort(cohort_mode="oh16230_head")
                if list(cohort.neurons) != self.neurons or list(cohort.worm_ids) != self.worms:
                    raise ValueError("Observed cohort does not match frozen atlas coordinates")
                self._cohort = cohort
        return self._cohort

    @lru_cache(maxsize=32)
    def signal_data(self, source, target, chemical):
        cohort = self.cohort()
        if chemical not in ("butanone", "pentanedione", "nacl"):
            raise ValueError("Select one actual chemical event")
        indices = [self.neurons.index(source), self.neurons.index(target)]
        times = np.arange(-80, 121) / cohort.fps
        rows = []
        for wi, (raw, schedule) in enumerate(zip(cohort.traces, cohort.stimulus_schedules)):
            event = schedule.chemical_name_by_event.index(chemical)
            onset, offset = schedule.event_intervals_seconds[event]
            start = int(round(onset * cohort.fps))
            trace = np.asarray(raw[:, indices], dtype=float)
            # Display-only standardization per neuron/worm; preserve all missing observations.
            standardized = (trace - np.nanmean(trace, axis=0)) / np.maximum(np.nanstd(trace, axis=0), 1e-8)
            frames = start + np.arange(-80, 121)
            window = np.full((len(times), 2), np.nan)
            valid = (frames >= 0) & (frames < len(trace))
            window[valid] = standardized[frames[valid]]
            rows.append({"worm_id": cohort.worm_ids[wi], "values": window.T, "duration": offset - onset})
        arr = np.stack([r["values"] for r in rows])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            quantiles = np.nanquantile(arr, [.1, .5, .9], axis=0)
        return {"source": source, "target": target, "chemical": chemical, "time": times,
                "rows": rows, "lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2],
                "n_observed": np.isfinite(arr).sum(axis=0), "n_worms": len(rows),
                "units": "Within-worm, within-neuron display z-score",
                "note": "Preprocessed calcium signals; one actual event per worm. Band = 10th–90th percentiles across worms, not a confidence interval or measurement-error model. Missing observations remain gaps. No between-chemical significance claim."}

    def signals(self, params):
        self.selection(params)
        return self.signal_data(params.get("source", "FLP"), params.get("target", "ADE"), params.get("chemical", "butanone"))

    @lru_cache(maxsize=5)
    def scaler(self, fold):
        if IS_PORTABLE:
            with np.load(PORTABLE / "fold_scalers.npz", allow_pickle=False) as z:
                return SimpleNamespace(mean=z["mean"][fold].copy(), scale=z["scale"][fold].copy())
        from conditional_neural_benchmark.data import FoldScaler
        cohort = self.cohort()
        frame = pd.read_csv(ROOT / "results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv")
        mapping = dict(zip(frame.worm_id.astype(str), frame.outer_fold.astype(int)))
        folds = np.asarray([mapping[worm] for worm in cohort.worm_ids], dtype=np.int64)
        training = np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5))
        return FoldScaler.fit(cohort.traces[i] for i in training)

    @lru_cache(maxsize=8)
    def sample_file(self, path):
        with np.load(path, allow_pickle=False) as z:
            keys = ["worm_indices", "worm_ids", "phase_names", "selected_source_indices", "selected_target_indices",
                    "chemical_name_by_worm_event", "observed_cut_times", "observed_source_window_bounds",
                    "observed_endpoint_samples", "fold", "n_particles", "observed_diagnostic_valid",
                    "observed_diagnostic_achieved_gap", "observed_diagnostic_distinct_ancestors_low",
                    "observed_diagnostic_distinct_ancestors_high", "endpoint_sample_axes"]
            values = {k: z[k] for k in keys}
            if str(z["matrix_orientation"]) != "target_row_source_column":
                raise ValueError("Saved sample orientation differs")
        expected = ["heldout_worm", "phase", "event", "sampler_replicate", "arm", "horizon", "future_particle", "selected_target", "source"]
        if values["endpoint_sample_axes"].astype(str).tolist() != expected:
            raise ValueError("Saved sample axes differ")
        return values

    def prediction(self, params):
        s, t, _, context, method = self.selection(params)
        lag, horizon = int(params.get("lag", 1)), int(params.get("horizon", 8))
        wi, seed, repeat = int(params.get("worm", 0)), int(params.get("seed", 1701)), int(params.get("repeat", 0))
        chemical = params.get("chemical", "butanone")
        if lag not in self.lags or horizon not in self.horizons or not 0 <= wi < len(self.worms) or seed not in (1701, 2903) or repeat not in (0, 1) or chemical not in ("butanone", "pentanedione", "nacl"):
            raise ValueError("Unknown saved sample selection")
        if method != PROGRESSIVE or context != "baseline":
            return {"available": False, "reason": "Endpoint sample banks cover progressive SMC baseline calibrations only."}
        exact = self.calibration
        exact = exact[exact.source_index.eq(s) & exact.target_index.eq(t) & exact.source_lag_frames.eq(lag) & exact.horizon_frames.eq(horizon) & exact.context.eq(context)]
        if exact.empty:
            return {"available": False, "reason": "This pair and timing were not included in the eight-coordinate sampling calibration."}
        pattern = f"*__ell{lag}__h{horizon}__N128__f*__s{seed}__*.npz"
        for path in sorted((SAMPLES / "responses").glob(pattern)):
            z = self.sample_file(path)
            worms = z["worm_indices"].tolist()
            sources, targets = z["selected_source_indices"].tolist(), z["selected_target_indices"].tolist()
            if wi not in worms or s not in sources or t not in targets:
                continue
            local, si, ti = worms.index(wi), sources.index(s), targets.index(t)
            phase = z["phase_names"].astype(str).tolist().index("baseline")
            event = z["chemical_name_by_worm_event"][local].astype(str).tolist().index(chemical)
            cut = int(z["observed_cut_times"][local, phase, event])
            scaler = self.scaler(int(z["fold"]))
            samples = z["observed_endpoint_samples"][local, phase, event, repeat, :, 0, :, ti, si]
            samples = samples * scaler.scale[t] + scaler.mean[t]
            diagnostic_index = (local, phase, event, repeat, si)
            ancestry = min(float(z["observed_diagnostic_distinct_ancestors_low"][diagnostic_index]), float(z["observed_diagnostic_distinct_ancestors_high"][diagnostic_index])) / int(z["n_particles"])
            return {"available": True, "low": samples[0], "high": samples[1],
                    "actual": float(self.cohort().traces[wi][cut + horizon, t]),
                    "worm_id": self.worms[wi], "chemical": chemical, "seed": seed, "repeat": repeat,
                    "cut_frame": cut, "cut_seconds": cut / 4, "readout_frame": cut + horizon, "readout_seconds": (cut + horizon) / 4,
                    "source_window_frames": z["observed_source_window_bounds"][local, phase, event],
                    "achieved_gap": float(z["observed_diagnostic_achieved_gap"][diagnostic_index]),
                    "support_valid": bool(z["observed_diagnostic_valid"][diagnostic_index]), "ancestor_fraction": ancestry,
                    "units": "Preprocessed calcium signal (pooled-scale units), not source-gap normalized",
                    "archive": str(path.relative_to(ROOT)), "archive_sha256": sha256(path)}
        raise ValueError("Saved calibration bank lacks the requested worm/seed")

    def references(self):
        if self._references is None:
            if IS_PORTABLE:
                self._references = json.loads((PORTABLE / "external_references.json").read_text())
            else:
                data = json.loads((ATLAS / "gui/explorer_data.json").read_text())
                self._references = data["external_references"]
        return self._references


def make_handler(data):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                return self.respond(403, b"Local host required", "text/plain")
            parsed = urlsplit(self.path)
            if parsed.path.startswith("/api/"):
                try:
                    raw = parse_qs(parsed.query, max_num_fields=15)
                    if any(len(v) != 1 for v in raw.values()):
                        raise ValueError("Repeated parameters are not supported")
                    params = {k: v[0] for k, v in raw.items()}
                    routes = {"/api/meta": lambda: data.meta(), "/api/edge": lambda: data.edge(params),
                              "/api/targets": lambda: data.targets(params), "/api/controls": lambda: data.controls(params),
                              "/api/signals": lambda: data.signals(params), "/api/references": lambda: data.references(),
                              "/api/prediction": lambda: data.prediction(params)}
                    if parsed.path not in routes:
                        return self.respond(404, b"Not found", "text/plain")
                    body = json.dumps(safe_json(routes[parsed.path]()), separators=(",", ":"), allow_nan=False).encode()
                    self.respond(200, body, "application/json; charset=utf-8")
                except (ValueError, KeyError) as exc:
                    self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                except Exception as exc:
                    print(f"Data error: {type(exc).__name__}: {exc}", flush=True)
                    self.respond(500, b'{"error":"The saved data could not be loaded. See the local server log."}', "application/json")
                return
            files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if parsed.path not in files:
                return self.respond(404, b"Not found", "text/plain")
            file = WEB / files[parsed.path]
            self.respond(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "text/plain")

        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser may cancel a superseded selection.

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18781)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    data = AtlasData()
    if args.validate:
        print(json.dumps(data.meta()["verified_inputs"], indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data))
    print(f"Neural atlas v2: http://127.0.0.1:{args.port} — read-only, verified inputs", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
