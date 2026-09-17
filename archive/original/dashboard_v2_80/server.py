"""Read-only local API for the optimized historical 80-neuron atlas."""

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
IS_PORTABLE = (PORTABLE / "bundle_manifest.json").is_file()
ATLAS_ROOT = PORTABLE if IS_PORTABLE else ROOT / "results/sbtg80_optimized_full_atlas_20260902"
CANONICAL = ATLAS_ROOT / "atlas"
EXTERNAL = ATLAS_ROOT / "external_reference_checks"
SEED_CAMPAIGN = PORTABLE / "seed_sensitivity" if IS_PORTABLE else ROOT / "results/sbtg80_flow_optimization_20260901/final"
PROGRESSIVE = "progressive_bridge_smc"
CHANNELS = {
    "endpoint_mean": ("Mean activity", "Difference in mean target activity at the forecast endpoint."),
    "cumulative_mean": ("Average activity over time", "Difference in time-averaged target activity; not a sum or integral."),
    "peak_mean": ("Peak activity", "Difference in the average pathwise maximum over the forecast window."),
    "event_probability": ("Event probability", "Difference in crossing probability, normalized by achieved source displacement."),
    "endpoint_sd": ("Activity spread (SD)", "Difference in endpoint standard deviation; not variance or gain modulation."),
    "endpoint_log_sd": ("Log activity spread", "Difference in log endpoint SD; not a percentage change."),
    "endpoint_wasserstein1": ("Distribution distance (W1)", "Distance between target distributions; positive values are not a detection test."),
}
CONTEXTS = {
    "baseline": "Before stimulus", "onset": "At stimulus onset", "active": "During stimulus",
    "offset": "At stimulus offset", "recovery": "After stimulus", "state_average": "Average across all five phases",
    "onset_minus_baseline": "Onset minus baseline", "butanone_onset": "Butanone onset",
    "pentanedione_onset": "Pentanedione onset", "nacl_onset": "NaCl onset",
    "butanone_onset_minus_baseline": "Butanone onset minus baseline",
    "pentanedione_onset_minus_baseline": "Pentanedione onset minus baseline",
    "nacl_onset_minus_baseline": "NaCl onset minus baseline",
}
CI_NOTE = (
    "Shading is the saved pointwise 95% bootstrap interval for the mean across 20 historical traces "
    "(256 whole-trace bootstrap resamples). It is conditional on fitted generator seed 1701 and "
    "excludes generator-refit, imputation, and multiplicity uncertainty."
)


def safe_json(value):
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_ledger(directory: Path) -> dict[str, object]:
    ledger = directory / "checksums.sha256"
    checked = 0
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        path = (directory / filename.lstrip("*")).resolve()
        allowed_root = WEB if IS_PORTABLE else ROOT
        if not path.is_relative_to(allowed_root):
            raise ValueError("Input ledger points outside the dashboard data root")
        if sha256(path) != digest:
            raise ValueError(f"Input checksum mismatch: {path}")
        checked += 1
    return {"ledger": str(ledger), "entries": checked, "sha256": sha256(ledger)}


@lru_cache(maxsize=128)
def atlas_array(key: str) -> np.ndarray:
    with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
        value = archive[key]
    value.flags.writeable = False
    return value


@lru_cache(maxsize=8)
def worm_array(channel: str, context: str) -> np.ndarray:
    with np.load(CANONICAL / "worm_matrices.npz", allow_pickle=False) as archive:
        value = archive[f"normalized__{channel}__{context}"]
    value.flags.writeable = False
    return value


def _display_worm_id(value: str) -> str:
    return value.replace("OH16230:", "trace ").replace("SBTG:", "trace ")


class AtlasData:
    def __init__(self, verify: bool = True):
        if verify and IS_PORTABLE:
            manifest = json.loads((PORTABLE / "bundle_manifest.json").read_text())
            for entry in manifest["files"]:
                path = (WEB / entry["path"]).resolve()
                if not path.is_relative_to(WEB) or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Portable input checksum mismatch: {entry['path']}")
            self.verified = [{"ledger": "data/bundle_manifest.json", "entries": len(manifest["files"]), "sha256": sha256(PORTABLE / "bundle_manifest.json")}]
        else:
            self.verified = [verify_ledger(CANONICAL), verify_ledger(EXTERNAL)] if verify else []
        if verify and not IS_PORTABLE:
            manifest = json.loads((WEB / "input_manifest.json").read_text())
            for entry in manifest["inputs"]:
                path = (ROOT / entry["path"]).resolve()
                if not path.is_relative_to(ROOT) or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Dashboard input changed: {entry['path']}")
            self.verified.append({"ledger": "dashboard_v2_80/input_manifest.json", "entries": len(manifest["inputs"]), "sha256": sha256(WEB / "input_manifest.json")})
        with np.load(CANONICAL / "atlas_matrices.npz", allow_pickle=False) as archive:
            self.neurons = archive["neurons"].astype(str).tolist()
            self.worms = archive["worm_ids"].astype(str).tolist()
            self.lags = archive["source_lag_frames"].astype(int).tolist()
            self.horizons = archive["horizon_frames"].astype(int).tolist()
            if str(archive["orientation"]) != "target_row_source_column":
                raise ValueError("Atlas orientation is not target-row/source-column")
            if int(archive["generator_seed_count"]) != 1:
                raise ValueError("Primary dashboard atlas must contain exactly one seed")
        if len(self.neurons) != 80 or len(self.worms) != 20:
            raise ValueError("Historical atlas geometry changed")
        self.support = pd.read_csv(CANONICAL / "support_cells.csv")
        self.top = pd.read_csv(CANONICAL / "top_effects.csv")
        self.seed_choice = json.loads((ATLAS_ROOT / "seed_choice.json").read_text())
        self._cohort = None
        self._references = None
        self.cohort_lock = threading.Lock()

    def shortcuts(self) -> list[dict[str, object]]:
        rows = self.top[(self.top.channel == "endpoint_mean") & (self.top.context == "state_average") & (self.top.source_lag_frames == 1) & (self.top.horizon_frames == 1)].sort_values("slice_rank")
        chosen, origins = [], set()
        for row in rows.itertuples():
            key = (str(row.source_origin), str(row.target_origin))
            if key in origins:
                continue
            origins.add(key)
            chosen.append({"source_neuron": row.source_neuron, "target_neuron": row.target_neuron, "source_lag_frames": 1, "horizon_frames": 1, "context": "state_average", "support_pass": float(row.valid_fraction) >= 0.5})
            if len(chosen) == 3:
                break
        return chosen

    def meta(self) -> dict[str, object]:
        return {
            "title": "Historical 80-neuron atlas · effects & uncertainty", "version": "2.0-80", "snapshot": "2026-09-02",
            "neurons": self.neurons, "worm_ids": self.worms, "worm_labels": [_display_worm_id(value) for value in self.worms],
            "fps": 4, "lags": self.lags, "horizons": self.horizons,
            "channels": [{"id": key, "label": value[0], "description": value[1]} for key, value in CHANNELS.items()],
            "contexts": [{"id": key, "label": value} for key, value in CONTEXTS.items()],
            "methods": [{"id": PROGRESSIVE, "label": "Best progressive SMC"}], "shortcuts": self.shortcuts(),
            "ci_note": CI_NOTE, "verified_inputs": self.verified, "portable": IS_PORTABLE, "original_explorer_available": False,
            "scope": "20 historical traces · 80 pooled head/tail classes · single fitted generator seed",
            "origin_counts": {"head": 63, "tail": 17},
            "sampling": {"particles": 64, "repair_branch_factor": 4, "future_branch_factor": 2, "integration_steps": 20},
            "seed_choice": self.seed_choice,
        }

    def selection(self, params):
        source, target = params.get("source", "FLP"), params.get("target", "ADE")
        channel, context = params.get("channel", "endpoint_mean"), params.get("context", "state_average")
        method = params.get("method", PROGRESSIVE)
        if source not in self.neurons or target not in self.neurons or channel not in CHANNELS or context not in CONTEXTS or method != PROGRESSIVE:
            raise ValueError("Unknown source, target, outcome, context, or method")
        return self.neurons.index(source), self.neurons.index(target), channel, context, method

    def edge(self, params) -> dict[str, object]:
        source, target, channel, context, method = self.selection(params)
        suffix = f"{method}__{channel}__{context}"
        series = {short: atlas_array(f"{prefix}__{suffix}")[:, :, target, source] for short, prefix in (("mean", "mean_normalized"), ("low", "ci_low_normalized"), ("high", "ci_high_normalized"))}
        series["valid_fraction"] = atlas_array(f"valid_fraction__{method}__{context}")[:, source]
        series["genealogy_fraction"] = atlas_array(f"genealogy_valid_fraction_0_10__{method}__{context}")[:, source]
        support = self.support[(self.support.source_neuron == self.neurons[source]) & (self.support.method == method) & (self.support.context == context)].copy()
        support["genealogy_strong_gate_pass"] = support.genealogy_valid_fraction_0_10 >= 0.5
        columns = ["source_lag_frames", "support_qualified", "genealogy_strong_gate_pass", "valid_fraction", "genealogy_valid_fraction_0_10", "genealogy_valid_fraction_0_20", "mean_min_distinct_ancestor_fraction", "mean_ess_low", "mean_ess_high", "mean_achieved_gap"]
        return {
            "selection": {"source": self.neurons[source], "target": self.neurons[target], "channel": channel, "context": context, "method": method},
            "series": {method: series}, "worms": worm_array(channel, context)[:, :, :, target, source], "worm_values_method": method,
            "cell_evidence": [], "edge_evidence": None, "calibration": [], "support": support[columns].to_dict("records"),
            "self_pair": source == target, "ci_note": CI_NOTE,
            "signed": channel != "endpoint_wasserstein1" or context.endswith("minus_baseline"),
            "lineage_warning": "Historical pseudo-paired head/tail and donor-imputed traces; not a clean simultaneous 80-neuron recording.",
        }

    def targets(self, params) -> dict[str, object]:
        source, _, channel, context, method = self.selection(params)
        lag, horizon = int(params.get("lag", 1)), int(params.get("horizon", 1))
        if lag not in self.lags or horizon not in self.horizons:
            raise ValueError("Unsupported timing")
        li, hi, suffix = self.lags.index(lag), self.horizons.index(horizon), f"{method}__{channel}__{context}"
        values = {key: atlas_array(f"{prefix}__{suffix}")[li, hi, :, source] for key, prefix in (("mean", "mean_normalized"), ("low", "ci_low_normalized"), ("high", "ci_high_normalized"))}
        return {"neurons": self.neurons, **values, "note": "Alphabetical targets; intervals are pointwise and descriptive."}

    def controls(self, params) -> dict[str, object]:
        self.selection(params)
        return {"available": False, "reason": "No separate matched sampling-null experiment was run for the historical 80-neuron atlas. Use ESS, validity, and ancestry as numerical diagnostics, not as a biological test."}

    def prediction(self, params) -> dict[str, object]:
        self.selection(params)
        return {"available": False, "reason": "The full historical-80 run archived response functionals and diagnostics, not individual generated endpoint draws."}

    def cohort(self):
        with self.cohort_lock:
            if self._cohort is None:
                if IS_PORTABLE:
                    with np.load(PORTABLE / "observed_cohort.npz", allow_pickle=False) as archive:
                        lengths = archive["lengths"].astype(int)
                        traces = tuple(archive["traces"][index, :length].copy() for index, length in enumerate(lengths))
                        intervals, chemicals = archive["event_intervals_seconds"], archive["chemical_name_by_worm_event"].astype(str)
                        schedules = tuple(SimpleNamespace(event_intervals_seconds=tuple(map(tuple, intervals[index].tolist())), chemical_name_by_event=tuple(chemicals[index].tolist())) for index in range(len(lengths)))
                        cohort = SimpleNamespace(traces=traces, neurons=archive["neurons"].astype(str).tolist(), worm_ids=archive["worm_ids"].astype(str).tolist(), fps=float(archive["fps"]), stimulus_schedules=schedules)
                else:
                    from conditional_neural_benchmark.data import load_sbtg_cohort
                    cohort = load_sbtg_cohort()
                if list(cohort.neurons) != self.neurons or list(cohort.worm_ids) != self.worms:
                    raise ValueError("Recorded cohort does not match atlas coordinates")
                self._cohort = cohort
        return self._cohort

    @lru_cache(maxsize=32)
    def signal_data(self, source: str, target: str, chemical: str):
        cohort = self.cohort()
        if chemical not in ("butanone", "pentanedione", "nacl"):
            raise ValueError("Select one actual chemical event")
        indices, times, rows = [self.neurons.index(source), self.neurons.index(target)], np.arange(-80, 121) / cohort.fps, []
        for index, (raw, schedule) in enumerate(zip(cohort.traces, cohort.stimulus_schedules)):
            event = schedule.chemical_name_by_event.index(chemical)
            onset, offset = schedule.event_intervals_seconds[event]
            start, trace = int(round(onset * cohort.fps)), np.asarray(raw[:, indices], dtype=float)
            standardized = (trace - np.nanmean(trace, axis=0)) / np.maximum(np.nanstd(trace, axis=0), 1e-8)
            frames, window = start + np.arange(-80, 121), np.full((len(times), 2), np.nan)
            valid = (frames >= 0) & (frames < len(trace)); window[valid] = standardized[frames[valid]]
            rows.append({"worm_id": cohort.worm_ids[index], "worm_label": _display_worm_id(cohort.worm_ids[index]), "values": window.T, "duration": offset - onset})
        values = np.stack([row["values"] for row in rows])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning); quantiles = np.nanquantile(values, [0.1, 0.5, 0.9], axis=0)
        return {"source": source, "target": target, "chemical": chemical, "time": times, "rows": rows, "lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2], "n_observed": np.isfinite(values).sum(axis=0), "n_worms": len(rows), "units": "Within-trace, within-neuron display z-score", "note": "Historical preprocessed signals. The band describes trace variation, not measurement uncertainty. Head/tail traces were pseudo-paired and missing traces donor-imputed."}

    def signals(self, params):
        self.selection(params)
        return self.signal_data(params.get("source", "FLP"), params.get("target", "ADE"), params.get("chemical", "butanone"))

    def references(self) -> dict[str, object]:
        if self._references is not None:
            return self._references
        if IS_PORTABLE:
            self._references = json.loads((PORTABLE / "external_references.json").read_text())
            return self._references
        all_metrics = pd.read_csv(EXTERNAL / "randi_cook_metrics.csv")
        progressive = all_metrics[
            (all_metrics.method == PROGRESSIVE)
            & (all_metrics.channel == "endpoint_mean")
            & (all_metrics.context == "state_average")
            & (all_metrics.scope == "all_estimated")
        ].copy()
        published = all_metrics[
            (all_metrics.method == "sbtg_published")
            & (all_metrics.lag_frames == 1)
            & (all_metrics.scope == "all_estimated")
        ].copy()
        progressive["timing_label"] = progressive.apply(lambda row: f"{int(row.lag_frames)/4:g} s lag · {float(row.horizon_frames)/4:g} s forecast", axis=1)
        published["timing_label"] = "published lag-1 score product"
        reference = pd.concat([progressive, published], ignore_index=True)
        reference["reference_or_network"], reference["continuous_spearman"] = reference.reference, reference.absolute_spearman
        reference["panel"], reference["scope_or_grid"] = "Randi/Cook · historical 80 · all estimated", reference.scope
        keep = ["reference_or_network", "method", "channel", "context", "panel", "scope_or_grid", "timing_label", "auroc", "auprc", "continuous_spearman"]
        rows = reference[keep].to_dict("records")
        ensemble = pd.read_csv(SEED_CAMPAIGN / "ensemble_primary_metrics.csv")
        ensemble = ensemble[(ensemble.evaluation_axis == "historical80") & (ensemble.scope == "all_estimated")].copy()
        ensemble["reference_or_network"], ensemble["method"] = ensemble.reference, "two_seed_sensitivity"
        ensemble["continuous_spearman"], ensemble["panel"] = ensemble.absolute_spearman, "Randi/Cook · historical 80 · all estimated"
        ensemble["scope_or_grid"], ensemble["timing_label"] = "two-seed sensitivity", "0.25 s lag · 0.25 s forecast"
        rows.extend(ensemble[keep].to_dict("records"))
        bentley = pd.read_csv(EXTERNAL / "bentley_metrics.csv")
        progressive_bentley = bentley[
            (bentley.method == PROGRESSIVE)
            & (bentley.channel.isin(["endpoint_mean", "endpoint_wasserstein1"]))
            & (bentley.context == "state_average")
            & (bentley.lag_frames == 1)
            & (bentley.horizon_frames == 1)
            & (bentley.scope == "eligible_support_qualified")
        ].copy()
        published_bentley = bentley[
            (bentley.method == "sbtg_published")
            & (bentley.lag_frames == 1)
            & (bentley.scope == "eligible_sources")
        ].copy()
        for frame in (progressive_bentley, published_bentley):
            frame["reference_or_network"] = frame.network
            frame["continuous_spearman"] = np.nan
            frame["panel"] = "Bentley · neuromodulator-specific · lag-1"
            frame["scope_or_grid"] = frame.scope
        progressive_bentley["timing_label"] = "0.25 s lag · 0.25 s forecast"
        published_bentley["timing_label"] = "published lag-1 score product"
        rows.extend(progressive_bentley[keep].to_dict("records"))
        rows.extend(published_bentley[keep].to_dict("records"))

        lagmax = pd.read_csv(EXTERNAL / "bentley_lagmax_inference.csv")
        lagmax = lagmax[
            (lagmax.lag_grid == "native_method_grid")
            & (
                (
                    (lagmax.method == PROGRESSIVE)
                    & (lagmax.channel.isin(["endpoint_mean", "endpoint_wasserstein1"]))
                    & (lagmax.context == "state_average")
                    & (lagmax.horizon_frames == 1)
                )
                | (lagmax.method == "sbtg_published")
            )
        ].copy()
        lagmax_records = {}
        lagmax_keep = [
            "method", "channel", "context", "network", "evaluable",
            "non_evaluable_reason", "n_common_supported_eligible_sources",
            "n_positive", "best_lag_frames", "best_lag_index_seconds",
            "best_auroc", "max_lag_permutation_p", "max_lag_bh_q",
            "lag_semantics", "inference_limit",
        ]
        for network, frame in lagmax.groupby("network"):
            lagmax_records[str(network)] = frame[lagmax_keep].to_dict("records")
        single_bootstrap = pd.read_csv(EXTERNAL / "paired_source_bootstrap.csv")
        single_bootstrap = single_bootstrap[single_bootstrap.metric == "auroc"]
        single_intervals = {row.reference: {"difference": row.progressive_minus_sbtg_published, "low": row.ci_low, "high": row.ci_high} for row in single_bootstrap.itertuples()}
        bootstrap = pd.read_csv(SEED_CAMPAIGN / "paired_source_bootstrap.csv"); bootstrap = bootstrap[bootstrap.metric == "auroc"]
        intervals = {row.reference: {"difference": row.progressive_minus_sbtg_published, "low": row.ci_low, "high": row.ci_high} for row in bootstrap.itertuples()}
        bentley_networks = sorted(progressive_bentley.network.unique().astype(str))
        reference_notes = {
            network: (
                "Bentley is a binary directed receptor/pathway reference. "
                "AUROC and AUPRC describe edge ranking; they do not measure receptor strength, "
                "causal propagation, or physical delay. Specific monoamines have only one or two "
                "eligible source neurons, and tyramine has no support-qualified progressive source at lag 1."
            )
            for network in bentley_networks
        }
        self._references = {
            "comparisons": rows,
            "bentley_lagmax": lagmax_records,
            "reference_notes": reference_notes,
            "primary_single_seed_delta_intervals": single_intervals,
            "primary_two_seed_delta_intervals": intervals,
            "note": (
                "The primary dashboard atlas uses seed 1701. Two-seed values are a separate lag-1 "
                "sensitivity; source-bootstrap intervals do not include generator refits. Bentley "
                "lag maxima use 999 within-source permutations and global Benjamini-Hochberg adjustment."
            ),
        }
        return self._references


def make_handler(data: AtlasData):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                return self.respond(403, b"Local host required", "text/plain")
            parsed = urlsplit(self.path)
            if parsed.path.startswith("/api/"):
                try:
                    raw = parse_qs(parsed.query, max_num_fields=15)
                    if any(len(values) != 1 for values in raw.values()):
                        raise ValueError("Repeated parameters are not supported")
                    params = {key: values[0] for key, values in raw.items()}
                    routes = {"/api/meta": data.meta, "/api/edge": lambda: data.edge(params), "/api/targets": lambda: data.targets(params), "/api/controls": lambda: data.controls(params), "/api/signals": lambda: data.signals(params), "/api/references": data.references, "/api/prediction": lambda: data.prediction(params)}
                    if parsed.path not in routes:
                        return self.respond(404, b"Not found", "text/plain")
                    body = json.dumps(safe_json(routes[parsed.path]()), separators=(",", ":"), allow_nan=False).encode()
                    self.respond(200, body, "application/json; charset=utf-8")
                except (ValueError, KeyError) as error:
                    self.respond(400, json.dumps({"error": str(error)}).encode(), "application/json")
                except Exception as error:
                    print(f"Data error: {type(error).__name__}: {error}", flush=True)
                    self.respond(500, b'{"error":"The saved data could not be loaded. See the local server log."}', "application/json")
                return
            files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if parsed.path not in files:
                return self.respond(404, b"Not found", "text/plain")
            path = WEB / files[parsed.path]
            self.respond(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "text/plain")

        def respond(self, status: int, body: bytes, content_type: str):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache"); self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--port", type=int, default=18783); parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(); data = AtlasData()
    if args.validate:
        print(json.dumps(data.meta()["verified_inputs"], indent=2)); return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data))
    print(f"Historical 80-neuron atlas: http://127.0.0.1:{args.port} — read-only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
