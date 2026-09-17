from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
STIMULUS_SCHEMA_VERSION = "neuropal_head_chemical_v1"
STIMULUS_NAMES = ("butanone", "pentanedione", "nacl")


@dataclass(frozen=True)
class StimulusSchedule:
    """Frozen recording-specific causal stimulus metadata."""

    worm_id: str
    strain: str
    source_recording: str
    native_fps: float
    analysis_fps: float
    stimulus_names: tuple[str, ...]
    event_intervals_seconds: tuple[tuple[float, float], ...]
    chemical_code_by_event: tuple[int, ...]
    chemical_name_by_event: tuple[str, ...]
    resampling_provenance: str

    def __post_init__(self) -> None:
        n_events = len(self.event_intervals_seconds)
        if n_events != 3:
            raise ValueError("the NeuroPAL schedule must contain exactly three events")
        if tuple(sorted(self.chemical_code_by_event)) != (1, 2, 3):
            raise ValueError("chemical codes must be a permutation of 1/2/3")
        if len(self.chemical_name_by_event) != n_events:
            raise ValueError("chemical name and interval counts disagree")
        expected = tuple(
            self.stimulus_names[code - 1] for code in self.chemical_code_by_event
        )
        if tuple(self.chemical_name_by_event) != expected:
            raise ValueError("chemical names do not match the 1-based code semantics")
        if self.native_fps <= 0 or self.analysis_fps <= 0:
            raise ValueError("sampling rates must be positive")
        for start, end in self.event_intervals_seconds:
            if not (0 <= start < end):
                raise ValueError("stimulus intervals must be positive half-open intervals")

    def to_dict(self) -> dict[str, object]:
        return {
            "worm_id": self.worm_id,
            "strain": self.strain,
            "source_recording": self.source_recording,
            "native_fps": self.native_fps,
            "analysis_fps": self.analysis_fps,
            "stimulus_names": list(self.stimulus_names),
            "event_intervals_seconds": [list(value) for value in self.event_intervals_seconds],
            "chemical_code_by_event": list(self.chemical_code_by_event),
            "chemical_name_by_event": list(self.chemical_name_by_event),
            "resampling_provenance": self.resampling_provenance,
        }


@dataclass(frozen=True)
class Cohort:
    traces: tuple[np.ndarray, ...]
    worm_ids: tuple[str, ...]
    strains: tuple[str, ...]
    neurons: tuple[str, ...]
    fps: float
    coverage: float
    stimulus_schedules: tuple[StimulusSchedule, ...] = ()
    cohort_mode: str = "unspecified"
    stimulus_schema_version: str = STIMULUS_SCHEMA_VERSION
    lineage_warning: str | None = None
    quality_dropped_neurons: tuple[str, ...] = ()
    raw_nonfinite_values: int = 0

    @property
    def n_worms(self) -> int:
        return len(self.traces)

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def stimulus_schema_fingerprint(self) -> str:
        if len(self.stimulus_schedules) != self.n_worms:
            raise ValueError("every cohort trace must have one frozen stimulus schedule")
        payload = {
            "version": self.stimulus_schema_version,
            "schedules": [schedule.to_dict() for schedule in self.stimulus_schedules],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def stimulus_schema_dict(self) -> dict[str, object]:
        return {
            "version": self.stimulus_schema_version,
            "fingerprint": self.stimulus_schema_fingerprint,
            "schedules": [schedule.to_dict() for schedule in self.stimulus_schedules],
        }


@dataclass(frozen=True)
class FoldScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, traces: Iterable[np.ndarray]) -> "FoldScaler":
        pooled = np.concatenate(tuple(traces), axis=0).astype(np.float64, copy=False)
        mean = np.nanmean(pooled, axis=0)
        scale = np.nanstd(pooled, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, trace: np.ndarray) -> np.ndarray:
        return ((trace - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class Windows:
    history: np.ndarray  # [examples, lag, neurons + stimulus]
    target: np.ndarray  # [examples, neurons]
    worm: np.ndarray  # integer cohort index
    time: np.ndarray  # target frame
    stratum: np.ndarray  # off, active, onset, offset, mixed
    chemical_code: np.ndarray  # 0 baseline, otherwise 1-based true chemical code
    event_position: np.ndarray  # 0 baseline, otherwise 1/2/3 epoch position
    n_dropped: int = 0

    def flat(self) -> np.ndarray:
        return self.history.reshape(len(self.history), -1)

    def take(self, index: np.ndarray) -> "Windows":
        index = np.asarray(index, dtype=np.int64)
        return Windows(
            history=self.history[index],
            target=self.target[index],
            worm=self.worm[index],
            time=self.time[index],
            stratum=self.stratum[index],
            chemical_code=self.chemical_code[index],
            event_position=self.event_position[index],
            n_dropped=self.n_dropped,
        )


def balance_window_strata(
    windows: Windows,
    *,
    seed: int,
    total_rows: int | None = None,
    replace_rare: bool = True,
) -> Windows:
    """Return an equal-stratum training or validation view.

    Training can retain the original row count by resampling rare regimes and
    downsampling the dominant off regime.  For validation, callers can set
    ``replace_rare=False`` and omit ``total_rows`` to use the largest balanced
    subset containing no duplicated rows.
    """
    labels = np.asarray(windows.stratum).astype(str)
    strata = np.unique(labels)
    if len(strata) < 2:
        return windows
    groups = {label: np.flatnonzero(labels == label) for label in strata}
    if total_rows is None:
        per_stratum = min(len(index) for index in groups.values())
    else:
        if total_rows < len(strata):
            raise ValueError("total_rows must provide at least one row per stratum")
        per_stratum = int(np.ceil(total_rows / len(strata)))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in strata:
        candidates = groups[label]
        use_replace = bool(replace_rare and per_stratum > len(candidates))
        size = per_stratum if use_replace else min(per_stratum, len(candidates))
        selected.append(rng.choice(candidates, size=size, replace=use_replace))
    index = np.concatenate(selected)
    rng.shuffle(index)
    if total_rows is not None:
        index = index[:total_rows]
    return windows.take(index)


def resample_window_strata(
    windows: Windows,
    *,
    proportions: dict[str, float],
    seed: int,
    total_rows: int | None = None,
) -> Windows:
    """Resample a fixed-size view toward declared stratum proportions."""
    labels = np.asarray(windows.stratum).astype(str)
    present = sorted(set(labels.tolist()))
    unknown = set(proportions) - set(present)
    if unknown:
        raise ValueError(f"requested absent strata: {sorted(unknown)}")
    raw = np.asarray([float(proportions.get(label, 0.0)) for label in present])
    if np.any(raw < 0) or not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ValueError("stratum proportions must be finite, nonnegative, and nonzero")
    weights = raw / raw.sum()
    n_rows = len(labels) if total_rows is None else int(total_rows)
    expected = weights * n_rows
    counts = np.floor(expected).astype(int)
    for index in np.argsort(expected - counts)[::-1][: n_rows - counts.sum()]:
        counts[index] += 1
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label, count in zip(present, counts):
        candidates = np.flatnonzero(labels == label)
        selected.append(
            rng.choice(candidates, size=int(count), replace=int(count) > len(candidates))
        )
    index = np.concatenate(selected)
    rng.shuffle(index)
    return windows.take(index)


STIMULUS_PERIODS_SECONDS = ((60.5, 70.5), (120.5, 130.5), (180.5, 190.5))


STIMULUS_ENCODINGS = (
    "binary_any_stimulus",
    "position_onehot",
    "chemical_scalar",
    "chemical_onehot",
    "chemical_plus_position_onehot",
    "chemical_onehot_subject_shuffle",
)


def _event_frame_mask(n_frames: int, schedule: StimulusSchedule, event: int) -> np.ndarray:
    """Return a half-open time mask on the schedule's explicit analysis grid."""
    start, end = schedule.event_intervals_seconds[event]
    time = np.arange(n_frames, dtype=np.float64) / schedule.analysis_fps
    return (time >= start) & (time < end)


def _stimulus_mask(n_frames: int, schedule: StimulusSchedule) -> np.ndarray:
    result = np.zeros(n_frames, dtype=np.float32)
    for event in range(len(schedule.event_intervals_seconds)):
        result[_event_frame_mask(n_frames, schedule, event)] = 1.0
    return result


def _stimulus_features(
    n_frames: int,
    schedule: StimulusSchedule,
    encoding: str,
    chemical_order_override: Iterable[int] | None = None,
) -> np.ndarray:
    """Return active-only causal features from a worm-specific frozen schedule.

    Chemical codes are 1-based categorical labels indexing ``stimulus_names``.
    A shuffled control must explicitly provide a per-animal permutation; there
    is no silent position-order default.
    """
    if encoding not in STIMULUS_ENCODINGS:
        raise ValueError(f"unknown stimulus encoding: {encoding}")
    actual = tuple(int(value) for value in schedule.chemical_code_by_event)
    if chemical_order_override is None:
        codes = actual
    else:
        codes = tuple(int(value) for value in chemical_order_override)
        if tuple(sorted(codes)) != (1, 2, 3):
            raise ValueError("chemical_order_override must be a permutation of 1/2/3")
    if encoding == "chemical_onehot_subject_shuffle":
        if chemical_order_override is None:
            raise ValueError("shuffled chemical encoding requires a frozen per-worm order")
    elif chemical_order_override is not None:
        raise ValueError("chemical_order_override is only valid for the shuffled control")

    n_channels = {
        "binary_any_stimulus": 1,
        "position_onehot": 3,
        "chemical_scalar": 1,
        "chemical_onehot": 3,
        "chemical_plus_position_onehot": 6,
        "chemical_onehot_subject_shuffle": 3,
    }[encoding]
    result = np.zeros((n_frames, n_channels), dtype=np.float32)
    for event in range(3):
        active = _event_frame_mask(n_frames, schedule, event)
        chemical = codes[event] - 1
        if encoding == "binary_any_stimulus":
            result[active, 0] = 1.0
        elif encoding == "position_onehot":
            result[active, event] = 1.0
        elif encoding == "chemical_scalar":
            result[active, 0] = float(codes[event])
        elif encoding == "chemical_plus_position_onehot":
            result[active, chemical] = 1.0
            result[active, 3 + event] = 1.0
        else:
            result[active, chemical] = 1.0
    return result


def _schedule_from_recording(
    *, strain: str, raw_id: str, data: dict, worm_index: int, analysis_fps: float,
) -> StimulusSchedule:
    names = tuple(str(value).strip().lower() for value in data["stim_names"])
    if names != STIMULUS_NAMES:
        raise ValueError(f"unexpected stimulus name order in {strain}: {names}")
    codes = tuple(int(value) for value in data["stims_per_worm"][worm_index])
    intervals = tuple(
        (float(value[0]), float(value[1])) for value in np.asarray(data["stim_times"])
    )
    source = data.get("source_recording")
    if source is None:
        source = ROOT / "SBTG" / "data" / f"Head_Activity_{strain}.mat"
    native_fps = float(data["fps"])
    resampling = (
        "none_native_grid"
        if np.isclose(native_fps, analysis_fps)
        else f"linear_zero_origin_{native_fps:g}_to_{analysis_fps:g}_hz_before_windowing"
    )
    return StimulusSchedule(
        worm_id=f"{strain}:{raw_id}",
        strain=strain,
        source_recording=str(Path(source).resolve()),
        native_fps=native_fps,
        analysis_fps=float(analysis_fps),
        stimulus_names=names,
        event_intervals_seconds=intervals,
        chemical_code_by_event=codes,
        chemical_name_by_event=tuple(names[code - 1] for code in codes),
        resampling_provenance=resampling,
    )


def load_cohort(
    coverage: float = 0.90,
    complete_case: bool = True,
    *,
    cohort_mode: str = "pooled_resampled",
) -> Cohort:
    """Load a head-only cohort with frozen per-recording stimulus metadata.

    ``oh16230_head`` is the primary clean native-4.0-Hz analysis.  The pooled
    sensitivity explicitly resamples OH15500 from 4.1 Hz to 4.0 Hz before
    windowing and preserves its 20-second final event.
    """
    from sid_elegans import combined_data as combined

    if cohort_mode not in {"oh16230_head", "pooled_resampled"}:
        raise ValueError("cohort_mode must be oh16230_head or pooled_resampled")

    traces, neurons, fps = combined.load_combined(
        coverage_frac=coverage,
        complete_case=complete_case,
        signal="raw",
        verbose=True,
    )
    if not complete_case:
        raise ValueError("the primary runner currently requires complete-case traces")

    # load_combined emits strains in this exact dataset/worm order. Recover the
    # retained identifiers and exact stimulus schedules without changing or
    # duplicating any trace.
    data16 = combined._prep.load_neuropal_data(
        combined.REPO / "data", include_tail=False, collapse_dv=True
    )
    data16["source_recording"] = str(
        combined.REPO / "data" / "Head_Activity_OH16230.mat"
    )
    data15 = combined._load15()
    worm_ids: list[str] = []
    strains: list[str] = []
    schedules: list[StimulusSchedule] = []
    for strain, data in (("OH16230", data16), ("OH15500", data15)):
        for worm_idx, raw_id in enumerate(data["worm_ids"]):
            matrix = combined._worm_matrix(data, worm_idx, neurons, complete_case=True)
            if matrix is not None:
                worm_ids.append(f"{strain}:{raw_id}")
                strains.append(strain)
                schedules.append(_schedule_from_recording(
                    strain=strain,
                    raw_id=str(raw_id),
                    data=data,
                    worm_index=worm_idx,
                    analysis_fps=float(fps),
                ))
    if len(worm_ids) != len(traces):
        raise RuntimeError("worm provenance reconstruction disagrees with combined loader")
    traces = tuple(np.asarray(x, dtype=np.float32) for x in traces)
    raw_nonfinite = int(sum((~np.isfinite(x)).sum() for x in traces))
    # A trace can exist while being almost entirely NaN. Two such coordinates
    # occur in individual OH15500 worms (AIN and ADA). Keeping them would either
    # discard most windows or manufacture a target trace. Apply a declared
    # cross-worm observation-quality rule instead: every retained coordinate
    # must be observed on at least 90% of frames in every retained worm.
    finite_fraction = np.asarray(
        [[np.isfinite(x[:, j]).mean() for j in range(x.shape[1])] for x in traces]
    )
    keep = finite_fraction.min(axis=0) >= 0.90
    dropped = tuple(np.asarray(neurons)[~keep].tolist())
    traces = tuple(x[:, keep] for x in traces)
    neurons = list(np.asarray(neurons)[keep])
    if cohort_mode == "oh16230_head":
        keep_worm = [index for index, strain in enumerate(strains) if strain == "OH16230"]
        traces = tuple(traces[index] for index in keep_worm)
        worm_ids = [worm_ids[index] for index in keep_worm]
        strains = [strains[index] for index in keep_worm]
        schedules = [schedules[index] for index in keep_worm]
    return Cohort(
        traces=traces,
        worm_ids=tuple(worm_ids),
        strains=tuple(strains),
        neurons=tuple(neurons),
        fps=float(fps),
        coverage=float(coverage),
        stimulus_schedules=tuple(schedules),
        cohort_mode=cohort_mode,
        quality_dropped_neurons=dropped,
        raw_nonfinite_values=raw_nonfinite,
    )


def load_sbtg_cohort(
    dataset: str = "full_traces_imputed",
    *,
    neuron_subset: Iterable[str] | None = None,
) -> Cohort:
    """Load the frozen SBTG 20-worm/80-neuron cache without new imputation.

    The cache is already globally standardized by the released SBTG preparation
    pipeline. A small number of nonfinite values occur in the final padded
    frames; :func:`make_windows` removes only windows touching those values.
    ``neuron_subset`` preserves the requested order and is used for the
    54-neuron bridge cohort.
    """
    directory = (
        ROOT / "SBTG" / "results" / "intermediate" / "datasets" / str(dataset)
    )
    if not directory.exists():
        raise FileNotFoundError(f"SBTG dataset cache is unavailable: {directory}")
    raw = np.load(directory / "X_segments.npy", allow_pickle=True)
    traces = tuple(np.asarray(value, dtype=np.float32) for value in raw)
    names_path = directory / "neuron_names.json"
    if names_path.exists():
        names = [str(value).strip().upper() for value in json.loads(names_path.read_text())]
    else:
        metadata = json.loads((directory / "standardization.json").read_text())
        names = [str(value).strip().upper() for value in metadata["node_order"]]
    if any(value.ndim != 2 or value.shape[1] != len(names) for value in traces):
        raise ValueError("SBTG trace dimensions disagree with the frozen neuron order")
    if neuron_subset is not None:
        requested = [str(value).strip().upper() for value in neuron_subset]
        lookup = {name: index for index, name in enumerate(names)}
        missing = [name for name in requested if name not in lookup]
        if missing:
            raise ValueError(f"SBTG bridge is missing requested neurons: {missing}")
        index = np.asarray([lookup[name] for name in requested], dtype=np.int64)
        traces = tuple(value[:, index] for value in traces)
        names = requested
    segments = pd.read_csv(directory / "segments.csv")
    if len(segments) != len(traces):
        raise ValueError("SBTG segments metadata disagrees with cached traces")
    worm_ids = tuple(f"SBTG:{value}" for value in segments.worm_id.astype(str))
    from sid_elegans import combined_data as combined
    raw = combined._prep.load_neuropal_data(
        combined.REPO / "data", include_tail=False, collapse_dv=True
    )
    raw["source_recording"] = str(
        combined.REPO / "data" / "Head_Activity_OH16230.mat"
    )
    raw_lookup = {str(value): index for index, value in enumerate(raw["worm_ids"])}
    schedules = tuple(
        _schedule_from_recording(
            strain="OH16230",
            raw_id=str(value),
            data=raw,
            worm_index=raw_lookup[str(value)],
            analysis_fps=4.0,
        )
        for value in segments.worm_id.astype(str)
    )
    raw_nonfinite = int(sum((~np.isfinite(value)).sum() for value in traces))
    return Cohort(
        traces=traces,
        worm_ids=worm_ids,
        strains=tuple("OH16230" for _ in traces),
        neurons=tuple(names),
        fps=4.0,
        coverage=1.0,
        stimulus_schedules=schedules,
        cohort_mode=f"historical_sbtg_{dataset}",
        lineage_warning=(
            "historical/contextual only: released 80-neuron lineage used invalid "
            "index-wise head/tail fusion and donor-trace imputation"
        ),
        raw_nonfinite_values=raw_nonfinite,
    )


def make_fold_assignments(cohort: Cohort, n_folds: int = 5, seed: int = 20260825) -> np.ndarray:
    """Stratify strains while keeping every worm wholly inside one fold."""
    if n_folds < 2 or n_folds > cohort.n_worms:
        raise ValueError("invalid number of folds")
    # StratifiedKFold cannot have more folds than the small OH15500 stratum.
    # Round-robin each shuffled strain instead, offsetting strata so rare worms
    # are spread across folds and all folds stay close in size.
    rng = np.random.default_rng(seed)
    folds = np.full(cohort.n_worms, -1, dtype=np.int64)
    offset = 0
    for strain in sorted(set(cohort.strains)):
        idx = np.flatnonzero(np.asarray(cohort.strains) == strain)
        rng.shuffle(idx)
        for position, worm_idx in enumerate(idx):
            folds[worm_idx] = (position + offset) % n_folds
        offset = (offset + len(idx)) % n_folds
    if np.any(folds < 0):
        raise RuntimeError("failed to assign every worm")
    return folds


def _strata(stimulus_windows: np.ndarray) -> np.ndarray:
    if stimulus_windows.shape[1] == 1:
        return np.where(stimulus_windows[:, 0] > 0.5, "active", "off")
    changes = np.diff(stimulus_windows, axis=1)
    onset = np.any(changes > 0.5, axis=1)
    offset = np.any(changes < -0.5, axis=1)
    result = np.full(len(stimulus_windows), "off", dtype="U8")
    result[np.all(stimulus_windows > 0.5, axis=1)] = "active"
    result[onset] = "onset"
    result[offset] = "offset"
    result[onset & offset] = "mixed"
    return result


def make_windows(
    cohort: Cohort,
    worm_indices: Iterable[int],
    lag: int,
    scaler: FoldScaler,
    *,
    stimulus_encoding: str = "binary_any_stimulus",
    chemical_permutations: dict[int, tuple[int, int, int]] | None = None,
) -> Windows:
    """Construct one-step targets and exactly aligned stimulus histories."""
    if lag < 1:
        raise ValueError("lag must be positive")
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    worms: list[np.ndarray] = []
    times: list[np.ndarray] = []
    strata: list[np.ndarray] = []
    chemical_codes: list[np.ndarray] = []
    event_positions: list[np.ndarray] = []
    if len(cohort.stimulus_schedules) != cohort.n_worms:
        raise ValueError("make_windows requires one frozen stimulus schedule per worm")
    for worm_idx in worm_indices:
        schedule = cohort.stimulus_schedules[int(worm_idx)]
        if not np.isclose(schedule.analysis_fps, cohort.fps):
            raise ValueError("schedule and cohort analysis clocks disagree")
        observed = scaler.transform(cohort.traces[int(worm_idx)])
        if len(observed) <= lag:
            continue
        # Causal carry-forward is used only inside histories. Rows whose target
        # vector was not actually observed remain excluded. Initial missing
        # values remain NaN and therefore exclude their affected windows.
        trace = observed.copy()
        for column in range(trace.shape[1]):
            finite = np.isfinite(trace[:, column])
            last = np.maximum.accumulate(np.where(finite, np.arange(len(trace)), -1))
            usable = last >= 0
            trace[usable, column] = trace[last[usable], column]
        active_stim = _stimulus_mask(len(trace), schedule)
        frame_chemical = np.zeros(len(trace), dtype=np.int8)
        frame_position = np.zeros(len(trace), dtype=np.int8)
        for event, code in enumerate(schedule.chemical_code_by_event):
            active = _event_frame_mask(len(trace), schedule, event)
            frame_chemical[active] = int(code)
            frame_position[active] = event + 1
        chemical_order = (
            None
            if chemical_permutations is None
            else chemical_permutations[int(worm_idx)]
        )
        stim = _stimulus_features(
            len(trace), schedule, stimulus_encoding, chemical_order
        )
        neural_windows = np.lib.stride_tricks.sliding_window_view(
            trace, window_shape=lag, axis=0
        )[:-1].transpose(0, 2, 1)
        stim_windows = np.lib.stride_tricks.sliding_window_view(
            stim, window_shape=lag, axis=0
        )[:-1].transpose(0, 2, 1)
        active_windows = np.lib.stride_tricks.sliding_window_view(
            active_stim, lag
        )[:-1]
        candidate_history = np.concatenate(
            [neural_windows, stim_windows], axis=2
        ).astype(np.float32, copy=False)
        candidate_target = observed[lag:].astype(np.float32, copy=False)
        valid = np.isfinite(candidate_history).all(axis=(1, 2)) & np.isfinite(candidate_target).all(axis=1)
        histories.append(candidate_history[valid])
        targets.append(candidate_target[valid])
        count = int(valid.sum())
        worms.append(np.full(count, int(worm_idx), dtype=np.int16))
        times.append(np.arange(lag, len(trace), dtype=np.int32)[valid])
        strata.append(_strata(active_windows)[valid])
        chemical_codes.append(frame_chemical[lag:][valid])
        event_positions.append(frame_position[lag:][valid])
    if not histories:
        raise ValueError("no windows were created")
    possible = sum(max(0, len(cohort.traces[int(i)]) - lag) for i in worm_indices)
    kept = sum(len(x) for x in histories)
    return Windows(
        history=np.concatenate(histories),
        target=np.concatenate(targets),
        worm=np.concatenate(worms),
        time=np.concatenate(times),
        stratum=np.concatenate(strata),
        chemical_code=np.concatenate(chemical_codes),
        event_position=np.concatenate(event_positions),
        n_dropped=int(possible - kept),
    )


def split_history(history: np.ndarray, n_neurons: int) -> tuple[np.ndarray, np.ndarray]:
    """Split neural and stimulus channels with an explicit neuron boundary."""
    history = np.asarray(history)
    if history.ndim < 3:
        raise ValueError("history must have shape [..., lag, channels]")
    if n_neurons < 1 or history.shape[-1] <= n_neurons:
        raise ValueError("history must contain neural channels followed by stimulus channels")
    return history[..., :n_neurons], history[..., n_neurons:]


def multiscale_features(history: np.ndarray, n_neurons: int) -> np.ndarray:
    """Compact causal features for linear VAR-style distribution baselines."""
    neural, stimulus = split_history(history, n_neurons)
    stim = np.max(stimulus, axis=-1)
    lag = neural.shape[1]
    pieces = [neural[:, -1]]
    if lag > 1:
        pieces.append(neural[:, -1] - neural[:, -2])
    for width in sorted(set(min(lag, w) for w in (2, 4, 8, 16, 32, 64))):
        pieces.append(neural[:, -width:].mean(axis=1))
        if width >= 4:
            pieces.append(neural[:, -width:].std(axis=1))
    stim_summary = np.stack(
        [stim[:, -1], stim.mean(axis=1), np.abs(np.diff(stim, axis=1)).sum(axis=1)],
        axis=1,
    )
    pieces.append(stim_summary)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def choose_evaluation_indices(
    strata: np.ndarray, max_rows: int, seed: int
) -> np.ndarray:
    """Keep rare stimulus regimes intact and subsample only abundant regimes."""
    n = len(strata)
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    present = sorted(set(strata.tolist()))
    rare_labels = [label for label in present if label != "off"]
    rare = np.concatenate([np.flatnonzero(strata == label) for label in rare_labels])
    if len(rare) >= max_rows:
        # Only relevant for very long lags: allocate the budget evenly so no
        # transition type is erased by an abundant active/mixed regime.
        allocations: list[np.ndarray] = []
        base = max_rows // max(1, len(rare_labels))
        for label in rare_labels:
            candidates = np.flatnonzero(strata == label)
            take = min(len(candidates), base)
            allocations.append(rng.choice(candidates, size=take, replace=False))
        chosen = np.concatenate(allocations)
        remaining = max_rows - len(chosen)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
            chosen = np.concatenate([chosen, rng.choice(pool, size=remaining, replace=False)])
    else:
        abundant = np.flatnonzero(strata == "off")
        remaining = min(max_rows - len(rare), len(abundant))
        chosen = np.concatenate(
            [rare, rng.choice(abundant, size=remaining, replace=False)]
        )
    return np.sort(np.unique(chosen.astype(np.int64)))
