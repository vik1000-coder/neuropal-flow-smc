from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray
VAR_MECHANISMS = (
    "nonlinear_location",
    "state_dependent_variance",
    "state_dependent_covariance",
    "heavy_tailed_innovations",
    "switching_regimes",
    "multimodal_future",
    "hidden_global_state",
    "calcium_observation_filter",
)


def _spectral_normalize(matrix: Array, radius: float) -> Array:
    value = max(abs(np.linalg.eigvals(matrix)))
    return np.asarray(matrix * (radius / max(float(value), 1e-8)), dtype=np.float64)


@dataclass
class SyntheticDataset:
    kind: str
    mechanism: str
    generator_seed: int
    train_h: Array
    train_y: Array
    validation_h: Array
    validation_y: Array
    test_h: Array
    test_y: Array
    anchor_h: Array
    anchor_state: Array
    stimulus: float
    source: int
    target: int
    stability_radius: float
    metadata: dict
    oracle_transition: callable
    oracle_conditioned: callable
    pulse_truth: callable | None = None


class NonlinearVAR:
    def __init__(self, seed: int, mechanism: str, dimension: int = 6):
        if mechanism not in VAR_MECHANISMS:
            raise ValueError(f"unknown VAR mechanism {mechanism}")
        self.seed = int(seed)
        self.mechanism = mechanism
        self.dimension = int(dimension)
        rng = np.random.default_rng(seed)
        self.A = _spectral_normalize(rng.normal(scale=0.25, size=(dimension, dimension)), 0.62)
        self.B = _spectral_normalize(rng.normal(scale=0.18, size=(dimension, dimension)), 0.28)
        self.G = rng.normal(scale=0.18, size=dimension)
        self.G[:2] += np.array([0.35, -0.20])
        self.factor = rng.normal(scale=0.12, size=(dimension, 2))
        self.hidden_loading = rng.normal(scale=0.12, size=dimension)
        self.observation_decay = 0.72

    @property
    def spectral_radius(self) -> float:
        return float(max(abs(np.linalg.eigvals(self.A))))

    def transition(
        self,
        state: Array,
        stimulus: Array,
        rng: np.random.Generator,
        *,
        hidden: Array | None = None,
    ) -> Array:
        x = np.asarray(state, dtype=np.float64)
        one = x.ndim == 1
        if one:
            x = x[None]
        s = np.broadcast_to(np.asarray(stimulus, dtype=np.float64), (len(x),))
        mean = x @ self.A.T + 0.24 * np.tanh(x @ self.B.T) + s[:, None] * self.G
        if self.mechanism == "nonlinear_location":
            mean[:, 1] += 0.18 * x[:, 0] * x[:, 2]
        if self.mechanism == "switching_regimes":
            mean += (x[:, :1] > 0).astype(float) * 0.15 * np.tanh(x[:, ::-1])
        scale = np.full_like(x, 0.16)
        if self.mechanism == "state_dependent_variance":
            scale *= 0.65 + 0.75 / (1 + np.exp(-x))
        noise = rng.normal(size=x.shape) * scale
        if self.mechanism == "state_dependent_covariance":
            rank_noise = rng.normal(size=(len(x), 2))
            noise += rank_noise @ (self.factor * (0.6 + 0.35 * np.tanh(x.mean(0))[:, None])).T
        elif self.mechanism == "heavy_tailed_innovations":
            noise *= np.sqrt(3.0 / rng.chisquare(5.0, size=(len(x), 1)))
        elif self.mechanism == "multimodal_future":
            occupancy = rng.random(len(x)) < 1 / (1 + np.exp(-1.5 * x[:, 0]))
            noise += np.where(occupancy[:, None], 0.22, -0.22) * np.r_[1.0, 0.8, np.zeros(self.dimension - 2)]
        elif self.mechanism == "hidden_global_state":
            if hidden is None:
                hidden = rng.normal(size=len(x))
            noise += np.asarray(hidden)[:, None] * self.hidden_loading
        result = mean + noise
        if self.mechanism == "calcium_observation_filter":
            result = self.observation_decay * x + (1 - self.observation_decay) * result
        return result[0] if one else result

    def dataset(self, n_total: int = 4200) -> SyntheticDataset:
        rng = np.random.default_rng(self.seed + 11_003)
        state = np.zeros(self.dimension)
        rows_h, rows_y, states, stimuli = [], [], [], []
        for time in range(n_total + 300):
            phase = time % 160
            stimulus = 1.0 if 40 <= phase < 64 else 0.0
            hidden = np.array([0.88 * math.sin(time / 37.0) + rng.normal(scale=0.25)])
            next_state = self.transition(state, stimulus, rng, hidden=hidden)
            if time >= 300:
                rows_h.append(np.r_[state, stimulus])
                rows_y.append(next_state - state)
                states.append(state.copy())
                stimuli.append(stimulus)
            state = next_state
        h = np.asarray(rows_h, dtype=np.float32)
        y = np.asarray(rows_y, dtype=np.float32)
        states_array = np.asarray(states, dtype=np.float64)
        train_end = int(0.65 * len(h))
        validation_end = int(0.80 * len(h))
        anchor_index = validation_end + int(0.45 * (len(h) - validation_end))
        anchor_h = h[anchor_index].copy()
        anchor_state = states_array[anchor_index].copy()
        anchor_stimulus = float(stimuli[anchor_index])

        def oracle(n: int, seed: int) -> Array:
            draw = self.transition(
                np.broadcast_to(anchor_state, (n, self.dimension)),
                np.full(n, anchor_stimulus),
                np.random.default_rng(seed),
            )
            return draw - anchor_state

        def oracle_at(condition: Array, n: int, seed: int) -> Array:
            condition = np.asarray(condition, dtype=np.float64)
            state0 = condition[: self.dimension]
            stimulus0 = float(condition[self.dimension])
            draw = self.transition(
                np.broadcast_to(state0, (n, self.dimension)),
                np.full(n, stimulus0),
                np.random.default_rng(seed),
            )
            return draw - state0

        return SyntheticDataset(
            kind="nonlinear_var",
            mechanism=self.mechanism,
            generator_seed=self.seed,
            train_h=h[:train_end],
            train_y=y[:train_end],
            validation_h=h[train_end:validation_end],
            validation_y=y[train_end:validation_end],
            test_h=h[validation_end:],
            test_y=y[validation_end:],
            anchor_h=anchor_h,
            anchor_state=anchor_state,
            stimulus=anchor_stimulus,
            source=0,
            target=1,
            stability_radius=self.spectral_radius,
            metadata={
                "A": self.A.tolist(),
                "B": self.B.tolist(),
                "G": self.G.tolist(),
                "factor": self.factor.tolist(),
                "training_support_source_min": float(h[:train_end, 0].min()),
                "training_support_source_max": float(h[:train_end, 0].max()),
            },
            oracle_transition=oracle,
            oracle_conditioned=oracle_at,
        )


class FitzHughNagumoNetwork:
    def __init__(self, seed: int, nodes: int = 8, dt: float = 0.02, sample_dt: float = 0.25):
        self.seed = int(seed)
        self.nodes = int(nodes)
        self.dt = float(dt)
        self.steps_per_frame = int(round(sample_dt / dt))
        self.effective_sample_dt = self.steps_per_frame * self.dt
        rng = np.random.default_rng(seed)
        graph = rng.normal(scale=0.06, size=(nodes, nodes))
        graph[rng.random((nodes, nodes)) > 0.28] = 0
        np.fill_diagonal(graph, 0)
        self.graph = graph
        self.a = rng.uniform(0.55, 0.78, size=nodes)
        self.b = rng.uniform(0.65, 0.9, size=nodes)
        self.epsilon = rng.uniform(0.05, 0.09, size=nodes)
        self.baseline = rng.uniform(0.18, 0.34, size=nodes)
        self.sigma_v = 0.035
        self.sigma_w = 0.012
        self.calcium_decay = 0.88
        self.strongest_target, self.strongest_source = np.unravel_index(
            np.argmax(np.abs(self.graph)), self.graph.shape
        )

    def _advance(
        self,
        v: Array,
        w: Array,
        calcium: Array,
        stimulus: Array,
        rng: np.random.Generator,
        *,
        pulse_node: int | None = None,
        pulse_amplitude: float = 0.0,
        supplied_noise: tuple[Array, Array] | None = None,
    ) -> tuple[Array, Array, Array]:
        v = np.asarray(v, dtype=np.float64).copy()
        w = np.asarray(w, dtype=np.float64).copy()
        calcium = np.asarray(calcium, dtype=np.float64).copy()
        one = v.ndim == 1
        if one:
            v, w, calcium = v[None], w[None], calcium[None]
        count = len(v)
        stim = np.broadcast_to(np.asarray(stimulus, dtype=float), (count,))
        for step in range(self.steps_per_frame):
            coupling = np.tanh(v) @ self.graph.T
            current = self.baseline[None] + stim[:, None] * np.r_[0.25, -0.12, np.zeros(self.nodes - 2)]
            if pulse_node is not None:
                current[:, pulse_node] += pulse_amplitude
            dv = v - v**3 / 3 - w + current + coupling
            dw = self.epsilon[None] * (v + self.a[None] - self.b[None] * w)
            if supplied_noise is None:
                zv = rng.normal(size=v.shape)
                zw = rng.normal(size=w.shape)
            else:
                zv, zw = supplied_noise[0][step], supplied_noise[1][step]
            v += self.dt * dv + self.sigma_v * np.sqrt(self.dt) * zv
            w += self.dt * dw + self.sigma_w * np.sqrt(self.dt) * zw
            step_decay = self.calcium_decay ** (1.0 / self.steps_per_frame)
            calcium = step_decay * calcium + (1 - step_decay) * np.logaddexp(0.0, v)
        if one:
            return v[0], w[0], calcium[0]
        return v, w, calcium

    def dataset(self, n_total: int = 3600) -> SyntheticDataset:
        rng = np.random.default_rng(self.seed + 71_009)
        v = rng.normal(scale=0.1, size=self.nodes)
        w = rng.normal(scale=0.05, size=self.nodes)
        calcium = np.logaddexp(0.0, v)
        rows_h, rows_y, latent, stimuli, pulses = [], [], [], [], []
        for time in range(n_total + 240):
            phase = time % 240
            stimulus = 0.8 if 70 <= phase < 95 else 0.0
            pulse = 0.3 * math.sin((time // 40 + self.seed) * 1.7) if 140 <= phase < 144 else 0.0
            nv, nw, nc = self._advance(
                v, w, calcium, stimulus, rng,
                pulse_node=int(self.strongest_source), pulse_amplitude=pulse,
            )
            if time >= 240:
                rows_h.append(np.r_[calcium, stimulus, pulse])
                rows_y.append(nc - calcium)
                latent.append(np.r_[v, w, calcium])
                stimuli.append(stimulus)
                pulses.append(pulse)
            v, w, calcium = nv, nw, nc
        h = np.asarray(rows_h, dtype=np.float32)
        y = np.asarray(rows_y, dtype=np.float32)
        latent = np.asarray(latent, dtype=np.float64)
        train_end = int(0.65 * len(h))
        validation_end = int(0.80 * len(h))
        candidates = np.flatnonzero(
            (np.arange(len(h)) >= validation_end) & (np.abs(np.asarray(pulses)) < 1e-12)
        )
        anchor_index = int(candidates[len(candidates) // 3])
        anchor = latent[anchor_index].copy()
        anchor_stimulus = float(stimuli[anchor_index])

        def oracle(n: int, seed: int) -> Array:
            v0, w0, c0 = np.split(anchor, [self.nodes, 2 * self.nodes])
            _, _, next_calcium = self._advance(
                np.broadcast_to(v0, (n, self.nodes)),
                np.broadcast_to(w0, (n, self.nodes)),
                np.broadcast_to(c0, (n, self.nodes)),
                np.full(n, anchor_stimulus),
                np.random.default_rng(seed),
            )
            return next_calcium - c0

        def oracle_at(condition: Array, n: int, seed: int) -> Array:
            condition = np.asarray(condition, dtype=np.float64)
            v0, w0, c0 = np.split(anchor, [self.nodes, 2 * self.nodes])
            next_v, next_w, next_calcium = self._advance(
                np.broadcast_to(v0, (n, self.nodes)),
                np.broadcast_to(w0, (n, self.nodes)),
                np.broadcast_to(c0, (n, self.nodes)),
                np.full(n, float(condition[-2])),
                np.random.default_rng(seed),
                pulse_node=int(self.strongest_source),
                pulse_amplitude=float(condition[-1]),
            )
            del next_v, next_w
            return next_calcium - c0

        def pulse_truth(amplitude: float, horizon: int, n: int, seed: int) -> Array:
            v0, w0, c0 = np.split(anchor, [self.nodes, 2 * self.nodes])
            v_base = np.broadcast_to(v0, (n, self.nodes)).copy()
            w_base = np.broadcast_to(w0, (n, self.nodes)).copy()
            c_base = np.broadcast_to(c0, (n, self.nodes)).copy()
            v_pulse, w_pulse, c_pulse = v_base.copy(), w_base.copy(), c_base.copy()
            rng = np.random.default_rng(seed)
            for frame in range(horizon):
                noises = (
                    rng.normal(size=(self.steps_per_frame, n, self.nodes)),
                    rng.normal(size=(self.steps_per_frame, n, self.nodes)),
                )
                v_base, w_base, c_base = self._advance(
                    v_base, w_base, c_base, anchor_stimulus,
                    np.random.default_rng(seed + frame), supplied_noise=noises,
                )
                v_pulse, w_pulse, c_pulse = self._advance(
                    v_pulse, w_pulse, c_pulse, anchor_stimulus,
                    np.random.default_rng(seed + frame), pulse_node=int(self.strongest_source),
                    pulse_amplitude=amplitude if frame == 0 else 0.0,
                    supplied_noise=noises,
                )
            return (c_pulse - c_base).mean(axis=0)

        return SyntheticDataset(
            kind="fitzhugh_nagumo",
            mechanism="coupled_fhn_calcium",
            generator_seed=self.seed,
            train_h=h[:train_end],
            train_y=y[:train_end],
            validation_h=h[train_end:validation_end],
            validation_y=y[train_end:validation_end],
            test_h=h[validation_end:],
            test_y=y[validation_end:],
            anchor_h=h[anchor_index].copy(),
            anchor_state=anchor,
            stimulus=anchor_stimulus,
            source=int(self.strongest_source),
            target=int(self.strongest_target),
            stability_radius=float("nan"),
            metadata={
                "graph": self.graph.tolist(),
                "a": self.a.tolist(),
                "b": self.b.tolist(),
                "epsilon": self.epsilon.tolist(),
                "dt": self.dt,
                "steps_per_frame": self.steps_per_frame,
                "effective_sample_dt": self.effective_sample_dt,
                "sigma_v": self.sigma_v,
                "sigma_w": self.sigma_w,
                "training_stimulus_max": 0.8,
                "training_source_pulse_max": float(np.max(np.abs(pulses[:train_end]))),
            },
            oracle_transition=oracle,
            oracle_conditioned=oracle_at,
            pulse_truth=pulse_truth,
        )


def fhn_solver_error(seed: int, duration_frames: int = 20) -> float:
    coarse = FitzHughNagumoNetwork(seed, dt=0.02)
    fine = FitzHughNagumoNetwork(seed, dt=0.01)
    rng = np.random.default_rng(seed + 99)
    v = rng.normal(scale=0.1, size=coarse.nodes)
    w = rng.normal(scale=0.05, size=coarse.nodes)
    c = np.logaddexp(0.0, v)
    vc, wc, cc = v.copy(), w.copy(), c.copy()
    vf, wf, cf = v.copy(), w.copy(), c.copy()
    # Deterministic convergence check isolates integration error from noise.
    coarse.sigma_v = coarse.sigma_w = 0.0
    fine.sigma_v = fine.sigma_w = 0.0
    for _ in range(duration_frames):
        vc, wc, cc = coarse._advance(vc, wc, cc, 0.0, np.random.default_rng(1))
        vf, wf, cf = fine._advance(vf, wf, cf, 0.0, np.random.default_rng(1))
    return float(np.max(np.abs(cc - cf)))
