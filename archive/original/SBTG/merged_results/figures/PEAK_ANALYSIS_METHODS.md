# Peak Lag Analysis: Methods & Results

To identify the characteristic timescales of synaptic, functional, and neuromodulatory networks, we analyzed the Model Performance ($F_1$ score) as a function of time lag.

## Methodology

We employed a two-step procedure to determine the optimal time lag ($t_{peak}$) for each network:

1.  **Discrete Peak Detection**:
    We first identified the sampled time lag $t_i$ that yielded the maximum observed $F_1$ score:
    $$ t*{max} = \arg\max*{t} F_1(t) $$
    This provides a coarse estimate constrained to the experimental sampling grid (e.g., 0.25s, 0.50s, ...).

2.  **Parabolic Interpolation (Sub-sample Refinement)**:
    To estimate the true biological peak between sampled points, we applied local parabolic interpolation. We fit a parabola to the discrete maximum $(t_{max}, y_{max})$ and its two immediate neighbors $(t_{max-1}, y_{prev})$ and $(t_{max+1}, y_{next})$.

    The interpolated peak time is calculated analytically from the vertex of this parabola:
    $$ t*{peak} = t*{max} + \frac{\Delta t}{2} \cdot \frac{y*{prev} - y*{next}}{y*{prev} - 2y*{max} + y\_{next}} $$
    where $\Delta t$ is the sampling interval. This method assumes the efficacy curve is locally smooth and concave around the peak.

    _Note_: For curves where the peak occurred at the boundary of the sampled range (e.g., Serotonin at 5.0s), interpolation was not performed, and the discrete peak was reported.

## Results

| Network Categories      | Discrete Peak (s) | **Interpolated Peak (s)** | Peak F1 Score |
| :---------------------- | :---------------- | :------------------------ | :------------ |
| **Synaptic Connectome** |                   |                           |               |
| Structural (Cook)       | 0.25              | **0.25**                  | 0.338         |
| Functional (Leifer)     | 2.00              | **1.88**                  | 0.695         |
| **Monoamine Networks**  |                   |                           |               |
| Tyramine                | 0.50              | **0.55**                  | 0.062         |
| Dopamine                | 0.75              | **0.82**                  | 0.020         |
| Octopamine              | 0.75              | **0.87**                  | 0.037         |
| Serotonin               | 5.00              | **5.00**                  | 0.069         |

### Interpretation

- **Fast Synaptic Dynamics**: Structural connectivity peaks immediately (~0.25s), consistent with fast neurotransmission.
- **Slow Functional Integration**: Functional connectivity peaks significantly later (~1.9s), reflecting the integration of multiple synaptic hops and slower network states.
- **Modulatory Timescales**:
  - **Fast Modulators**: Tyramine, Dopamine, and Octopamine effectively modulate activity on sub-second timescales (0.5s - 0.9s).
  - **Slow Modulators**: Serotonin acts on a much slower timescale (>5s), suggesting a role in persistent state regulation rather than rapid signaling.
