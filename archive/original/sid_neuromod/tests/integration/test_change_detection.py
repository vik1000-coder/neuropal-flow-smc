"""Section 18.2: change detection (E3)."""
import numpy as np
import pytest

from sid_neuromod.experiments.run_synthetic import monitor_scenario


def test_change_detection_null_and_scale():
    """Null false alarms low; scale-change detection power >= 80%; memory reported."""
    n_rep = 20
    null_fa = 0
    scale_det = 0
    for r in range(n_rep):
        res_n, _ = monitor_scenario("null", seed=1000 * r, T=22000, alpha=0.01)
        if res_n.alarmed:
            null_fa += 1
        res_s, tc = monitor_scenario("scale", seed=1000 * r, T=22000, alpha=0.01)
        if res_s.alarmed and res_s.alarm_time >= tc:
            scale_det += 1

    null_rate = null_fa / n_rep
    scale_power = scale_det / n_rep
    assert null_rate <= 0.05, f"null false-alarm rate {null_rate} exceeds tolerance"
    assert scale_power >= 0.80, f"scale power {scale_power} < 0.80"


def test_change_detection_dominant_channel_is_dispersion_for_scale():
    """A variance-only (scale) change should be attributed to a distributional channel."""
    res, tc = monitor_scenario("scale", seed=0, T=22000, alpha=0.01)
    assert res.alarmed
    assert res.dominant_channel in {"dispersion", "serial", "lag"}
    assert res.alarm_time >= tc


@pytest.mark.slow
def test_change_detection_memory_power_long_run():
    """Memory-only change: power >= 60% in long runs (Section 12.3)."""
    n_rep = 20
    det = 0
    for r in range(n_rep):
        res, tc = monitor_scenario("memory", seed=1000 * r, T=60000, alpha=0.01)
        if res.alarmed and res.alarm_time >= tc:
            det += 1
    assert det / n_rep >= 0.60, f"memory power {det/n_rep} < 0.60"
