import pandas as pd

from conditional_neural_benchmark.focused_world_model_runner import (
    _family,
    candidate_configs,
    choose_finalists,
)


def test_focused_grid_is_unique_and_finalists_preserve_family_coverage():
    configs = candidate_configs()
    assert len(configs) == 16
    assert len({config.model_id for config in configs}) == len(configs)
    board = pd.DataFrame(
        {
            "model_id": [config.model_id for config in configs],
            "energy__mean": list(range(len(configs))),
        }
    )
    selected = choose_finalists(board, configs, 4)
    assert len(selected) == 4
    assert len({config.model_id for config in selected}) == 4
    assert "contrastive_energy" in {_family(config) for config in selected}
    assert "exact_density" in {_family(config) for config in selected}
