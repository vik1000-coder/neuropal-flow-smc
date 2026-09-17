from pathlib import Path

import pytest
import yaml

from neuromod_benchmark.capabilities import validate_channel_contract
from neuromod_benchmark.dgp import simulate_dataset
from neuromod_benchmark.external_causal import DEFAULT_WORKER_PYTHON
from neuromod_benchmark.features import build_supervised, grouped_split, subset
from neuromod_benchmark.registry import make_method
from neuromod_benchmark.schema import DGPConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep this list explicit: adding a method to any finalized frozen config must also
# add a fitted contract check here.  Aliases matter because their fixed constructor
# settings can select different channel-producing code paths.
FROZEN_ALIASES = (
    "conditional_covariance_ridge",
    "conditional_granger_ridge",
    "conditional_schrodinger_bridge_d015",
    "conditional_schrodinger_bridge_d030",
    "conditional_schrodinger_bridge_d060",
    "constant_gaussian",
    "gaussian_mlp_dsm_s025",
    "gaussian_mlp_dsm_s050",
    "gaussian_mlp_nll",
    "gaussian_nll",
    "heteroskedastic_ridge",
    "lagged_correlation",
    "lagged_sparse_transition",
    "latent_neuromodulated_mixture_ssm",
    "latent_neuromodulated_ssm",
    "mdn_k3",
    "mdn_k7",
    "pcmci_parcorr",
    "pysindy_discrete_official",
    "ridge_full_covariance",
    "ridge_var",
    "sbtg_feature_bilinear_s025",
    "sbtg_feature_bilinear_s050",
    "sbtg_feature_student_s025",
    "sbtg_feature_student_s050",
    "sbtg_linear_s025",
    "score_mlp_gaussian_s025",
    "score_mlp_gaussian_s050",
    "score_mlp_student_s025",
    "score_mlp_student_s050",
    "sid_dsm_s075",
    "sid_dsm_s100",
    "sid_hyvarinen",
    "sindy_discrete",
    "student_t_ridge",
    "var_lingam_official",
)

OFFICIAL_CAUSAL_ALIASES = {
    "pcmci_parcorr",
    "pysindy_discrete_official",
    "var_lingam_official",
}


def _configured_frozen_aliases() -> set[str]:
    aliases: set[str] = set()
    config_root = PROJECT_ROOT / "configs"
    final_directories = sorted(
        path
        for path in config_root.glob("frozen_v[0-9]*")
        if path.is_dir() and not path.name.endswith("_draft")
    )
    assert final_directories, "no finalized frozen config directory found"
    for directory in final_directories:
        for path in directory.glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            aliases.update(method["name"] for method in payload.get("methods", ()))
    return aliases


def _resource_reduced_parameters(alias: str) -> dict:
    """Reduce fit cost without changing an alias's semantic channel path."""

    params: dict = {}
    if alias.startswith(("gaussian_mlp_", "mdn_", "score_mlp_")):
        params.update(
            hidden=8,
            layers=1,
            max_epochs=1,
            patience=1,
            batch_size=512,
            dropout=0.0,
        )
    if alias.startswith("sbtg_"):
        params.update(
            hidden=8,
            layers=1,
            feature_dim=4,
            max_epochs=1,
            patience=1,
            batch_size=512,
            dropout=0.0,
        )
    if alias.startswith("latent_neuromodulated_"):
        params.update(
            n_modulators=1,
            max_epochs=1,
            patience=1,
            truncation=16,
            warmup=2,
        )
    if alias == "gaussian_nll":
        params.update(maxiter=60, ridge=0.1)
    if alias == "pysindy_discrete_official":
        params.update(degree=1, max_iter=2, threshold=0.05, alpha=0.001)
    if alias == "sindy_discrete":
        params.update(degree=1, max_iter=2, threshold=0.02, ridge=0.001)
    return params


@pytest.fixture(scope="module")
def tiny_fitted_contract_parts():
    dataset = simulate_dataset(
        DGPConfig(
            n_neurons=3,
            n_modulators=1,
            n_trajectories=6,
            n_steps=70,
            burn_in=8,
            mechanism="mixed",
            seed=123,
        )
    )
    data = build_supervised(
        dataset,
        view="complete_state",
        history_lags=(1,),
        horizon=1,
    )
    masks = grouped_split(
        data.groups,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    ).masks(data)
    train, validation, _ = [subset(data, mask) for mask in masks]
    return train, validation


def test_fitted_contract_smoke_covers_every_final_frozen_alias() -> None:
    assert len(FROZEN_ALIASES) == 36
    assert set(FROZEN_ALIASES) == _configured_frozen_aliases()


@pytest.mark.parametrize("alias", FROZEN_ALIASES)
def test_frozen_alias_emits_only_declared_semantic_channels(
    alias: str,
    tiny_fitted_contract_parts,
) -> None:
    if alias in OFFICIAL_CAUSAL_ALIASES and not Path(DEFAULT_WORKER_PYTHON).exists():
        pytest.skip("optional causal environment not installed")
    train, validation = tiny_fitted_contract_parts
    model = make_method(alias, _resource_reduced_parameters(alias), seed=17)
    model.fit(train, validation)
    assert model.benchmark_id == alias
    validate_channel_contract(model.capabilities, getattr(model, "channels_", {}))
