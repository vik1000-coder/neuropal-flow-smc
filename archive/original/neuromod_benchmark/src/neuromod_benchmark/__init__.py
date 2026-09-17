"""Mechanism-first recovery benchmark for stochastic neuromodulated dynamics."""

from .dgp import simulate_dataset
from .mechanistic import MechanisticConfig
from .mechanistic_dataset import simulate_mechanistic_dataset
from .schema import BenchmarkConfig, DGPConfig, ResourceBudget

__all__ = [
    "BenchmarkConfig",
    "DGPConfig",
    "MechanisticConfig",
    "ResourceBudget",
    "simulate_dataset",
    "simulate_mechanistic_dataset",
]
__version__ = "0.1.0"
