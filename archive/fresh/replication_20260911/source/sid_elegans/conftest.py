import warnings

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "smoke: end-to-end smoke tests")
    warnings.filterwarnings("ignore", message="invalid eta2", category=RuntimeWarning)
