from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def wasserman_sim(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Wasserman simulation: 10000 hypotheses, pi0=0.85, covariate in [0, 3]."""
    m = 10000
    covariates = rng.uniform(0, 3, size=m)
    signals = rng.binomial(1, 0.15, size=m)
    z = rng.normal(loc=signals * covariates)
    pvalues = 1 - norm.cdf(z)
    return pvalues, covariates
