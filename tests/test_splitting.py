from __future__ import annotations

import numpy as np

from pyihw.splitting import assign_folds, groups_by_filter


def test_groups_by_filter_equal_sizes() -> None:
    rng = np.random.default_rng(42)
    covariates = rng.uniform(size=1000)
    groups = groups_by_filter(covariates, nbins=7, rng=rng)
    counts = np.bincount(groups)
    assert counts.max() - counts.min() <= 1


def test_groups_by_filter_correct_nbins() -> None:
    rng = np.random.default_rng(42)
    covariates = rng.uniform(size=100)
    groups = groups_by_filter(covariates, nbins=10, rng=rng)
    assert len(np.unique(groups)) == 10


def test_groups_by_filter_preserves_order() -> None:
    """Higher covariate values should map to higher group indices."""
    rng = np.random.default_rng(42)
    covariates = np.arange(100, dtype=np.float64)
    groups = groups_by_filter(covariates, nbins=5, rng=rng)
    assert np.all(groups[:20] == 0)
    assert np.all(groups[80:] == 4)


def test_groups_by_filter_deterministic() -> None:
    covariates = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    g1 = groups_by_filter(covariates, nbins=2, rng=np.random.default_rng(99))
    g2 = groups_by_filter(covariates, nbins=2, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(g1, g2)


def test_assign_folds_correct_shape() -> None:
    rng = np.random.default_rng(42)
    folds = assign_folds(1000, nfolds=5, rng=rng)
    assert folds.shape == (1000,)
    assert set(np.unique(folds)).issubset(set(range(5)))


def test_assign_folds_deterministic() -> None:
    f1 = assign_folds(100, nfolds=3, rng=np.random.default_rng(7))
    f2 = assign_folds(100, nfolds=3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(f1, f2)
