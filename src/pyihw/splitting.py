from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def groups_by_filter(
    covariates: NDArray[np.float64],
    nbins: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Stratify hypotheses into equal-size bins by covariate rank.

    Parameters
    ----------
    covariates : NDArray[np.float64]
        Numeric covariate values, one per hypothesis.
    nbins : int
        Number of bins to create.
    rng : numpy.random.Generator
        Random number generator for breaking ties.

    Returns
    -------
    NDArray[np.intp]
        0-indexed bin assignment for each hypothesis. Bins are ordered
        so that lower covariate values map to lower bin indices.

    Notes
    -----
    Ties in the covariate are broken randomly using *rng*. The resulting
    bins differ in size by at most 1.
    """
    n = len(covariates)
    jitter = rng.permutation(n).astype(np.float64) / (n + 1)
    order = np.lexsort((jitter, covariates))
    ranks = np.empty(n, dtype=np.intp)
    ranks[order] = np.arange(n)
    return (ranks * nbins) // n


def assign_folds(
    n: int,
    nfolds: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Randomly assign hypotheses to cross-validation folds.

    Parameters
    ----------
    n : int
        Number of hypotheses.
    nfolds : int
        Number of folds.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    NDArray[np.intp]
        0-indexed fold assignment for each hypothesis.
    """
    return rng.integers(0, nfolds, size=n)
