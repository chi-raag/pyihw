from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def thresholds_to_weights(
    thresholds: NDArray[np.float64],
    m_groups: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Convert per-bin thresholds to weights satisfying the weight constraint.

    Parameters
    ----------
    thresholds : NDArray[np.float64]
        Rejection threshold for each bin.
    m_groups : NDArray[np.intp]
        Number of hypotheses in each bin.

    Returns
    -------
    NDArray[np.float64]
        Weights ``w_g`` such that the weighted mean equals 1:
        ``sum(w_g * m_g) / sum(m_g) == 1``.

    Notes
    -----
    Formula: ``w_g = t_g * m / sum(m_g * t_g)`` where ``m = sum(m_g)``.
    If all thresholds are zero, returns uniform weights of 1.
    """
    if len(thresholds) != len(m_groups):
        raise ValueError(
            f"Length mismatch: {len(thresholds)} thresholds vs {len(m_groups)} groups"
        )
    if np.all(thresholds == 0.0):
        return np.ones(len(thresholds))
    m = np.sum(m_groups)
    return thresholds * m / np.sum(m_groups * thresholds)


def total_variation(weights: NDArray[np.float64]) -> float:
    """Total variation of a weight vector: ``sum(|w_{g+1} - w_g|)``.

    Parameters
    ----------
    weights : NDArray[np.float64]
        Weight vector.

    Returns
    -------
    float
        Sum of absolute consecutive differences.

    Notes
    -----
    This penalty encourages smoothness in the weight function across
    ordered covariate bins. It equals zero when all weights are identical.
    """
    return float(np.sum(np.abs(np.diff(weights))))


def uniform_deviation(weights: NDArray[np.float64]) -> float:
    """Uniform deviation of a weight vector: ``sum(|w_g - 1|)``.

    Parameters
    ----------
    weights : NDArray[np.float64]
        Weight vector.

    Returns
    -------
    float
        Sum of absolute deviations from 1.

    Notes
    -----
    This penalty discourages weights that stray far from the uniform
    baseline of 1, acting as a regulariser toward the unweighted BH
    procedure.
    """
    return float(np.sum(np.abs(weights - 1.0)))
