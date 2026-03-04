from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bh_threshold(
    pvalues: NDArray[np.float64],
    alpha: float,
    m_total: int | None = None,
) -> float:
    """Compute the Benjamini-Hochberg step-up rejection threshold.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Vector of p-values.
    alpha : float
        Significance level.
    m_total : int, optional
        Total number of tests. Defaults to ``len(pvalues)``.

    Returns
    -------
    float
        The BH threshold *t* such that hypotheses with ``p <= t`` are
        rejected. Returns 0.0 if no hypothesis is rejected.
    """
    m = len(pvalues)
    if m_total is None:
        m_total = m
    sorted_p = np.sort(pvalues)
    thresholds = np.arange(1, m + 1) / m_total * alpha
    rejected = np.where(sorted_p <= thresholds)[0]
    if len(rejected) == 0:
        return 0.0
    return float(sorted_p[rejected[-1]])


def bh_adjust(
    pvalues: NDArray[np.float64],
    m_total: int | None = None,
) -> NDArray[np.float64]:
    """Compute Benjamini-Hochberg adjusted p-values.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Vector of unadjusted p-values.
    m_total : int, optional
        Total number of tests. Defaults to ``len(pvalues)``.

    Returns
    -------
    NDArray[np.float64]
        BH-adjusted p-values, same length as input.

    Notes
    -----
    Equivalent to R's ``p.adjust(p, method="BH", n=m_total)``.
    """
    m = len(pvalues)
    if m_total is None:
        m_total = m
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    adjusted = sorted_p * m_total / np.arange(1, m + 1)
    np.minimum.accumulate(adjusted[::-1], out=adjusted[::-1])
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def safe_divide(
    pvalues: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted p-values ``pvalues / weights`` with safety guards.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Numerator p-values.
    weights : NDArray[np.float64]
        Denominator weights.

    Returns
    -------
    NDArray[np.float64]
        ``pvalues / weights``, where: zero numerators yield 0,
        zero denominators with nonzero numerators yield 1, and results
        are clipped to [0, 1].

    Notes
    -----
    Mirrors the ``mydiv`` helper in the R IHW package.
    """
    result = np.where(
        pvalues == 0.0,
        0.0,
        np.where(
            weights == 0.0,
            1.0,
            np.minimum(pvalues / np.where(weights == 0.0, 1.0, weights), 1.0),
        ),
    )
    return result
