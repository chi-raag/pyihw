from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass(frozen=True)
class GrenanderResult:
    """Result of the Grenander estimator (least concave majorant of ECDF).

    Parameters
    ----------
    x_knots : NDArray[np.float64]
        X-coordinates of the LCM knot points (left endpoints of segments).
    y_knots : NDArray[np.float64]
        Y-coordinates of the LCM knot points.
    slopes : NDArray[np.float64]
        Slope of each LCM segment. Slopes are non-increasing (concave).
    """

    x_knots: NDArray[np.float64]
    y_knots: NDArray[np.float64]
    slopes: NDArray[np.float64]


def grenander_estimator(
    sorted_pvalues: NDArray[np.float64],
    m_total: int,
) -> GrenanderResult:
    """Compute the Grenander estimator: least concave majorant of the ECDF.

    Parameters
    ----------
    sorted_pvalues : NDArray[np.float64]
        P-values sorted in ascending order within a single stratum.
    m_total : int
        Total number of hypotheses in this stratum.

    Returns
    -------
    GrenanderResult
        Knot coordinates and slopes of the piecewise-linear LCM.

    Notes
    -----
    Equivalent to ``fdrtool::gcmlcm(x, y, type="lcm")`` in R.
    """
    unique_pvalues, counts = np.unique(sorted_pvalues, return_counts=True)
    ecdf_values = np.cumsum(counts) / m_total

    if unique_pvalues[0] > 0:
        unique_pvalues = np.concatenate(([0.0], unique_pvalues))
        ecdf_values = np.concatenate(([0.0], ecdf_values))

    if unique_pvalues[-1] < 1.0 and ecdf_values[-1] < 1.0:
        unique_pvalues = np.concatenate((unique_pvalues, [1.0]))
        ecdf_values = np.concatenate((ecdf_values, [1.0]))

    # Upper convex hull for least concave majorant
    hull_x: list[float] = [float(unique_pvalues[0])]
    hull_y: list[float] = [float(ecdf_values[0])]

    for i in range(1, len(unique_pvalues)):
        while len(hull_x) >= 2:
            dx1 = hull_x[-1] - hull_x[-2]
            dy1 = hull_y[-1] - hull_y[-2]
            dx2 = float(unique_pvalues[i]) - hull_x[-1]
            dy2 = float(ecdf_values[i]) - hull_y[-1]
            if dy1 * dx2 <= dy2 * dx1:
                hull_x.pop()
                hull_y.pop()
            else:
                break
        hull_x.append(float(unique_pvalues[i]))
        hull_y.append(float(ecdf_values[i]))

    hull_x_arr = np.array(hull_x)
    hull_y_arr = np.array(hull_y)
    slopes = np.diff(hull_y_arr) / np.diff(hull_x_arr)

    return GrenanderResult(
        x_knots=hull_x_arr[:-1],
        y_knots=hull_y_arr[:-1],
        slopes=slopes,
    )


def weighted_storey_pi0(
    pvalues: NDArray[np.float64],
    weights: NDArray[np.float64],
    tau: float = 0.5,
    m: int | None = None,
) -> float:
    """Estimate the null proportion using a weighted Storey estimator.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        P-values.
    weights : NDArray[np.float64]
        Per-hypothesis weights.
    tau : float
        Threshold for the estimator. Defaults to 0.5.
    m : int, optional
        Total number of hypotheses. Defaults to ``len(pvalues)``.

    Returns
    -------
    float
        Estimated proportion of null hypotheses.
    """
    if m is None:
        m = len(pvalues)
    w_inf = float(np.max(weights))
    numerator = w_inf + float(np.sum(weights * (pvalues > tau)))
    return numerator / (m * (1.0 - tau))
