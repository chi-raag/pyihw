from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from pyihw.utils import grenander_estimator


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


def ihw_convex(
    split_sorted_pvalues: list[NDArray[np.float64]],
    alpha: float,
    m_groups: NDArray[np.intp],
    m_groups_grenander: NDArray[np.intp],
    penalty: str,
    lambda_: float,
    adjustment_type: str,
) -> NDArray[np.float64]:
    """Solve the IHW convex relaxation LP to find optimal per-bin weights.

    Given sorted p-values within each covariate bin, this function formulates
    and solves a linear program that maximises the expected number of
    rejections subject to the FDR (BH) or FWER (Bonferroni) constraint, the
    Grenander-based CDF envelope constraints, and an optional regularisation
    penalty on the weight function.

    Parameters
    ----------
    split_sorted_pvalues : list[NDArray[np.float64]]
        Sorted p-values for each covariate bin.
    alpha : float
        Significance level for the multiple testing procedure.
    m_groups : NDArray[np.intp]
        Number of hypotheses per bin used for the weight budget.
    m_groups_grenander : NDArray[np.intp]
        Number of hypotheses per bin used for the Grenander estimator
        (typically equal to ``m_groups``).
    penalty : str
        Regularisation penalty type: ``"total_variation"`` or
        ``"uniform_deviation"``.
    lambda_ : float
        Regularisation strength. Use 0 for uniform weights, ``np.inf``
        for no regularisation (only the FDR/FWER constraint).
    adjustment_type : str
        ``"bh"`` for Benjamini-Hochberg FDR control or ``"bonferroni"``
        for Bonferroni FWER control.

    Returns
    -------
    NDArray[np.float64]
        Per-bin weights satisfying the weight constraint
        (weighted mean equals 1).

    Notes
    -----
    The LP variables are ``[y_1 .. y_G, t_1 .. t_G]`` where ``y_g``
    approximates the CDF at the threshold and ``t_g`` is the threshold
    for bin *g*.  The Grenander estimator provides a piecewise-linear
    concave majorant of the ECDF, yielding linear constraints on
    ``(y_g, t_g)`` pairs.

    When ``lambda_ < inf``, auxiliary variables are introduced for the
    absolute-value terms of the chosen penalty, and a budget constraint
    links the penalty to the sum of weighted thresholds.

    Implements the algorithm described in Ignatiadis et al. (2016),
    *Nature Methods*, following the R ``IHW`` package's
    ``ihw_convex`` function.
    """
    nbins = len(split_sorted_pvalues)

    # --- Early return for zero regularisation --------------------------------
    if lambda_ == 0.0:
        return np.ones(nbins)

    # --- Numerical stability: clip tiny p-values to 0 -----------------------
    split_sorted_pvalues = [np.where(p > 1e-20, p, 0.0) for p in split_sorted_pvalues]

    m = int(np.sum(m_groups))

    # --- Grenander estimator per bin -----------------------------------------
    grenander_list = [
        grenander_estimator(pv, int(mg))
        for pv, mg in zip(split_sorted_pvalues, m_groups_grenander)
    ]

    # --- Build LP in COO triplet form ----------------------------------------
    # Variables: y_1..y_G (indices 0..G-1), t_1..t_G (indices G..2G-1)
    # Then optional auxiliary variables for regularisation.

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs_list: list[float] = []
    row_idx = 0

    # Grenander constraints: y_g - slope_k * t_g <= y_knot_k - slope_k * x_knot_k
    for g, gr in enumerate(grenander_list):
        n_knots = len(gr.slopes)
        for k in range(n_knots):
            # y_g coefficient: +1
            rows.append(row_idx)
            cols.append(g)
            vals.append(1.0)
            # t_g coefficient: -slope_k
            rows.append(row_idx)
            cols.append(nbins + g)
            vals.append(-float(gr.slopes[k]))
            # RHS: y_knot_k - slope_k * x_knot_k
            rhs_list.append(
                float(gr.y_knots[k]) - float(gr.slopes[k]) * float(gr.x_knots[k])
            )
            row_idx += 1

    # Objective: maximize sum(m_g / m * nbins * y_g)
    # linprog minimises, so we negate.
    n_base_vars = 2 * nbins
    c_obj = np.zeros(n_base_vars)
    for g in range(nbins):
        c_obj[g] = -float(m_groups[g]) / m * nbins  # negate for minimisation

    # --- Regularisation auxiliary variables -----------------------------------
    n_aux = 0
    if lambda_ < np.inf:
        if penalty == "total_variation":
            n_aux = nbins - 1
            # Auxiliary vars f_1..f_{G-1} at indices 2G .. 2G + G - 2

            # Constraint set 1: t_{g+1} - t_g - f_g <= 0
            for g in range(nbins - 1):
                # t_{g+1}
                rows.append(row_idx)
                cols.append(nbins + g + 1)
                vals.append(1.0)
                # -t_g
                rows.append(row_idx)
                cols.append(nbins + g)
                vals.append(-1.0)
                # -f_g
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(-1.0)
                rhs_list.append(0.0)
                row_idx += 1

            # Constraint set 2: -t_{g+1} + t_g - f_g <= 0
            for g in range(nbins - 1):
                # -t_{g+1}
                rows.append(row_idx)
                cols.append(nbins + g + 1)
                vals.append(-1.0)
                # +t_g
                rows.append(row_idx)
                cols.append(nbins + g)
                vals.append(1.0)
                # -f_g
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(-1.0)
                rhs_list.append(0.0)
                row_idx += 1

            # TV budget constraint: sum(f_g) - lambda * sum(m_g/m * t_g) <= 0
            for g in range(nbins - 1):
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(1.0)
            for g in range(nbins):
                rows.append(row_idx)
                cols.append(nbins + g)
                vals.append(-lambda_ * float(m_groups[g]) / m)
            rhs_list.append(0.0)
            row_idx += 1

        elif penalty == "uniform_deviation":
            n_aux = nbins
            # Auxiliary vars f_1..f_G at indices 2G .. 3G - 1

            # Build the diff_matrix pattern from R:
            # diff_matrix[g, :] has m on diagonal, -m_groups elsewhere
            # i.e. row g: -m_groups for all columns, then add m on diagonal
            # So: coeff for t_h = -m_groups[h] for h != g, (m - m_groups[g]) for h == g
            # Which simplifies to: m * delta_{gh} - m_groups[h]

            # Constraint set 1: m*t_g - sum(m_i * t_i) - f_g <= 0
            for g in range(nbins):
                for h in range(nbins):
                    coeff = float(m) if h == g else 0.0
                    coeff -= float(m_groups[h])
                    if coeff != 0.0:
                        rows.append(row_idx)
                        cols.append(nbins + h)
                        vals.append(coeff)
                # -f_g
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(-1.0)
                rhs_list.append(0.0)
                row_idx += 1

            # Constraint set 2: -m*t_g + sum(m_i * t_i) - f_g <= 0
            for g in range(nbins):
                for h in range(nbins):
                    coeff = -float(m) if h == g else 0.0
                    coeff += float(m_groups[h])
                    if coeff != 0.0:
                        rows.append(row_idx)
                        cols.append(nbins + h)
                        vals.append(coeff)
                # -f_g
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(-1.0)
                rhs_list.append(0.0)
                row_idx += 1

            # UD budget constraint: sum(f_g) - lambda * sum(m_g * t_g) <= 0
            for g in range(nbins):
                rows.append(row_idx)
                cols.append(n_base_vars + g)
                vals.append(1.0)
            for g in range(nbins):
                rows.append(row_idx)
                cols.append(nbins + g)
                vals.append(-lambda_ * float(m_groups[g]))
            rhs_list.append(0.0)
            row_idx += 1

        else:
            raise ValueError(f"Unknown penalty: {penalty!r}")

    # --- FDR / FWER constraint -----------------------------------------------
    if adjustment_type == "bh":
        # sum(m_g * t_g) - alpha * sum(m_g * y_g) <= 0
        for g in range(nbins):
            # -alpha * m_g * y_g
            rows.append(row_idx)
            cols.append(g)
            vals.append(-alpha * float(m_groups[g]))
            # +m_g * t_g
            rows.append(row_idx)
            cols.append(nbins + g)
            vals.append(float(m_groups[g]))
        rhs_list.append(0.0)
        row_idx += 1
    elif adjustment_type == "bonferroni":
        # sum(m_g * t_g) <= alpha
        for g in range(nbins):
            rows.append(row_idx)
            cols.append(nbins + g)
            vals.append(float(m_groups[g]))
        rhs_list.append(alpha)
        row_idx += 1
    else:
        raise ValueError(
            f"Unknown adjustment_type: {adjustment_type!r}. Use 'bh' or 'bonferroni'."
        )

    # --- Assemble sparse matrix and solve ------------------------------------
    n_total_vars = n_base_vars + n_aux
    # Extend objective for auxiliary variables (zero cost)
    c_full = np.zeros(n_total_vars)
    c_full[:n_base_vars] = c_obj

    a_ub = csc_matrix(
        (vals, (rows, cols)),
        shape=(row_idx, n_total_vars),
    )
    b_ub = np.array(rhs_list)

    # Variable bounds: y_g in [0, 2], t_g in [0, 2], aux in [0, inf)
    bounds: list[tuple[float, float | None]] = []
    for _ in range(nbins):
        bounds.append((0.0, 2.0))  # y_g
    for _ in range(nbins):
        bounds.append((0.0, 2.0))  # t_g
    for _ in range(n_aux):
        bounds.append((0.0, None))  # f_g

    result = linprog(
        c=c_full,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return np.ones(nbins)

    # --- Extract thresholds and convert to weights ---------------------------
    ts = result.x[nbins : 2 * nbins]
    ts = np.maximum(ts, 0.0)  # clip negative rounding artefacts
    return thresholds_to_weights(ts, m_groups)
