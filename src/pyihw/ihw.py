"""Main IHW algorithm: k-fold cross-validation loop and public entry point."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyihw._types import IHWResult
from pyihw.splitting import assign_folds, groups_by_filter
from pyihw.utils import bh_adjust, safe_divide, weighted_storey_pi0
from pyihw.weighting import ihw_convex


def _ihw_internal(
    sorted_groups: NDArray[np.intp],
    sorted_pvalues: NDArray[np.float64],
    alpha: float,
    lambdas: NDArray[np.float64],
    m_groups: NDArray[np.intp],
    penalty: str,
    nfolds: int,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    null_proportion: bool,
    null_proportion_level: float,
    rng: np.random.Generator,
    sorted_folds: NDArray[np.intp] | None = None,
) -> dict[str, Any]:
    """Perform the IHW k-fold cross-validation loop.

    Parameters
    ----------
    sorted_groups : NDArray[np.intp]
        Bin (stratum) assignment for each hypothesis, sorted by p-value.
    sorted_pvalues : NDArray[np.float64]
        P-values sorted in ascending order.
    alpha : float
        Significance level.
    lambdas : NDArray[np.float64]
        Candidate regularisation parameters.
    m_groups : NDArray[np.intp]
        Number of hypotheses in each bin. May differ from observed counts
        when adjusting for the total number of tests.
    penalty : str
        Regularisation penalty type.
    nfolds : int
        Number of cross-validation folds.
    nfolds_internal : int
        Number of folds for the internal (nested) CV used for lambda selection.
    nsplits_internal : int
        Number of random fold splits for internal CV.
    adjustment_type : str
        ``"bh"`` or ``"bonferroni"``.
    null_proportion : bool
        Whether to apply weighted Storey null-proportion adjustment.
    null_proportion_level : float
        Threshold *tau* for the Storey estimator.
    rng : numpy.random.Generator
        Random number generator.
    sorted_folds : NDArray[np.intp] or None
        Pre-specified fold assignments. If ``None``, folds are assigned
        randomly.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``fold_lambdas``, ``rjs``, ``sorted_pvalues``,
        ``sorted_weighted_pvalues``, ``sorted_adj_p``, ``sorted_weights``,
        ``sorted_groups``, ``sorted_folds``, ``weight_matrix``.

    Notes
    -----
    Implements the core IHW cross-validation loop described in Ignatiadis
    et al. (2016). For each fold, weights are learned on held-out data
    and applied to the current fold, preventing overfitting from inflating
    the FDR.
    """
    n = len(sorted_pvalues)
    nbins = len(m_groups)

    # --- Fold assignment ------------------------------------------------------
    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = assign_folds(n, nfolds, rng)

    # --- Observed counts per group (the available data) -----------------------
    m_groups_available = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)

    # --- Storage --------------------------------------------------------------
    sorted_weights = np.full(n, np.nan)
    weight_matrix = np.full((nbins, nfolds), np.nan)
    fold_lambdas = np.full(nfolds, np.nan)

    # --- Main fold loop -------------------------------------------------------
    for fold_idx in range(nfolds):
        fold_mask = sorted_folds == fold_idx

        if not np.any(fold_mask):
            # No hypotheses in this fold — leave uniform weights
            weight_matrix[:, fold_idx] = 1.0
            continue
        train_mask = ~fold_mask

        if nfolds == 1:
            # Single fold: use all data as both train and holdout
            train_mask = np.ones(n, dtype=bool)
            fold_mask_for_weight = np.ones(n, dtype=bool)
        else:
            fold_mask_for_weight = fold_mask

        train_groups = sorted_groups[train_mask]
        train_pvalues = sorted_pvalues[train_mask]
        train_group_counts = np.bincount(train_groups, minlength=nbins).astype(np.intp)

        # --- Compute m_groups_holdout and m_groups_train ----------------------
        if nfolds == 1:
            # Single fold: all data is used as both train and holdout
            m_groups_holdout = m_groups.copy()
            m_groups_train = m_groups.copy()
        elif folds_prespecified:
            # With pre-specified folds, holdout counts are exact
            holdout_group_counts = np.bincount(
                sorted_groups[fold_mask], minlength=nbins
            ).astype(np.intp)
            m_groups_holdout = holdout_group_counts
            m_groups_train = m_groups - m_groups_holdout
        else:
            # R formula for random folds:
            # m_groups_holdout = (m_groups - m_groups_available) / nfolds
            #                    + m_groups_available - train_group_counts
            m_groups_holdout = (
                (m_groups - m_groups_available) / nfolds
                + m_groups_available
                - train_group_counts
            ).astype(np.intp)
            m_groups_train = (m_groups - m_groups_holdout).astype(np.intp)

        # Ensure non-negative
        m_groups_holdout = np.maximum(m_groups_holdout, 0)
        m_groups_train = np.maximum(m_groups_train, 0)

        # --- Split training p-values by group, sorted -------------------------
        train_split_pvalues = _split_pvalues_by_group(
            train_pvalues, train_groups, nbins
        )

        # --- Lambda selection via nested CV -----------------------------------
        if len(lambdas) == 1:
            best_lambda = float(lambdas[0])
        else:
            best_lambda = _select_lambda(
                sorted_groups=train_groups,
                sorted_pvalues=train_pvalues,
                alpha=alpha,
                lambdas=lambdas,
                m_groups=m_groups_train,
                penalty=penalty,
                nfolds_internal=nfolds_internal,
                nsplits_internal=nsplits_internal,
                adjustment_type=adjustment_type,
                null_proportion=null_proportion,
                null_proportion_level=null_proportion_level,
                rng=rng,
            )

        fold_lambdas[fold_idx] = best_lambda

        # --- Solve for weights using ihw_convex -------------------------------
        ws = ihw_convex(
            split_sorted_pvalues=train_split_pvalues,
            alpha=alpha,
            m_groups=m_groups_holdout,
            m_groups_grenander=m_groups_train,
            penalty=penalty,
            lambda_=best_lambda,
            adjustment_type=adjustment_type,
        )

        # --- Optionally apply Storey null-proportion adjustment ---------------
        if null_proportion:
            holdout_pvalues = sorted_pvalues[fold_mask]
            holdout_groups = sorted_groups[fold_mask]
            holdout_weights_per_hyp = ws[holdout_groups]
            pi0_est = weighted_storey_pi0(
                holdout_pvalues,
                holdout_weights_per_hyp,
                tau=null_proportion_level,
                m=int(np.sum(m_groups_holdout)),
            )
            pi0_est = min(pi0_est, 1.0)
            if pi0_est > 0:
                ws = ws / pi0_est

        weight_matrix[:, fold_idx] = ws

        # --- Assign per-bin weights to hypotheses in this fold ----------------
        sorted_weights[fold_mask_for_weight] = ws[sorted_groups[fold_mask_for_weight]]

    # --- Compute weighted p-values and adjusted p-values ----------------------
    sorted_weighted_pvalues = safe_divide(sorted_pvalues, sorted_weights)

    if adjustment_type == "bh":
        sorted_adj_p = bh_adjust(sorted_weighted_pvalues, m_total=n)
    elif adjustment_type == "bonferroni":
        sorted_adj_p = np.minimum(sorted_weighted_pvalues * n, 1.0)
    else:
        raise ValueError(
            f"Unknown adjustment_type: {adjustment_type!r}. Use 'bh' or 'bonferroni'."
        )

    rjs = int(np.sum(sorted_adj_p <= alpha))

    return {
        "fold_lambdas": fold_lambdas,
        "rjs": rjs,
        "sorted_pvalues": sorted_pvalues,
        "sorted_weighted_pvalues": sorted_weighted_pvalues,
        "sorted_adj_p": sorted_adj_p,
        "sorted_weights": sorted_weights,
        "sorted_groups": sorted_groups,
        "sorted_folds": sorted_folds,
        "weight_matrix": weight_matrix,
    }


def _split_pvalues_by_group(
    pvalues: NDArray[np.float64],
    groups: NDArray[np.intp],
    nbins: int,
) -> list[NDArray[np.float64]]:
    """Split p-values by group and sort each group.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        P-values.
    groups : NDArray[np.intp]
        Group assignments.
    nbins : int
        Number of groups.

    Returns
    -------
    list[NDArray[np.float64]]
        Sorted p-values for each group.
    """
    result: list[NDArray[np.float64]] = []
    for g in range(nbins):
        mask = groups == g
        group_pvalues = pvalues[mask]
        result.append(np.sort(group_pvalues))
    return result


def _select_lambda(
    sorted_groups: NDArray[np.intp],
    sorted_pvalues: NDArray[np.float64],
    alpha: float,
    lambdas: NDArray[np.float64],
    m_groups: NDArray[np.intp],
    penalty: str,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    null_proportion: bool,
    null_proportion_level: float,
    rng: np.random.Generator,
) -> float:
    """Select the best regularisation parameter via nested cross-validation.

    Parameters
    ----------
    sorted_groups : NDArray[np.intp]
        Group assignments for the training data.
    sorted_pvalues : NDArray[np.float64]
        P-values for the training data (need not be globally sorted).
    alpha : float
        Significance level.
    lambdas : NDArray[np.float64]
        Candidate lambda values.
    m_groups : NDArray[np.intp]
        Number of hypotheses per group for the training set.
    penalty : str
        Regularisation penalty type.
    nfolds_internal : int
        Number of internal CV folds.
    nsplits_internal : int
        Number of random fold splits.
    adjustment_type : str
        ``"bh"`` or ``"bonferroni"``.
    null_proportion : bool
        Whether to apply Storey null-proportion adjustment.
    null_proportion_level : float
        Threshold for the Storey estimator.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        The lambda value that maximises the mean number of rejections
        across internal splits.
    """
    # Re-sort training data by p-value for internal calls
    order = np.argsort(sorted_pvalues)
    internal_sorted_pvalues = sorted_pvalues[order]
    internal_sorted_groups = sorted_groups[order]

    rjs_per_lambda = np.zeros(len(lambdas), dtype=np.float64)

    for _split in range(nsplits_internal):
        for lam_idx, lam in enumerate(lambdas):
            result = _ihw_internal(
                sorted_groups=internal_sorted_groups,
                sorted_pvalues=internal_sorted_pvalues,
                alpha=alpha,
                lambdas=np.array([lam]),
                m_groups=m_groups,
                penalty=penalty,
                nfolds=nfolds_internal,
                nfolds_internal=1,
                nsplits_internal=1,
                adjustment_type=adjustment_type,
                null_proportion=null_proportion,
                null_proportion_level=null_proportion_level,
                rng=rng,
            )
            rjs_per_lambda[lam_idx] += result["rjs"]

    # Average across splits
    rjs_per_lambda /= nsplits_internal

    # Pick lambda with most rejections (first one in case of tie)
    best_idx = int(np.argmax(rjs_per_lambda))
    return float(lambdas[best_idx])


def ihw(
    pvalues: NDArray[np.float64],
    covariates: NDArray[np.float64],
    alpha: float,
    *,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas: NDArray[np.float64] | str = "auto",
    adjustment_type: str = "bh",
    null_proportion: bool = False,
    null_proportion_level: float = 0.5,
    folds: NDArray[np.intp] | None = None,
    rng: np.random.Generator | None = None,
) -> IHWResult:
    """Apply Independent Hypothesis Weighting to a set of p-values.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Vector of p-values, one per hypothesis.
    covariates : NDArray[np.float64]
        Independent covariate values, one per hypothesis.
    alpha : float
        Target significance level (FDR or FWER).
    covariate_type : str
        ``"ordinal"`` (default) or ``"nominal"``. Determines the penalty:
        ordinal uses total variation, nominal uses uniform deviation.
    nbins : int or str
        Number of covariate bins. ``"auto"`` (default) sets
        ``max(1, min(40, n // 1500))``.
    nfolds : int
        Number of cross-validation folds. Default 5.
    nfolds_internal : int
        Number of folds for nested CV lambda selection. Default 5.
    nsplits_internal : int
        Number of random splits for internal CV. Default 1.
    lambdas : NDArray[np.float64] or str
        Candidate regularisation parameters. ``"auto"`` (default) generates
        a data-driven grid.
    adjustment_type : str
        ``"bh"`` (default) for Benjamini-Hochberg or ``"bonferroni"`` for
        Bonferroni.
    null_proportion : bool
        Whether to apply weighted Storey null-proportion adjustment.
        Default ``False``.
    null_proportion_level : float
        Threshold *tau* for the Storey estimator. Default 0.5.
    folds : NDArray[np.intp] or None
        Pre-specified fold assignments. If ``None``, folds are assigned
        randomly.
    rng : numpy.random.Generator or None
        Random number generator. If ``None``, a default generator is created.

    Returns
    -------
    IHWResult
        Container with adjusted p-values, weights, and diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid (NaN p-values, out-of-range alpha, mismatched
        lengths, etc.).

    Notes
    -----
    Implements the IHW procedure of Ignatiadis et al. (2016), *Nature
    Methods*. The covariate is binned, weights are learned via k-fold
    cross-validation to prevent overfitting, and a weighted
    Benjamini-Hochberg (or Bonferroni) procedure is applied.
    """
    # --- Input validation -----------------------------------------------------
    pvalues = np.asarray(pvalues, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    if np.any(np.isnan(pvalues)):
        raise ValueError("p-values must not contain NaN values.")
    if np.any(np.isnan(covariates)):
        raise ValueError("Covariates must not contain NaN values.")
    if np.any(pvalues < 0.0) or np.any(pvalues > 1.0):
        raise ValueError("All p-values must be in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if len(pvalues) != len(covariates):
        raise ValueError(
            f"Length mismatch: {len(pvalues)} p-values vs {len(covariates)} covariates."
        )
    if len(pvalues) == 0:
        raise ValueError("Input arrays must not be empty.")
    if adjustment_type not in ("bh", "bonferroni"):
        raise ValueError(
            f"Unknown adjustment_type: {adjustment_type!r}. Use 'bh' or 'bonferroni'."
        )
    if covariate_type not in ("ordinal", "nominal"):
        raise ValueError(
            f"Unknown covariate_type: {covariate_type!r}. Use 'ordinal' or 'nominal'."
        )

    n = len(pvalues)

    # --- Defaults -------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    # Penalty from covariate type
    penalty = "total_variation" if covariate_type == "ordinal" else "uniform_deviation"

    # Auto nbins
    if isinstance(nbins, str):
        if nbins != "auto":
            raise ValueError(f"nbins must be an integer or 'auto', got {nbins!r}.")
        nbins = max(1, min(40, n // 1500))

    # Bin covariates
    groups = groups_by_filter(covariates, nbins=nbins, rng=rng)
    m_groups = np.bincount(groups, minlength=nbins).astype(np.intp)

    # Auto lambdas
    if isinstance(lambdas, str):
        if lambdas != "auto":
            raise ValueError(f"lambdas must be an array or 'auto', got {lambdas!r}.")
        lambdas = np.array(
            sorted(set([0.0, 1.0, nbins / 8, nbins / 4, nbins / 2, nbins, np.inf]))
        )

    # --- Single bin shortcut --------------------------------------------------
    if nbins == 1:
        nfolds = 1
        sorted_order = np.argsort(pvalues)
        sorted_pvalues = pvalues[sorted_order]
        sorted_groups = groups[sorted_order]

        # Uniform weights
        weights = np.ones(n, dtype=np.float64)
        weighted_pvalues = sorted_pvalues.copy()

        if adjustment_type == "bh":
            adj_p_sorted = bh_adjust(sorted_pvalues, m_total=n)
        else:
            adj_p_sorted = np.minimum(sorted_pvalues * n, 1.0)

        # Unsort back to original order
        inv_order = np.argsort(sorted_order)
        adj_pvalues = adj_p_sorted[inv_order]
        weighted_pvalues_out = weighted_pvalues[inv_order]

        fold_assignments = np.zeros(n, dtype=np.intp)
        weight_mat = np.ones((1, 1), dtype=np.float64)

        return IHWResult(
            pvalues=pvalues,
            adj_pvalues=adj_pvalues,
            weights=weights,
            weighted_pvalues=weighted_pvalues_out,
            covariates=covariates,
            groups=groups,
            folds=fold_assignments,
            weight_matrix=weight_mat,
            alpha=alpha,
            nbins=nbins,
            nfolds=nfolds,
            regularization_terms=np.array([0.0]),
            m_groups=m_groups,
            penalty=penalty,
            covariate_type=covariate_type,
            adjustment_type=adjustment_type,
        )

    # --- Sort by p-value ------------------------------------------------------
    sorted_order = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_order]
    sorted_groups = groups[sorted_order]

    # Pre-specified folds (sort them too)
    sorted_folds_arg: NDArray[np.intp] | None = None
    if folds is not None:
        folds = np.asarray(folds, dtype=np.intp)
        if len(folds) != n:
            raise ValueError(
                f"Length mismatch: {n} hypotheses vs {len(folds)} fold assignments."
            )
        sorted_folds_arg = folds[sorted_order]

    # --- Run internal IHW -----------------------------------------------------
    result = _ihw_internal(
        sorted_groups=sorted_groups,
        sorted_pvalues=sorted_pvalues,
        alpha=alpha,
        lambdas=lambdas,
        m_groups=m_groups,
        penalty=penalty,
        nfolds=nfolds,
        nfolds_internal=nfolds_internal,
        nsplits_internal=nsplits_internal,
        adjustment_type=adjustment_type,
        null_proportion=null_proportion,
        null_proportion_level=null_proportion_level,
        rng=rng,
        sorted_folds=sorted_folds_arg,
    )

    # --- Unsort back to original order ----------------------------------------
    inv_order = np.argsort(sorted_order)
    adj_pvalues = result["sorted_adj_p"][inv_order]
    weights = result["sorted_weights"][inv_order]
    weighted_pvalues = result["sorted_weighted_pvalues"][inv_order]
    out_groups = result["sorted_groups"][inv_order]
    out_folds = result["sorted_folds"][inv_order]

    return IHWResult(
        pvalues=pvalues,
        adj_pvalues=adj_pvalues,
        weights=weights,
        weighted_pvalues=weighted_pvalues,
        covariates=covariates,
        groups=out_groups,
        folds=out_folds,
        weight_matrix=result["weight_matrix"],
        alpha=alpha,
        nbins=nbins,
        nfolds=nfolds,
        regularization_terms=result["fold_lambdas"],
        m_groups=m_groups,
        penalty=penalty,
        covariate_type=covariate_type,
        adjustment_type=adjustment_type,
    )
