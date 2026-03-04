from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class IHWResult:
    """Result container for Independent Hypothesis Weighting.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Original input p-values, in the original input order.
    adj_pvalues : NDArray[np.float64]
        Adjusted p-values after the weighted BH (or Bonferroni) procedure.
    weights : NDArray[np.float64]
        Per-hypothesis weights learned by IHW.
    weighted_pvalues : NDArray[np.float64]
        P-values divided by their weights (``pvalues / weights``).
    covariates : NDArray[np.float64]
        Original input covariates, in the original input order.
    groups : NDArray[np.intp]
        Bin (stratum) assignment for each hypothesis, 0-indexed.
    folds : NDArray[np.intp]
        Fold assignment for each hypothesis, 0-indexed.
    weight_matrix : NDArray[np.float64]
        Weight per (bin, fold) combination, shape ``(nbins, nfolds)``.
    alpha : float
        Nominal significance level.
    nbins : int
        Number of covariate bins (strata).
    nfolds : int
        Number of cross-validation folds.
    regularization_terms : NDArray[np.float64]
        Regularization parameter (lambda) chosen for each fold.
    m_groups : NDArray[np.intp]
        Number of hypotheses in each stratum.
    penalty : str
        Regularization penalty type: ``"total_variation"`` or
        ``"uniform_deviation"``.
    covariate_type : str
        Whether the covariate is ``"ordinal"`` or ``"nominal"``.
    adjustment_type : str
        Multiple testing adjustment: ``"bh"`` or ``"bonferroni"``.
    """

    pvalues: NDArray[np.float64]
    adj_pvalues: NDArray[np.float64]
    weights: NDArray[np.float64]
    weighted_pvalues: NDArray[np.float64]
    covariates: NDArray[np.float64]
    groups: NDArray[np.intp]
    folds: NDArray[np.intp]
    weight_matrix: NDArray[np.float64]
    alpha: float
    nbins: int
    nfolds: int
    regularization_terms: NDArray[np.float64]
    m_groups: NDArray[np.intp]
    penalty: str
    covariate_type: str
    adjustment_type: str

    @property
    def rejected_hypotheses(self) -> NDArray[np.bool_]:
        """Boolean mask of hypotheses rejected at level ``alpha``."""
        return self.adj_pvalues <= self.alpha

    @property
    def n_rejections(self) -> int:
        """Total number of rejected hypotheses."""
        return int(np.sum(self.rejected_hypotheses))
