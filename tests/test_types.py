from __future__ import annotations

import numpy as np

from pyihw._types import IHWResult


def test_ihw_result_rejected_hypotheses() -> None:
    result = IHWResult(
        pvalues=np.array([0.01, 0.5, 0.001]),
        adj_pvalues=np.array([0.05, 0.5, 0.01]),
        weights=np.array([1.5, 0.5, 1.0]),
        weighted_pvalues=np.array([0.0067, 1.0, 0.001]),
        covariates=np.array([10.0, 2.0, 8.0]),
        groups=np.array([0, 1, 0]),
        folds=np.array([0, 1, 0]),
        weight_matrix=np.array([[1.5, 1.0], [0.5, 1.0]]),
        alpha=0.1,
        nbins=2,
        nfolds=2,
        regularization_terms=np.array([1.0, 1.0]),
        m_groups=np.array([2, 1]),
        penalty="total_variation",
        covariate_type="ordinal",
        adjustment_type="bh",
    )
    expected_mask = np.array([True, False, True])
    np.testing.assert_array_equal(result.rejected_hypotheses, expected_mask)
    assert result.n_rejections == 2


def test_ihw_result_is_frozen() -> None:
    result = IHWResult(
        pvalues=np.array([0.01]),
        adj_pvalues=np.array([0.05]),
        weights=np.array([1.0]),
        weighted_pvalues=np.array([0.01]),
        covariates=np.array([1.0]),
        groups=np.array([0]),
        folds=np.array([0]),
        weight_matrix=np.array([[1.0]]),
        alpha=0.1,
        nbins=1,
        nfolds=1,
        regularization_terms=np.array([0.0]),
        m_groups=np.array([1]),
        penalty="total_variation",
        covariate_type="ordinal",
        adjustment_type="bh",
    )
    try:
        result.alpha = 0.2  # type: ignore[misc]
        raise AssertionError("Should not be able to mutate frozen dataclass")
    except AttributeError:
        pass
