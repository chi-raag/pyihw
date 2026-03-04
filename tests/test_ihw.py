from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pyihw._types import IHWResult
from pyihw.ihw import _ihw_internal, ihw
from pyihw.splitting import groups_by_filter
from pyihw.utils import bh_threshold


def _wasserman_sim(
    rng: np.random.Generator, m: int = 10000, pi0: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    covariates = rng.uniform(0, 3, size=m)
    signals = rng.binomial(1, 1 - pi0, size=m)
    z = rng.normal(loc=signals * covariates)
    pvalues = 1 - norm.cdf(z)
    return pvalues, covariates


class TestIhwInternal:
    def test_single_fold_returns_weights(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        groups = groups_by_filter(covariates, nbins=10, rng=np.random.default_rng(1))
        order = np.argsort(pvalues)
        sorted_groups = groups[order]
        sorted_pvalues = pvalues[order]
        m_groups = np.bincount(sorted_groups, minlength=10)

        result = _ihw_internal(
            sorted_groups=sorted_groups,
            sorted_pvalues=sorted_pvalues,
            alpha=0.1,
            lambdas=np.array([0.0, 1.0, 5.0, np.inf]),
            m_groups=m_groups,
            penalty="total_variation",
            nfolds=1,
            nfolds_internal=1,
            nsplits_internal=1,
            adjustment_type="bh",
            null_proportion=False,
            null_proportion_level=0.5,
            rng=np.random.default_rng(2),
        )
        np.testing.assert_allclose(
            result["sorted_weights"].sum(), len(pvalues), atol=1.0
        )

    def test_multi_fold_weight_budget(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        groups = groups_by_filter(covariates, nbins=10, rng=np.random.default_rng(1))
        order = np.argsort(pvalues)
        sorted_groups = groups[order]
        sorted_pvalues = pvalues[order]
        m_groups = np.bincount(sorted_groups, minlength=10)

        result = _ihw_internal(
            sorted_groups=sorted_groups,
            sorted_pvalues=sorted_pvalues,
            alpha=0.1,
            lambdas=np.array([0.0, 5.0, np.inf]),
            m_groups=m_groups,
            penalty="total_variation",
            nfolds=5,
            nfolds_internal=5,
            nsplits_internal=1,
            adjustment_type="bh",
            null_proportion=False,
            null_proportion_level=0.5,
            rng=np.random.default_rng(2),
        )
        assert result["rjs"] > 0
        assert result["weight_matrix"].shape == (10, 5)


class TestIhwPublic:
    def test_returns_ihw_result(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert isinstance(result, IHWResult)

    def test_more_rejections_than_bh(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        t_bh = bh_threshold(pvalues, alpha=0.1)
        bh_rejections = int(np.sum(pvalues <= t_bh))
        assert result.n_rejections >= bh_rejections

    def test_weight_budget(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        np.testing.assert_allclose(result.weights.sum(), len(pvalues), atol=5.0)

    def test_lower_alpha_fewer_rejections(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        r1 = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        r2 = ihw(pvalues, covariates, alpha=0.01, rng=np.random.default_rng(1))
        assert r2.n_rejections < r1.n_rejections

    def test_single_bin_reduces_to_bh(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng, m=500)
        result = ihw(
            pvalues, covariates, alpha=0.1, nbins=1, rng=np.random.default_rng(1)
        )
        assert result.nbins == 1
        assert result.nfolds == 1
        np.testing.assert_allclose(result.weights, 1.0)

    def test_bonferroni_mode(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(
            pvalues,
            covariates,
            alpha=0.1,
            adjustment_type="bonferroni",
            rng=np.random.default_rng(1),
        )
        assert result.adjustment_type == "bonferroni"
        result_bh = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert result.n_rejections <= result_bh.n_rejections

    def test_output_lengths_match_input(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng, m=5000)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert len(result.pvalues) == 5000
        assert len(result.adj_pvalues) == 5000
        assert len(result.weights) == 5000
        assert len(result.weighted_pvalues) == 5000
        assert len(result.groups) == 5000
        assert len(result.folds) == 5000
        assert len(result.covariates) == 5000

    def test_invalid_pvalues_raises(self) -> None:
        try:
            ihw(np.array([-0.1, 0.5]), np.array([1.0, 2.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_nan_pvalues_raises(self) -> None:
        try:
            ihw(np.array([np.nan, 0.5]), np.array([1.0, 2.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_mismatched_lengths_raises(self) -> None:
        try:
            ihw(np.array([0.1, 0.5]), np.array([1.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_alpha_raises(self) -> None:
        try:
            ihw(np.array([0.1, 0.5]), np.array([1.0, 2.0]), alpha=1.5)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


def test_public_imports() -> None:
    from pyihw import IHWResult, bh_threshold, ihw

    assert callable(ihw)
    assert callable(bh_threshold)
    assert IHWResult is not None
