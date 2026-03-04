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
    from pyihw import IHWResult, bh_threshold, ihw, load_airway

    assert callable(ihw)
    assert callable(bh_threshold)
    assert callable(load_airway)
    assert IHWResult is not None


def test_load_airway() -> None:
    from pyihw import load_airway

    pvalues, basemean = load_airway()
    assert pvalues.shape == (33469,)
    assert basemean.shape == (33469,)
    assert pvalues.dtype == np.float64
    assert basemean.dtype == np.float64
    assert np.all(pvalues >= 0.0) and np.all(pvalues <= 1.0)
    assert np.all(basemean >= 0.0)


class TestAirway:
    """Test IHW on real genomic data with extreme p-values."""

    def test_more_rejections_than_bh(self) -> None:
        """IHW must beat BH on the airway dataset (extreme p-values)."""
        from pyihw import load_airway

        pvalues, basemean = load_airway()
        result = ihw(pvalues, basemean, alpha=0.1, rng=np.random.default_rng(42))
        t_bh = bh_threshold(pvalues, alpha=0.1)
        bh_rejections = int(np.sum(pvalues <= t_bh))
        # R IHW gets ~4892 vs BH's 4099.  We should get a substantial gain.
        assert result.n_rejections > bh_rejections + 100

    def test_nonuniform_weights(self) -> None:
        """Weights must vary across bins — uniform means the LP failed."""
        from pyihw import load_airway

        pvalues, basemean = load_airway()
        result = ihw(pvalues, basemean, alpha=0.1, rng=np.random.default_rng(42))
        assert result.weights.min() < 0.5
        assert result.weights.max() > 1.5


class TestRReference:
    """Compare ihw_convex output against R IHW package on the same inputs."""

    def test_lp_weights_match_r(self) -> None:
        """With identical bin assignments, the LP solver should match R within 1e-5."""
        from pathlib import Path

        import csv

        data_dir = Path(__file__).parent / "data"

        # Load p-values from R reference
        pvalues = []
        with open(data_dir / "r_reference.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pvalues.append(float(row["pvalue"]))
        pv = np.array(pvalues)

        # Load R group assignments (1-indexed in R)
        r_groups = []
        with open(data_dir / "r_groups_single_fold.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_groups.append(int(row["group"]))
        groups = np.array(r_groups) - 1  # convert to 0-indexed

        # Sort and split by R's groups
        order = np.argsort(pv)
        sorted_pvalues = pv[order]
        sorted_groups = groups[order]
        nbins = 10
        m_groups = np.bincount(sorted_groups, minlength=nbins)
        split_sorted_pvalues = [
            np.sort(sorted_pvalues[sorted_groups == g]) for g in range(nbins)
        ]

        from pyihw.weighting import ihw_convex

        ws = ihw_convex(
            split_sorted_pvalues=split_sorted_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bh",
        )

        # R reference weights (nfolds=1, lambda=Inf, nbins=10)
        r_weights = np.array(
            [
                0.000000,
                0.032857,
                0.053412,
                0.079898,
                0.011118,
                1.447015,
                1.818260,
                2.563411,
                2.070199,
                1.923829,
            ]
        )

        np.testing.assert_allclose(ws, r_weights, atol=1e-5)
        # Weight constraint
        np.testing.assert_allclose(
            np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6
        )
