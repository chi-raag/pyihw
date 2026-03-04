from __future__ import annotations

import numpy as np

from pyihw.utils import bh_adjust, bh_threshold, grenander_estimator, safe_divide


class TestBhThreshold:
    def test_known_values(self) -> None:
        pvalues = np.array([0.001, 0.01, 0.03, 0.04, 0.5, 0.8])
        assert bh_threshold(pvalues, alpha=0.1) == 0.04

    def test_no_rejections(self) -> None:
        pvalues = np.array([0.5, 0.6, 0.7])
        assert bh_threshold(pvalues, alpha=0.01) == 0.0

    def test_all_rejected(self) -> None:
        pvalues = np.array([0.001, 0.002, 0.003])
        assert bh_threshold(pvalues, alpha=0.5) == 0.003

    def test_with_mtotal_larger(self) -> None:
        pvalues = np.array([0.001, 0.01, 0.04])
        t1 = bh_threshold(pvalues, alpha=0.1)
        t2 = bh_threshold(pvalues, alpha=0.1, m_total=100)
        assert t2 <= t1


class TestBhAdjust:
    def test_equivalence_with_threshold(self) -> None:
        rng = np.random.default_rng(1)
        pvalues = np.concatenate([rng.uniform(size=1000), rng.beta(0.5, 7, size=200)])
        alpha = 0.1
        adj = bh_adjust(pvalues)
        t = bh_threshold(pvalues, alpha)
        np.testing.assert_array_equal(adj <= alpha, pvalues <= t)

    def test_adjusted_pvalues_are_monotone_on_sorted(self) -> None:
        pvalues = np.array([0.001, 0.01, 0.05, 0.5])
        adj = bh_adjust(pvalues)
        sorted_adj = adj[np.argsort(pvalues)]
        assert np.all(np.diff(sorted_adj) >= 0)

    def test_adjusted_pvalues_clipped_to_one(self) -> None:
        pvalues = np.array([0.9, 0.95, 0.99])
        adj = bh_adjust(pvalues)
        assert np.all(adj <= 1.0)

    def test_with_mtotal(self) -> None:
        pvalues = np.array([0.01, 0.02])
        adj1 = bh_adjust(pvalues)
        adj2 = bh_adjust(pvalues, m_total=100)
        assert np.all(adj2 >= adj1)


class TestSafeDivide:
    def test_normal_division(self) -> None:
        np.testing.assert_allclose(
            safe_divide(np.array([0.1]), np.array([2.0])), [0.05]
        )

    def test_zero_numerator(self) -> None:
        np.testing.assert_array_equal(
            safe_divide(np.array([0.0]), np.array([0.0])), [0.0]
        )

    def test_zero_denominator_nonzero_numerator(self) -> None:
        np.testing.assert_array_equal(
            safe_divide(np.array([0.5]), np.array([0.0])), [1.0]
        )

    def test_result_clipped_to_one(self) -> None:
        np.testing.assert_array_equal(
            safe_divide(np.array([0.8]), np.array([0.5])), [1.0]
        )


class TestGrenanderEstimator:
    def test_uniform_pvalues(self) -> None:
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.uniform(size=10000))
        result = grenander_estimator(pvalues, m_total=10000)
        assert np.all(result.slopes >= 0.8)
        assert np.all(result.slopes <= 1.4)

    def test_knots_are_sorted(self) -> None:
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.beta(0.5, 7, size=500))
        result = grenander_estimator(pvalues, m_total=500)
        assert np.all(np.diff(result.x_knots) > 0)
        assert np.all(np.diff(result.y_knots) >= 0)

    def test_slopes_are_nonincreasing(self) -> None:
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.beta(0.5, 7, size=500))
        result = grenander_estimator(pvalues, m_total=500)
        assert np.all(np.diff(result.slopes) <= 1e-12)

    def test_m_total_larger_than_sample(self) -> None:
        pvalues = np.sort(np.array([0.01, 0.02, 0.05, 0.1, 0.3]))
        result = grenander_estimator(pvalues, m_total=100)
        assert result.y_knots[-1] <= 0.06

    def test_single_pvalue(self) -> None:
        result = grenander_estimator(np.array([0.5]), m_total=1)
        assert len(result.slopes) >= 1
