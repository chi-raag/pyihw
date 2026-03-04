from __future__ import annotations

import numpy as np

from pyihw.weighting import (
    ihw_convex,
    thresholds_to_weights,
    total_variation,
    uniform_deviation,
)


class TestThresholdsToWeights:
    def test_weight_constraint(self) -> None:
        ts = np.array([0.01, 0.05, 0.02])
        m_groups = np.array([100, 200, 100])
        ws = thresholds_to_weights(ts, m_groups)
        np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0)

    def test_all_zero_thresholds(self) -> None:
        ts = np.array([0.0, 0.0, 0.0])
        m_groups = np.array([100, 200, 100])
        ws = thresholds_to_weights(ts, m_groups)
        np.testing.assert_array_equal(ws, [1.0, 1.0, 1.0])

    def test_proportional_to_thresholds(self) -> None:
        ts = np.array([0.02, 0.04])
        m_groups = np.array([100, 100])
        ws = thresholds_to_weights(ts, m_groups)
        expected = ts * 2 / np.sum(ts)
        np.testing.assert_allclose(ws, expected)

    def test_length_mismatch_raises(self) -> None:
        try:
            thresholds_to_weights(np.array([0.1, 0.2]), np.array([100]))
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestPenalties:
    def test_total_variation_uniform(self) -> None:
        ws = np.array([1.0, 1.0, 1.0])
        assert total_variation(ws) == 0.0

    def test_total_variation_known(self) -> None:
        ws = np.array([0.5, 1.5, 0.5])
        np.testing.assert_allclose(total_variation(ws), 2.0)

    def test_uniform_deviation_uniform(self) -> None:
        ws = np.array([1.0, 1.0, 1.0])
        assert uniform_deviation(ws) == 0.0

    def test_uniform_deviation_known(self) -> None:
        ws = np.array([0.5, 1.5, 1.0])
        np.testing.assert_allclose(uniform_deviation(ws), 1.0)


class TestIhwConvex:
    def test_lambda_zero_returns_uniform(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.uniform(size=100)) for _ in range(3)]
        m_groups = np.array([100, 100, 100])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=0.0,
            adjustment_type="bh",
        )
        np.testing.assert_array_equal(ws, [1.0, 1.0, 1.0])

    def test_weight_constraint_bh(self) -> None:
        rng = np.random.default_rng(42)
        signal_p = np.sort(rng.beta(0.3, 5, size=200))
        null_p = np.sort(rng.uniform(size=200))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        np.testing.assert_allclose(
            np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6
        )

    def test_weight_constraint_bonferroni(self) -> None:
        rng = np.random.default_rng(42)
        signal_p = np.sort(rng.beta(0.3, 5, size=200))
        null_p = np.sort(rng.uniform(size=200))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bonferroni",
        )
        np.testing.assert_allclose(
            np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6
        )

    def test_signal_bin_gets_higher_weight(self) -> None:
        rng = np.random.default_rng(42)
        signal_p = np.sort(rng.beta(0.3, 5, size=500))
        null_p = np.sort(rng.uniform(size=500))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([500, 500])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        assert ws[0] > ws[1]

    def test_uniform_deviation_penalty(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.beta(0.5, 5, size=200)) for _ in range(3)]
        m_groups = np.array([200, 200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="uniform_deviation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        np.testing.assert_allclose(
            np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6
        )

    def test_weights_are_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.uniform(size=200)) for _ in range(5)]
        m_groups = np.array([200, 200, 200, 200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=5.0,
            adjustment_type="bh",
        )
        assert np.all(ws >= -1e-10)
