from __future__ import annotations

import numpy as np

from pyihw.weighting import thresholds_to_weights, total_variation, uniform_deviation


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
