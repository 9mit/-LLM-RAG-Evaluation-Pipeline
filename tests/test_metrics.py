"""Tests for src.metrics — unit tests that don't require ML models.

These tests only exercise the _softmax static method, which has no
external dependencies beyond numpy.  The full LLMEvaluator class is
imported lazily so the test suite can run without torch installed.
"""

from __future__ import annotations

import numpy as np


def _get_softmax():
    """Import LLMEvaluator._softmax without triggering torch at module level."""
    # We inline the import so torch is only needed if the test actually runs
    # in an environment where it's available; otherwise the loader tests
    # still pass.
    try:
        from src.metrics import LLMEvaluator
    except ImportError:
        import pytest

        pytest.skip("torch/sentence-transformers not installed")
    return LLMEvaluator._softmax


class TestSoftmax:
    def test_uniform_input(self):
        softmax = _get_softmax()
        result = softmax(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)

    def test_sums_to_one(self):
        softmax = _get_softmax()
        result = softmax(np.array([2.0, 1.0, 0.1]))
        assert abs(result.sum() - 1.0) < 1e-7

    def test_large_values_no_overflow(self):
        """Numerically-stable softmax should handle large inputs."""
        softmax = _get_softmax()
        result = softmax(np.array([1000.0, 1000.0, 1000.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)

    def test_negative_values(self):
        softmax = _get_softmax()
        result = softmax(np.array([-1.0, -2.0, -3.0]))
        assert result[0] > result[1] > result[2]
        assert abs(result.sum() - 1.0) < 1e-7
