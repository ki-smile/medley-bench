"""Tests for core statistical metrics."""
import numpy as np
import pytest
from src.core.metrics import (
    brier_score,
    expected_calibration_error,
    fleiss_kappa,
    spearmanr_safe,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0

    def test_worst_predictions(self):
        assert brier_score([0.0, 1.0], [True, False]) == 1.0

    def test_moderate(self):
        score = brier_score([0.7, 0.3, 0.5], [True, False, True])
        assert 0.0 < score < 0.5

    def test_empty_returns_default(self):
        assert brier_score([], []) == 0.5


class TestECE:
    def test_perfect_calibration(self):
        # All predictions match reality
        ece = expected_calibration_error([1.0, 0.0], [True, False], n_bins=2)
        assert ece < 0.1

    def test_empty_returns_default(self):
        assert expected_calibration_error([], []) == 0.5


class TestFleissKappa:
    def test_perfect_agreement(self):
        # 5 subjects, 3 categories, 4 raters all agree
        matrix = np.array([
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [4, 0, 0],
            [0, 4, 0],
        ])
        assert fleiss_kappa(matrix) == 1.0

    def test_random_agreement(self):
        # Uniform random assignment should give kappa near 0
        rng = np.random.default_rng(42)
        matrix = np.zeros((20, 3))
        for i in range(20):
            for _ in range(5):
                matrix[i, rng.integers(0, 3)] += 1
        kappa = fleiss_kappa(matrix)
        assert -0.3 < kappa < 0.3


class TestSpearmanrSafe:
    def test_perfect_correlation(self):
        rho, p = spearmanr_safe([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(rho - 1.0) < 0.01

    def test_constant_returns_zero(self):
        rho, p = spearmanr_safe([1, 1, 1], [2, 3, 4])
        assert rho == 0.0

    def test_too_few_returns_default(self):
        rho, p = spearmanr_safe([1, 2], [3, 4])
        assert rho == 0.0
        assert p == 1.0
