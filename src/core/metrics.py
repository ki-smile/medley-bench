"""Statistical metrics for MEDLEY-BENCH.

Brier score, ECE, Fleiss kappa, and safe statistical wrappers.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def brier_score(predicted_probs: list[float], outcomes: list[bool]) -> float:
    """Brier score: mean squared error between predicted probabilities and outcomes.

    Lower is better. Range [0, 1].
    """
    if not predicted_probs or len(predicted_probs) != len(outcomes):
        return 0.5
    preds = np.array(predicted_probs, dtype=float)
    outs = np.array(outcomes, dtype=float)
    return float(np.mean((preds - outs) ** 2))


def expected_calibration_error(
    predicted_probs: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions into n_bins equally spaced intervals and computes
    the weighted average of |accuracy - confidence| per bin.
    """
    if not predicted_probs or len(predicted_probs) != len(outcomes):
        return 0.5
    preds = np.array(predicted_probs, dtype=float)
    outs = np.array(outcomes, dtype=float)
    n = len(preds)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (preds > bin_edges[i]) & (preds <= bin_edges[i + 1])
        if i == 0:
            mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = preds[mask].mean()
        avg_acc = outs[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Fleiss' kappa for inter-rater reliability.

    Args:
        ratings_matrix: (n_subjects, n_categories) matrix where each entry
            is the number of raters who assigned that category to that subject.

    Returns:
        Fleiss' kappa coefficient. Range [-1, 1].
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix[0].sum()

    if n_raters <= 1 or n_subjects == 0:
        return 0.0

    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (
        (ratings_matrix ** 2).sum(axis=1) - n_raters
    ) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    return float((P_bar - P_e) / (1 - P_e))


def spearmanr_safe(
    x: list | np.ndarray, y: list | np.ndarray
) -> tuple[float, float]:
    """Safe wrapper around scipy.stats.spearmanr.

    Returns (0.0, 1.0) for degenerate inputs instead of raising.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) < 3 or len(x_arr) != len(y_arr):
        return 0.0, 1.0

    # Check for constant arrays
    if np.std(x_arr) < 1e-10 or np.std(y_arr) < 1e-10:
        return 0.0, 1.0

    result = stats.spearmanr(x_arr, y_arr)
    rho = result.statistic if hasattr(result, "statistic") else result[0]
    pval = result.pvalue if hasattr(result, "pvalue") else result[1]

    if np.isnan(rho):
        return 0.0, 1.0

    return float(rho), float(pval)
