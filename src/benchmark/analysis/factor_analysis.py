"""Post-pilot factor analysis for MEDLEY-BENCH.

Determines whether the six theoretical dimensions are empirically independent
or collapse into fewer factors (expected: the three tiers).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_factor_analysis(
    measure_matrix: np.ndarray,
    measure_names: list[str],
    n_components: int = 6,
) -> dict:
    """Run PCA-based factor analysis on the measure matrix.

    Args:
        measure_matrix: (n_instances, n_measures) matrix of scores
        measure_names: names corresponding to columns
        n_components: max components to extract

    Returns:
        Dict with explained variance, loadings, and interpretation.
    """
    from sklearn.decomposition import PCA

    if measure_matrix.shape[0] < 10 or measure_matrix.shape[1] < 3:
        return {"error": "Insufficient data for factor analysis"}

    # Standardize
    means = measure_matrix.mean(axis=0)
    stds = measure_matrix.std(axis=0)
    stds[stds < 1e-10] = 1.0
    standardized = (measure_matrix - means) / stds

    # PCA
    n_comp = min(n_components, measure_matrix.shape[1], measure_matrix.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(standardized)

    # How many components needed for 80% variance?
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_for_80 = int(np.searchsorted(cumvar, 0.80)) + 1

    # Loadings
    loadings = {}
    for i in range(min(3, n_comp)):
        component_loadings = {
            measure_names[j]: round(float(pca.components_[i, j]), 4)
            for j in range(len(measure_names))
        }
        # Sort by absolute loading
        sorted_loadings = dict(sorted(
            component_loadings.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ))
        loadings[f"component_{i+1}"] = sorted_loadings

    return {
        "n_instances": measure_matrix.shape[0],
        "n_measures": measure_matrix.shape[1],
        "explained_variance_ratio": [
            round(float(v), 4) for v in pca.explained_variance_ratio_[:6]
        ],
        "cumulative_variance": [round(float(v), 4) for v in cumvar[:6]],
        "n_components_for_80pct": n_for_80,
        "loadings": loadings,
        "interpretation": (
            f"{'Three' if n_for_80 <= 3 else 'More than three'} factors explain "
            f"80% of variance ({n_for_80} components). "
            f"{'This supports the three-tier structure.' if n_for_80 <= 3 else 'The six dimensions may be partially independent.'}"
        ),
    }
