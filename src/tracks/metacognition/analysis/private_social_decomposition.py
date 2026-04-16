"""Analysis: private vs social revision decomposition.

Analyzes the three-step decomposition across all instances and models
to understand the relative contributions of self-revision vs social influence.
"""
from __future__ import annotations

import numpy as np


def decompose_revision_sources(instance_results: list[dict]) -> dict:
    """Decompose revision patterns across all instances.

    Returns summary statistics for the private_vs_social_delta distribution,
    plus identification of extreme patterns.
    """
    deltas = []
    for r in instance_results:
        d = r.get("computed", {}).get("private_vs_social_delta")
        if d is not None:
            deltas.append(d)

    if not deltas:
        return {"n_instances": 0}

    arr = np.array(deltas)
    return {
        "n_instances": len(deltas),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "pct_self_dominated": round(float((arr > 0.7).mean()), 4),
        "pct_social_dominated": round(float((arr < 0.3).mean()), 4),
        "pct_balanced": round(float(((arr >= 0.3) & (arr <= 0.7)).mean()), 4),
    }


def compare_models(all_results: dict[str, list[dict]]) -> list[dict]:
    """Compare private_vs_social_delta across models.

    Returns sorted list from most self-driven to most social-driven.
    """
    model_stats = []
    for model, results in all_results.items():
        decomp = decompose_revision_sources(results)
        if decomp.get("n_instances", 0) > 0:
            model_stats.append({
                "model": model,
                "mean_delta": decomp["mean"],
                "median_delta": decomp["median"],
                "n_instances": decomp["n_instances"],
                "pct_self_dominated": decomp["pct_self_dominated"],
                "pct_social_dominated": decomp["pct_social_dominated"],
            })

    model_stats.sort(key=lambda x: x["mean_delta"], reverse=True)
    return model_stats
