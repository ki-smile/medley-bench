"""Judge variance analysis for MEDLEY-BENCH.

Computes inter-rater reliability across judge models, including
the v3.1/v3.2 sophistry-specific IRR check.
"""
from __future__ import annotations

import logging

import numpy as np

from src.core.metrics import fleiss_kappa

logger = logging.getLogger(__name__)

# Judge measures that need IRR analysis
JUDGED_MEASURES = [
    "transparency", "capitulation_quality", "normative_vs_informational",
    "position_laundering", "attribution_depth", "steelmanning_quality",
    "logical_grounding", "disagreement_source_id",
    "minority_endorsement_depth", "blind_spot_acknowledgment",
    "resistance_quality",
]

# Sophistry-specific measures for targeted IRR check (v3.1)
SOPHISTRY_MEASURES = ["logical_grounding", "synthesis_necessity"]

# Bins for converting continuous scores to categorical for Fleiss kappa
SCORE_BINS = [0.0, 0.25, 0.50, 0.75, 1.01]  # 4 categories


def _scores_to_ratings_matrix(
    all_judge_scores: list[dict],
    measure: str,
    instance_ids: list[str],
) -> np.ndarray | None:
    """Convert judge scores to a ratings matrix for Fleiss kappa.

    Args:
        all_judge_scores: List of {judge_model, instance_id, scores: {measure: float}}
        measure: Which measure to build matrix for
        instance_ids: Which instances to include

    Returns:
        (n_instances, n_categories) matrix, or None if insufficient data.
    """
    n_categories = len(SCORE_BINS) - 1

    # Group by instance
    by_instance = {}
    for entry in all_judge_scores:
        iid = entry.get("instance_id")
        if iid not in instance_ids:
            continue
        score = entry.get("scores", {}).get(measure)
        if score is not None:
            by_instance.setdefault(iid, []).append(float(score))

    if len(by_instance) < 5:
        return None

    # Build matrix
    matrix = []
    for iid in instance_ids:
        scores = by_instance.get(iid, [])
        if len(scores) < 2:
            continue
        row = np.zeros(n_categories)
        for s in scores:
            bin_idx = min(np.searchsorted(SCORE_BINS[1:], s), n_categories - 1)
            row[bin_idx] += 1
        matrix.append(row)

    if len(matrix) < 5:
        return None

    return np.array(matrix)


def compute_overall_irr(
    all_judge_scores: list[dict],
    instance_ids: list[str],
) -> dict:
    """Compute overall inter-rater reliability across all judged measures.

    Returns dict with per-measure kappa, overall kappa, and summary.
    """
    kappas = {}
    for measure in JUDGED_MEASURES:
        matrix = _scores_to_ratings_matrix(all_judge_scores, measure, instance_ids)
        if matrix is not None:
            kappas[measure] = fleiss_kappa(matrix)

    if not kappas:
        return {"overall_kappa": None, "per_measure": {}, "n_measures": 0}

    overall = float(np.mean(list(kappas.values())))

    return {
        "overall_kappa": round(overall, 4),
        "per_measure": {k: round(v, 4) for k, v in kappas.items()},
        "n_measures": len(kappas),
        "meets_g7": overall >= 0.40,
        "meets_g8": overall >= 0.60,
    }


def check_sophistry_irr(
    all_judge_scores: list[dict],
    instance_ids: list[str],
) -> dict:
    """Targeted IRR check on sophistry-related measures (v3.1/v3.2).

    Returns sophistry kappa and the tiered weight determination.
    """
    kappas = {}
    for measure in SOPHISTRY_MEASURES:
        matrix = _scores_to_ratings_matrix(all_judge_scores, measure, instance_ids)
        if matrix is not None:
            kappas[measure] = fleiss_kappa(matrix)

    if not kappas:
        return {
            "sophistry_kappa": None,
            "per_measure": {},
            "weight_determination": "no_data",
            "recommended_weight": 0.10,
        }

    sophistry_kappa = float(np.mean(list(kappas.values())))

    # v3.2 tiered response
    if sophistry_kappa >= 0.60:
        weight = 0.10
        status = "substantial_agreement"
        recommendation = "Full weight — sophistry scoring is sufficiently reliable"
    elif sophistry_kappa >= 0.40:
        weight = 0.05
        status = "moderate_agreement"
        recommendation = "Halved weight — flag submission with reduced reliability note"
    else:
        weight = 0.00
        status = "poor_agreement"
        recommendation = (
            "Qualitative only — add worked examples to judge prompt and re-run. "
            "Freed weight redistributed to Tiers 1 and 2."
        )

    return {
        "sophistry_kappa": round(sophistry_kappa, 4),
        "per_measure": {k: round(v, 4) for k, v in kappas.items()},
        "weight_determination": status,
        "recommended_weight": weight,
        "recommendation": recommendation,
    }


def run_full_judge_variance(
    all_judge_scores: list[dict],
    instance_ids: list[str],
) -> dict:
    """Full judge variance analysis including sophistry check.

    Runs overall IRR + targeted sophistry check.
    Returns combined results for metadata.json.
    """
    overall = compute_overall_irr(all_judge_scores, instance_ids)
    sophistry = check_sophistry_irr(all_judge_scores, instance_ids)

    return {
        "overall": overall,
        "sophistry": sophistry,
        "judge_reliability": {
            "overall_kappa": overall["overall_kappa"],
            "sophistry_kappa": sophistry["sophistry_kappa"],
        },
    }
