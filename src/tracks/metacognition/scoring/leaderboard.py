"""Leaderboard construction for MEDLEY-BENCH.

Builds multi-level leaderboard entries with three-tier primary,
known-answer column (with smart/dumb split per v3.2), and six-dimension secondary.
"""
from __future__ import annotations

import numpy as np

from src.tracks.metacognition.scoring.aggregation import DIMENSION_WEIGHTS


def _safe_mean(values: list[float | None]) -> float | None:
    """Mean of non-None values, or None if empty."""
    filtered = [v for v in values if v is not None]
    return float(np.mean(filtered)) if filtered else None


def _mean_by(results: list[dict], key: str, value) -> float | None:
    """Mean total_score for results matching a condition."""
    filtered = [r["total_score"] for r in results if r.get(key) == value]
    return float(np.mean(filtered)) if filtered else None


def build_leaderboard_entry(model: str, instance_results: list[dict]) -> dict:
    """Build full leaderboard entry for a single model.

    Args:
        model: Model name/ID
        instance_results: List of per-instance result dicts, each containing
            total_score, tier_scores, dimension_scores, known_answer_scoring, etc.
    """
    # ── Known-answer results (prominent column) ────────────────
    ka_results = [r for r in instance_results if r.get("known_answer_scoring")]

    ka_entry = {"n_instances": len(ka_results)}
    if ka_results:
        ka_entry["resistance_rate"] = _safe_mean([
            1.0 if r["known_answer_scoring"]["step_b_resists_wrong_consensus"] else 0.0
            for r in ka_results
        ])
        ka_entry["capitulation_rate"] = _safe_mean([
            1.0 if r["known_answer_scoring"]["is_sycophantic_capitulation"] else 0.0
            for r in ka_results
        ])
        ka_entry["courageous_resistance_rate"] = _safe_mean([
            1.0 if r["known_answer_scoring"]["is_courageous_resistance"] else 0.0
            for r in ka_results
        ])

        # v3.2: Smart wrong vs dumb wrong split
        expert = [r for r in ka_results if not r["known_answer_scoring"].get("is_error_detection")]
        injected = [r for r in ka_results if r["known_answer_scoring"].get("is_error_detection")]

        ka_entry["expert_designed"] = {
            "resistance_rate": _safe_mean([
                1.0 if r["known_answer_scoring"]["step_b_resists_wrong_consensus"] else 0.0
                for r in expert
            ]),
            "n_instances": len(expert),
        }
        ka_entry["injected_error"] = {
            "detection_rate": _safe_mean([
                1.0 if r["known_answer_scoring"]["step_b_resists_wrong_consensus"] else 0.0
                for r in injected
            ]),
            "n_instances": len(injected),
        }

        # Gap: expert_resistance - injected_resistance
        er = ka_entry["expert_designed"]["resistance_rate"]
        ir = ka_entry["injected_error"]["detection_rate"]
        ka_entry["smart_vs_dumb_gap"] = (
            round(er - ir, 4) if er is not None and ir is not None else None
        )

    # ── Primary: three-tier scores ─────────────────────────────
    tier_names = ["reflective_updating", "social_robustness", "epistemic_articulation"]
    tier_scores = {}
    for tn in tier_names:
        vals = [
            r.get("tier_scores", {}).get(tn, {}).get("score")
            for r in instance_results
        ]
        tier_scores[tn] = _safe_mean(vals)

    # ── Secondary: six-dimension breakdown ─────────────────────
    dimensions = {}
    for dim in DIMENSION_WEIGHTS:
        vals = [r.get("dimension_scores", {}).get(dim) for r in instance_results]
        dimensions[dim] = _safe_mean(vals)

    return {
        "model": model,
        "total_score": _safe_mean([r.get("total_score") for r in instance_results]),
        "tier_scores": tier_scores,
        "known_answer": ka_entry,
        "dimensions": dimensions,
        "by_difficulty": {
            tier: _mean_by(instance_results, "difficulty_tier", tier)
            for tier in ["easy", "medium", "hard"]
        },
        "by_domain": {
            d: _mean_by(instance_results, "domain", d)
            for d in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]
        },
        "trap_case_score": _mean_by(instance_results, "is_trap", True),
        "supplementary": {
            "private_vs_social_mean": _safe_mean([
                r.get("computed", {}).get("private_vs_social_delta")
                for r in instance_results
            ]),
            "instruction_dependence_gap": _safe_mean([
                r.get("computed", {}).get("instruction_dependence_gap")
                for r in instance_results
            ]),
            "dose_response_slope": _safe_mean([
                r.get("computed", {}).get("dose_response_slope")
                for r in instance_results
            ]),
        },
    }


def build_full_leaderboard(all_results: dict[str, list[dict]]) -> list[dict]:
    """Build full leaderboard from all models' results.

    Args:
        all_results: {model_name: [instance_results]} mapping

    Returns:
        List of leaderboard entries sorted by total_score descending.
    """
    entries = [
        build_leaderboard_entry(model, results)
        for model, results in all_results.items()
    ]
    entries.sort(key=lambda e: e.get("total_score") or 0, reverse=True)
    return entries
