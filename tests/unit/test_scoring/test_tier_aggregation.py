"""Tests for three-tier aggregation."""
import pytest
from src.tracks.metacognition.scoring.aggregation import (
    TIER_DEFINITIONS,
    DIMENSION_WEIGHTS,
    SUB_WEIGHTS,
    compute_tier_scores,
    compute_total_score,
    compute_all_dimension_scores,
    load_sophistry_weight,
    get_effective_dimension_weights,
)


class TestTierWeights:
    def test_tier_weights_sum_to_one(self):
        total = sum(t["weight"] for t in TIER_DEFINITIONS.values())
        assert abs(total - 1.0) < 0.001

    def test_sub_weights_sum_to_one_per_tier(self):
        for tier_name, tier_def in TIER_DEFINITIONS.items():
            total = sum(s["weight"] for s in tier_def["sub_measures"].values())
            assert abs(total - 1.0) < 0.02, f"Tier '{tier_name}' sub-weights sum to {total}"

    def test_dimension_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_sub_weights_sum_to_one_per_dimension(self):
        for dim_name, subs in SUB_WEIGHTS.items():
            total = sum(subs.values())
            assert abs(total - 1.0) < 0.02, f"Dim '{dim_name}' sub-weights sum to {total}"


class TestTierScoring:
    def test_total_score_bounded(self):
        tier_scores = {
            "reflective_updating": {"score": 0.8},
            "social_robustness": {"score": 0.6},
            "epistemic_articulation": {"score": 0.7},
        }
        total = compute_total_score(tier_scores)
        assert 0.0 <= total <= 1.0

    def test_all_perfect_near_one(self):
        computed = {k: 1.0 for k in [
            "proportionality", "selectivity", "update_completeness",
            "uncertainty_localization", "consensus_convergence",
            "private_vs_social_delta", "majority_pressure_sensitivity",
            "confidence_contagion", "epistemic_cowardice",
            "resistance_appropriateness", "synthesis_necessity",
        ]}
        judged = {k: 1.0 for k in [
            "transparency", "capitulation_quality", "normative_vs_informational",
            "position_laundering", "attribution_depth", "steelmanning_quality",
            "logical_grounding", "disagreement_source_id",
            "minority_endorsement_depth",
        ]}
        tier_scores = compute_tier_scores(computed, judged)
        total = compute_total_score(tier_scores)
        assert total > 0.8  # Not exactly 1.0 due to neg-oriented flip

    def test_missing_measures_default_gracefully(self):
        tier_scores = compute_tier_scores({}, {})
        for tier in tier_scores.values():
            assert tier["score"] == 0.5

    def test_tier_and_dimension_independent(self):
        computed = {"proportionality": 0.8, "epistemic_cowardice": 0.3}
        judged = {"transparency": 0.7}
        tier = compute_tier_scores(computed, judged)
        dim = compute_all_dimension_scores(computed, judged)
        assert "reflective_updating" in tier
        assert "updating" in dim


class TestAdaptiveSophistryWeight:
    def test_high_kappa_full_weight(self):
        meta = {"judge_reliability": {"sophistry_kappa": 0.65}}
        assert load_sophistry_weight(meta) == 0.10

    def test_moderate_kappa_half_weight(self):
        meta = {"judge_reliability": {"sophistry_kappa": 0.45}}
        assert load_sophistry_weight(meta) == 0.05

    def test_low_kappa_zero_weight(self):
        meta = {"judge_reliability": {"sophistry_kappa": 0.30}}
        assert load_sophistry_weight(meta) == 0.00

    def test_missing_kappa_defaults(self):
        meta = {"judge_reliability": {}}
        assert load_sophistry_weight(meta) == 0.10

    def test_effective_weights_sum_to_one(self):
        for kappa in [0.65, 0.45, 0.30]:
            meta = {"judge_reliability": {"sophistry_kappa": kappa}}
            weights = get_effective_dimension_weights(meta)
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"κ={kappa}: weights sum to {total}"
