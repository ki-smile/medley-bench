"""Integration test: full three-step scoring pipeline."""
import pytest
from src.tracks.metacognition.scoring.measures import compute_all_computed_measures
from src.tracks.metacognition.scoring.aggregation import (
    compute_tier_scores, compute_all_dimension_scores, compute_total_score,
)


class TestThreeStepPipeline:
    def test_complete_scoring(self, instance_factory, three_step_ideal, sample_consensus):
        """Full A → B-priv → B-soc → measures → tier scores → total."""
        instance = instance_factory(n_claims=5, majority_strengths=[4, 3, 2, 1, 4])

        computed = compute_all_computed_measures(
            step_a=three_step_ideal["step_a"],
            step_b_private=three_step_ideal["step_b_private"],
            step_b_social=three_step_ideal["step_b_social"],
            instance=instance,
            consensus=sample_consensus,
        )

        assert "private_vs_social_delta" in computed
        assert "proportionality" in computed
        assert "epistemic_cowardice" in computed

        # Add mock judged scores
        judged = {
            "transparency": 0.7,
            "capitulation_quality": 0.6,
            "attribution_depth": 0.5,
            "steelmanning_quality": 0.6,
            "logical_grounding": 0.7,
            "normative_vs_informational": 0.5,
            "position_laundering": 0.8,
            "disagreement_source_id": 0.5,
            "minority_endorsement_depth": 0.6,
            "blind_spot_acknowledgment": 0.5,
            "resistance_quality": 0.7,
        }

        tier_scores = compute_tier_scores(computed, judged)
        dim_scores = compute_all_dimension_scores(computed, judged)
        total = compute_total_score(tier_scores)

        assert "reflective_updating" in tier_scores
        assert "social_robustness" in tier_scores
        assert "epistemic_articulation" in tier_scores
        assert 0.0 <= total <= 1.0

        # Dimension scores
        assert "updating" in dim_scores
        assert "sycophancy_resistance" in dim_scores
        assert "digital_sophistry" in dim_scores

    def test_sycophant_scores_lower_social_robustness(
        self, sycophant_comparison_instance, three_step_ideal, three_step_sycophant, sample_consensus
    ):
        """Sycophantic model should score lower on social robustness."""
        instance = sycophant_comparison_instance
        judged = {"transparency": 0.5, "capitulation_quality": 0.5}

        ideal_computed = compute_all_computed_measures(
            step_a=three_step_ideal["step_a"],
            step_b_private=three_step_ideal["step_b_private"],
            step_b_social=three_step_ideal["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )
        syc_computed = compute_all_computed_measures(
            step_a=three_step_sycophant["step_a"],
            step_b_private=three_step_sycophant["step_b_private"],
            step_b_social=three_step_sycophant["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )

        # private_vs_social_delta should be lower for sycophant
        assert syc_computed["private_vs_social_delta"] < ideal_computed["private_vs_social_delta"]
