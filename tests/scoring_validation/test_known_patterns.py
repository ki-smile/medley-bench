"""Validate that known behavioral archetypes produce expected score patterns."""
import pytest
from src.tracks.metacognition.scoring.measures import (
    compute_all_computed_measures,
    private_vs_social_delta,
    epistemic_cowardice_score,
)
from src.tracks.metacognition.scoring.aggregation import compute_tier_scores, compute_total_score


class TestThreeStepPatterns:
    """Three-step archetypes should produce distinguishable score profiles."""

    def test_sycophant_low_social_robustness(
        self, sycophant_comparison_instance, three_step_sycophant, sample_consensus
    ):
        """Sycophant: no self-revision + total capitulation → low Tier 2."""
        instance = sycophant_comparison_instance
        computed = compute_all_computed_measures(
            step_a=three_step_sycophant["step_a"],
            step_b_private=three_step_sycophant["step_b_private"],
            step_b_social=three_step_sycophant["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )
        assert computed["private_vs_social_delta"] < 0.15

    def test_anchored_neutral_scores(
        self, instance_factory, three_step_anchored, sample_consensus
    ):
        """Anchored: no change → neutral private_vs_social_delta."""
        instance = instance_factory(n_claims=5)
        computed = compute_all_computed_measures(
            step_a=three_step_anchored["step_a"],
            step_b_private=three_step_anchored["step_b_private"],
            step_b_social=three_step_anchored["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )
        assert computed["private_vs_social_delta"] == 0.5

    def test_ideal_beats_sycophant(
        self, sycophant_comparison_instance, three_step_ideal, three_step_sycophant, sample_consensus
    ):
        """Ideal pattern should have higher private_vs_social_delta than sycophant."""
        instance = sycophant_comparison_instance
        ideal = compute_all_computed_measures(
            step_a=three_step_ideal["step_a"],
            step_b_private=three_step_ideal["step_b_private"],
            step_b_social=three_step_ideal["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )
        syc = compute_all_computed_measures(
            step_a=three_step_sycophant["step_a"],
            step_b_private=three_step_sycophant["step_b_private"],
            step_b_social=three_step_sycophant["step_b_social"],
            instance=instance, consensus=sample_consensus,
        )
        assert ideal["private_vs_social_delta"] > syc["private_vs_social_delta"]


class TestMonotonicity:
    """Score monotonicity: more of X → predictably higher/lower score."""

    def test_more_social_influence_lower_delta(self, sycophant_comparison_instance, step_response_factory):
        """Increasing social influence → lower private_vs_social_delta."""
        instance = sycophant_comparison_instance
        step_a = step_response_factory(["very_high"] * 5)
        # Some private change (so denominator isn't zero for low-social case)
        step_b_priv = step_response_factory(["high", "high", "very_high", "very_high", "high"])

        # Low social influence (small additional change)
        step_b_soc_low = step_response_factory(["high", "moderate", "very_high", "very_high", "high"])
        score_low_social = private_vs_social_delta(step_a, step_b_priv, step_b_soc_low, instance)

        # High social influence (large additional change)
        step_b_soc_high = step_response_factory(["low", "low", "low", "low", "low"])
        score_high_social = private_vs_social_delta(step_a, step_b_priv, step_b_soc_high, instance)

        assert score_high_social < score_low_social

    def test_more_hedging_higher_cowardice(self, step_response_factory):
        """More moderate-clustering → higher epistemic cowardice."""
        varied = step_response_factory(["very_high", "low", "very_high", "very_low", "high"])
        hedged = step_response_factory(["moderate"] * 5)

        assert epistemic_cowardice_score(hedged) > epistemic_cowardice_score(varied)


class TestNeutralDefaults:
    """Missing or minimal data should produce neutral (0.5) scores."""

    def test_empty_instance_neutral(self, step_response_factory):
        """Empty instance → neutral scores for all measures."""
        from src.tracks.metacognition.scoring.measures import (
            update_proportionality, update_selectivity, uncertainty_localization,
        )
        step_a = step_response_factory(["high"])
        step_b = step_response_factory(["low"])
        empty_instance = {"key_claims": []}

        assert update_proportionality(step_a, step_b, empty_instance) == 0.5
        assert update_selectivity(step_a, step_b, empty_instance) == 0.5
        assert uncertainty_localization(step_b, empty_instance) == 0.5
