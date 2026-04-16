"""Tests for instructional instability diagnostic."""
import pytest
from src.tracks.metacognition.scoring.measures import instructional_instability


class TestInstructionalInstability:
    def test_selective_revision_scores_high(self, instance_factory, step_response_factory):
        """Low-confidence claims change more -> good (score > 0.5)."""
        instance = instance_factory(n_claims=5)
        step_a = step_response_factory(["very_high", "very_high", "low", "very_low", "very_high"])
        step_b_priv = step_response_factory(["very_high", "very_high", "moderate", "low", "very_high"])
        score = instructional_instability(step_a, step_b_priv, instance)
        assert score > 0.5

    def test_unstable_revision_scores_low(self, instance_factory, step_response_factory):
        """High-confidence claims change more -> bad (score < 0.5)."""
        instance = instance_factory(n_claims=5)
        step_a = step_response_factory(["very_high", "very_high", "low", "very_low", "very_high"])
        step_b_priv = step_response_factory(["moderate", "moderate", "low", "very_low", "moderate"])
        score = instructional_instability(step_a, step_b_priv, instance)
        assert score < 0.5

    def test_no_change_neutral(self, instance_factory, step_response_factory):
        """No change at all -> neutral 0.5."""
        instance = instance_factory(n_claims=5)
        same = step_response_factory(["high"] * 5)
        score = instructional_instability(same, same, instance)
        assert score == 0.5

    def test_too_few_claims(self, instance_factory, step_response_factory):
        """Fewer than 3 claims -> default 0.5."""
        instance = instance_factory(n_claims=2)
        step_a = step_response_factory(["high", "low"])
        step_b = step_response_factory(["low", "high"])
        score = instructional_instability(step_a, step_b, instance)
        assert score == 0.5
