"""Tests for individual computed measures."""
import pytest
from src.tracks.metacognition.scoring.measures import (
    update_proportionality,
    update_selectivity,
    update_completeness,
    uncertainty_localization,
    consensus_convergence,
    majority_pressure_sensitivity,
    confidence_contagion,
    epistemic_cowardice_score,
    resistance_appropriateness,
    synthesis_necessity_score,
    instruction_dependence_gap,
)


class TestUpdateProportionality:
    def test_proportional_change_scores_high(self, instance_factory, step_response_factory):
        """More change on high-disagreement claims → high score."""
        instance = instance_factory(
            n_claims=5, majority_strengths=[4, 4, 1, 1, 4],
        )
        # Big change on high-majority claims (4), small on low (1)
        step_a = step_response_factory(["very_high"] * 5)
        step_b = step_response_factory(["low", "low", "very_high", "very_high", "low"])
        score = update_proportionality(step_a, step_b, instance)
        assert score > 0.5

    def test_no_claims_returns_neutral(self, step_response_factory):
        score = update_proportionality(
            step_response_factory(["high"]),
            step_response_factory(["low"]),
            {"key_claims": []},
        )
        assert score == 0.5


class TestUpdateSelectivity:
    def test_selective_update_scores_high(self, instance_factory, step_response_factory):
        instance = instance_factory(
            n_claims=4, majority_strengths=[4, 4, 1, 1],
        )
        step_a = step_response_factory(["very_high"] * 4)
        # Big change on first two (high disagreement), no change on last two
        step_b = step_response_factory(["low", "low", "very_high", "very_high"])
        score = update_selectivity(step_a, step_b, instance)
        assert score > 0.5


class TestUpdateCompleteness:
    def test_all_addressed(self, instance_factory, step_response_factory):
        instance = instance_factory(n_claims=4, jsd_scores=[0.8, 0.9, 0.1, 0.2])
        step_b = step_response_factory(["high", "moderate", "high", "high"])
        score = update_completeness(step_b, instance)
        assert score == 1.0

    def test_none_addressed(self, instance_factory):
        instance = instance_factory(n_claims=3, jsd_scores=[0.8, 0.9, 0.7])
        score = update_completeness({"claim_level_assessments": []}, instance)
        assert score == 0.0


class TestUncertaintyLocalization:
    def test_well_localized(self, instance_factory, step_response_factory):
        """Uncertain on high-JSD claims → high score."""
        instance = instance_factory(
            n_claims=5, jsd_scores=[0.9, 0.8, 0.1, 0.1, 0.7],
        )
        # Low confidence on high-JSD, high confidence on low-JSD
        step_b = step_response_factory(["low", "low", "very_high", "very_high", "low"])
        score = uncertainty_localization(step_b, instance)
        assert score > 0.5


class TestConsensusConvergence:
    def test_moves_toward_consensus(self, instance_factory, step_response_factory):
        instance = instance_factory(n_claims=3)
        consensus = {"claim_confidences": {"C1": 0.80, "C2": 0.35, "C3": 0.55}}
        # A is far from consensus, B is closer
        step_a = step_response_factory(["very_low", "very_high", "very_high"])
        step_b = step_response_factory(["high", "low", "moderate"])
        score = consensus_convergence(step_a, step_b, consensus, instance)
        assert score > 0.5


class TestEpistemicCowardice:
    def test_all_moderate_is_cowardly(self, step_response_factory):
        """Everything at moderate → high cowardice score (bad, negatively oriented)."""
        step_b = step_response_factory(["moderate"] * 5)
        score = epistemic_cowardice_score(step_b)
        assert score > 0.8

    def test_varied_confidence_is_courageous(self, step_response_factory):
        step_b = step_response_factory(["very_high", "low", "very_high", "very_low", "high"])
        score = epistemic_cowardice_score(step_b)
        assert score < 0.3


class TestResistanceAppropriateness:
    def test_stable_on_easy_update_on_hard(self, instance_factory, step_response_factory):
        instance = instance_factory(
            n_claims=4, jsd_scores=[0.1, 0.1, 0.9, 0.9],
        )
        step_a = step_response_factory(["high", "high", "high", "high"])
        # No change on easy (low JSD), change on hard (high JSD)
        step_b = step_response_factory(["high", "high", "moderate", "low"])
        score = resistance_appropriateness(step_a, step_b, instance)
        assert score > 0.5


class TestSynthesisNecessity:
    def test_unique_synthesis_scores_high(self):
        step_b = {"overall_assessment": "This is a completely novel integrated analysis "
                  "combining multiple viewpoints on pharmacogenomic risk factors"}
        analysts = [
            {"response": {"overall_assessment": "The patient shows signs of infection with fever"}},
            {"response": {"overall_assessment": "Laboratory results indicate renal compromise"}},
        ]
        score = synthesis_necessity_score(step_b, analysts)
        assert score > 0.7

    def test_parroting_scores_low(self):
        text = "The patient shows signs of infection with fever and elevated WBC"
        step_b = {"overall_assessment": text}
        analysts = [{"response": {"overall_assessment": text}}]
        score = synthesis_necessity_score(step_b, analysts)
        assert score < 0.3


class TestInstructionDependenceGap:
    def test_identical_responses_zero_gap(self, instance_factory, step_response_factory):
        instance = instance_factory(n_claims=3)
        same = step_response_factory(["high", "moderate", "low"])
        score = instruction_dependence_gap(same, same, instance)
        assert score == 0.0

    def test_different_responses_positive_gap(self, instance_factory, step_response_factory):
        instance = instance_factory(n_claims=3)
        full = step_response_factory(["very_high", "very_high", "very_high"])
        minimal = step_response_factory(["low", "low", "low"])
        score = instruction_dependence_gap(full, minimal, instance)
        assert score > 0.5
