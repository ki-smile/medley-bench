"""Validate that all computed measures produce scores in [0, 1]."""
import numpy as np
import pytest
from src.tracks.metacognition.scoring.measures import compute_all_computed_measures


class TestScoreBounds:
    def test_all_measures_bounded(self, instance_factory, step_response_factory, sample_consensus):
        """All computed measures should be in [0, 1] for random inputs."""
        rng = np.random.default_rng(123)
        labels = ["very_high", "high", "moderate", "low", "very_low"]

        for _ in range(20):
            n = int(rng.integers(3, 8))
            ms = rng.integers(0, 5, n).tolist()
            jsd = (rng.random(n) * 0.8).tolist()
            instance = instance_factory(n_claims=n, majority_strengths=ms, jsd_scores=jsd)
            confs = lambda: rng.choice(labels, n).tolist()

            measures = compute_all_computed_measures(
                step_a=step_response_factory(confs()),
                step_b_private=step_response_factory(confs()),
                step_b_social=step_response_factory(confs()),
                instance=instance,
                consensus=sample_consensus,
            )

            for name, value in measures.items():
                if isinstance(value, (int, float)):
                    assert 0.0 <= value <= 1.0, (
                        f"Measure '{name}' = {value} is out of bounds"
                    )

    def test_ideal_pattern_reasonable(self, instance_factory, three_step_ideal, sample_consensus):
        """Ideal three-step pattern should produce reasonable scores."""
        instance = instance_factory(n_claims=5, majority_strengths=[4, 3, 2, 1, 4])
        measures = compute_all_computed_measures(
            step_a=three_step_ideal["step_a"],
            step_b_private=three_step_ideal["step_b_private"],
            step_b_social=three_step_ideal["step_b_social"],
            instance=instance,
            consensus=sample_consensus,
        )
        # private_vs_social_delta should be moderate-to-high (has both self and social revision)
        assert 0.3 <= measures["private_vs_social_delta"] <= 0.9

    def test_sycophant_low_private_delta(self, sycophant_comparison_instance, three_step_sycophant, sample_consensus):
        """Sycophantic pattern: no self-revision, all social -> low private_vs_social_delta."""
        instance = sycophant_comparison_instance
        measures = compute_all_computed_measures(
            step_a=three_step_sycophant["step_a"],
            step_b_private=three_step_sycophant["step_b_private"],
            step_b_social=three_step_sycophant["step_b_social"],
            instance=instance,
            consensus=sample_consensus,
        )
        assert measures["private_vs_social_delta"] < 0.15
