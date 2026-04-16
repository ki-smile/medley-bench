"""Validate all weight constraints across tiers and dimensions."""
import pytest
from src.tracks.metacognition.scoring.aggregation import (
    TIER_DEFINITIONS,
    DIMENSION_WEIGHTS,
    SUB_WEIGHTS,
    get_effective_dimension_weights,
    get_effective_tier_weights,
)


class TestTierWeightConstraints:
    def test_tier_weights_sum_to_one(self):
        total = sum(t["weight"] for t in TIER_DEFINITIONS.values())
        assert abs(total - 1.0) < 0.001

    def test_sub_weights_sum_to_one_per_tier(self):
        for tier_name, tier_def in TIER_DEFINITIONS.items():
            total = sum(s["weight"] for s in tier_def["sub_measures"].values())
            assert abs(total - 1.0) < 0.02, f"Tier '{tier_name}': {total}"

    def test_all_sub_measures_have_source(self):
        for tier_name, tier_def in TIER_DEFINITIONS.items():
            for measure, config in tier_def["sub_measures"].items():
                assert config["source"] in ("computed", "judged"), (
                    f"{tier_name}/{measure} has invalid source: {config['source']}"
                )


class TestDimensionWeightConstraints:
    def test_dimension_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_sub_weights_sum_to_one_per_dimension(self):
        for dim_name, subs in SUB_WEIGHTS.items():
            total = sum(subs.values())
            assert abs(total - 1.0) < 0.02, f"Dim '{dim_name}': {total}"

    def test_all_dimensions_have_sub_weights(self):
        for dim in DIMENSION_WEIGHTS:
            assert dim in SUB_WEIGHTS, f"Dimension '{dim}' missing from SUB_WEIGHTS"


class TestAdaptiveWeightsConsistency:
    """Test that adaptive weighting preserves total = 1.0 under all scenarios."""

    @pytest.mark.parametrize("kappa", [0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.10, 0.0])
    def test_dimension_weights_sum_to_one(self, kappa):
        meta = {"judge_reliability": {"sophistry_kappa": kappa}}
        weights = get_effective_dimension_weights(meta)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"κ={kappa}: dimension weights sum to {total}"

    @pytest.mark.parametrize("kappa", [0.70, 0.50, 0.30])
    def test_tier_weights_sum_to_one(self, kappa):
        meta = {"judge_reliability": {"sophistry_kappa": kappa}}
        weights = get_effective_tier_weights(meta)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"κ={kappa}: tier weights sum to {total}"

    def test_no_kappa_defaults_correctly(self):
        weights = get_effective_dimension_weights({})
        assert weights["digital_sophistry"] == 0.10
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestNoDoubleCounting:
    """Ensure no measure appears in multiple tiers except the intentionally shared ones.

    `capitulation_quality` and `normative_vs_informational` are intentionally
    scored in both Tier 2 (social_robustness) and Tier 3 (epistemic_articulation):
    they carry signal for both social-pressure response and the quality of the
    reasoning the model produces under that pressure. The two tiers weight them
    differently and combine via geometric mean, so the overlap is not additive
    double-counting.
    """

    INTENTIONAL_CROSS_TIER = frozenset({
        "capitulation_quality",
        "normative_vs_informational",
    })

    def test_no_unintended_cross_tier_measures(self):
        measure_tiers: dict[str, str] = {}
        for tier_name, tier_def in TIER_DEFINITIONS.items():
            for measure in tier_def["sub_measures"]:
                if measure in measure_tiers and measure not in self.INTENTIONAL_CROSS_TIER:
                    pytest.fail(
                        f"Measure '{measure}' appears in both "
                        f"'{measure_tiers[measure]}' and '{tier_name}' "
                        f"but is not listed as intentional cross-tier."
                    )
                measure_tiers[measure] = tier_name
