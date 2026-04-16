"""Metacognition track v1.0 - Behavioral metacognition under social pressure.

Three-step protocol: Solo -> Self-Revision -> Social Revision
Two scores: MMS (tier aggregate) and MAS (ability mean)
130 instances, 5 domains, 35 models evaluated.
"""

__version__ = "1.0"
TRACK_NAME = "metacognition"
DEEPMIND_ABILITY = "Metacognition"
DOMAINS = ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]

# Versioning: dataset and scoring are versioned independently of the library.
# This ensures result comparability across library upgrades.
DATASET_VERSION = "1.0"    # Frozen instance data (vignettes, claims, analysts, consensus)
SCORING_VERSION = "1.0"    # Scoring measures, weights, judge rubric, aggregation

# What triggers a version bump:
#   DATASET_VERSION:
#     - 1.0 -> 1.1: add new instances (old instances unchanged)
#     - 1.0 -> 2.0: change existing instances (vignettes, claims, consensus)
#   SCORING_VERSION:
#     - 1.0 -> 1.1: add new measure, change weight, fix bug (minor)
#     - 1.0 -> 2.0: change protocol (e.g., 3-step -> 4-step), change tiers (major)
