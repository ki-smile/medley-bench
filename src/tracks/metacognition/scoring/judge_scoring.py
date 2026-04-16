"""Pre-computed judge score loading for MEDLEY-BENCH.

Loads judge scores from the exported precomputed_judge_scores.json.
These are generated during Part 1 judge pre-computation (Pass 2)
and used by Part 2 during benchmark scoring.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PRECOMPUTED_JUDGES: dict[str, dict] = {}


def load_judge_scores(path: Path | str | None = None) -> None:
    """Load pre-computed judge scores from JSON file. Called once at startup."""
    global PRECOMPUTED_JUDGES
    if path is None:
        path = Path("data") / "export" / "precomputed_judge_scores.json"
    path = Path(path)
    if path.exists():
        with open(path) as f:
            PRECOMPUTED_JUDGES = json.load(f)
        logger.info("Loaded judge scores for %d instance-model pairs", len(PRECOMPUTED_JUDGES))
    else:
        logger.warning("Judge scores file not found: %s", path)


def get_judge_scores(instance_id: str, model_name: str) -> dict:
    """Get judged measure scores for a specific instance + model pair.

    Returns empty dict if no pre-computed scores available.
    """
    key = f"{instance_id}_{model_name}"
    entry = PRECOMPUTED_JUDGES.get(key, {})
    return entry.get("judged_measures", {})


def get_judge_metadata(instance_id: str, model_name: str) -> dict | None:
    """Get full judge metadata including judge model ID and reasoning."""
    key = f"{instance_id}_{model_name}"
    return PRECOMPUTED_JUDGES.get(key)
