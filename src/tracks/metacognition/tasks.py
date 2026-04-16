"""Kaggle benchmark task definitions for MEDLEY-BENCH.

Each domain gets one @kbench.task that executes the three-step flow:
Step A (solo) → Step B-Private (self-revision) → Step B-Social (under pressure).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.core.parsing import parse_json_response
from src.tracks.metacognition.prompts.step_a import build_prompt as build_step_a
from src.tracks.metacognition.prompts.step_b_private import build_prompt as build_step_b_private
from src.tracks.metacognition.prompts.step_b_social import build_prompt as build_step_b_social
from src.tracks.metacognition.prompts.step_b_minimal import build_prompt as build_step_b_minimal
from src.tracks.metacognition.scoring.measures import compute_all_computed_measures
from src.tracks.metacognition.scoring.aggregation import (
    compute_tier_scores, compute_all_dimension_scores, compute_total_score,
)
from src.tracks.metacognition.scoring.known_answer import score_known_answer_instance

logger = logging.getLogger(__name__)

DOMAINS = ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def load_benchmark_data(data_dir: Path | str | None = None) -> dict:
    """Load all exported benchmark data."""
    if data_dir is None:
        from data import get_default_data_dir
        data_dir = get_default_data_dir()
    data_dir = Path(data_dir)

    data = {
        "instances": {},
        "consensus": {},
        "metadata": _load_json(data_dir / "metadata.json"),
        "precomputed_judges": {},
    }

    for domain in DOMAINS:
        instances_path = data_dir / "instances" / f"{domain}.json"
        consensus_path = data_dir / "consensus" / f"{domain}.json"

        if instances_path.exists():
            data["instances"][domain] = _load_json(instances_path)
        if consensus_path.exists():
            data["consensus"][domain] = _load_json(consensus_path)

    # Judge scores (Pass 2)
    judge_path = data_dir / "precomputed_judge_scores.json"
    if judge_path.exists():
        data["precomputed_judges"] = _load_json(judge_path)

    return data


def run_instance(
    llm,
    instance: dict,
    consensus: dict,
    precomputed_judges: dict,
) -> dict:
    """Execute three-step benchmark on a single instance.

    Args:
        llm: Object with a .prompt(text) -> str method (Kaggle harness)
        instance: Instance dict from exported data
        consensus: Consensus dict for this instance
        precomputed_judges: Pre-computed judge scores

    Returns:
        Full result dict with scores, tier scores, dimension scores.
    """
    instance_id = instance["instance_id"]

    # ═══ STEP A: Solo Response ═════════════════════════════
    prompt_a = build_step_a(vignette=instance["vignette"])
    response_a_raw = llm.prompt(prompt_a)
    step_a = parse_json_response(response_a_raw)

    # ═══ STEP B-PRIVATE: Private Revision ══════════════════
    prompt_b_priv = build_step_b_private(
        step_a_response=response_a_raw,
        vignette=instance["vignette"],
    )
    response_b_priv_raw = llm.prompt(prompt_b_priv)
    step_b_private = parse_json_response(response_b_priv_raw)

    # ═══ STEP B-SOCIAL: Social Revision ════════════════════
    analyst_text = json.dumps(instance.get("ensemble_outputs", []), indent=2)
    consensus_text = json.dumps(consensus, indent=2)
    prompt_b_soc = build_step_b_social(
        step_a_response=response_a_raw,
        vignette=instance["vignette"],
        analyst_outputs=analyst_text,
        jackknifed_consensus=consensus_text,
    )
    response_b_soc_raw = llm.prompt(prompt_b_soc)
    step_b_social = parse_json_response(response_b_soc_raw)

    # ═══ OPTIONAL: Minimal-instruction probe ═══════════════
    step_b_minimal = None
    if instance.get("is_minimal_instruction"):
        prompt_b_min = build_step_b_minimal(
            step_a_response=response_a_raw,
            vignette=instance["vignette"],
            analyst_outputs=analyst_text,
        )
        step_b_minimal = parse_json_response(llm.prompt(prompt_b_min))

    # ═══ OPTIONAL: Dose-response probe ═════════════════════
    step_b_partial = None
    if instance.get("is_dose_response"):
        partial_analyst_text = json.dumps(
            instance.get("probe_ensemble_outputs", []), indent=2
        )
        prompt_b_dr = build_step_b_social(
            step_a_response=response_a_raw,
            vignette=instance["vignette"],
            analyst_outputs=partial_analyst_text,
            jackknifed_consensus=consensus_text,
        )
        step_b_partial = parse_json_response(llm.prompt(prompt_b_dr))

    # ═══ COMPUTED MEASURES ═════════════════════════════════
    computed = compute_all_computed_measures(
        step_a=step_a,
        step_b_private=step_b_private,
        step_b_social=step_b_social,
        instance=instance,
        consensus=consensus,
        analyst_outputs=instance.get("ensemble_outputs"),
        step_b_partial=step_b_partial,
        step_b_minimal=step_b_minimal,
    )

    # ═══ JUDGE SCORES ═════════════════════════════════════
    model_name = getattr(llm, "model_name", "unknown")
    judge_key = f"{instance_id}_{model_name}"
    judged = precomputed_judges.get(judge_key, {}).get("judged_measures", {})

    # ═══ AGGREGATION ═══════════════════════════════════════
    tier_scores = compute_tier_scores(computed, judged)
    dim_scores = compute_all_dimension_scores(computed, judged)
    total = compute_total_score(tier_scores)

    # ═══ KNOWN-ANSWER SCORING ══════════════════════════════
    ka_scoring = None
    if instance.get("is_known_answer"):
        ka_scoring = score_known_answer_instance(
            instance_id, step_a, step_b_social, instance,
        )

    return {
        "instance_id": instance_id,
        "domain": instance.get("domain"),
        "difficulty_tier": instance.get("difficulty_tier"),
        "is_known_answer": instance.get("is_known_answer", False),
        "is_trap": instance.get("is_trap", False),
        "total_score": total,
        "tier_scores": tier_scores,
        "dimension_scores": dim_scores,
        "computed": computed,
        "judged": judged,
        "known_answer_scoring": ka_scoring,
    }
