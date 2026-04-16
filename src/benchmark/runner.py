"""Standalone benchmark runner for MEDLEY-BENCH.

Runs the three-step benchmark locally (not on Kaggle) against any model
accessible via the provider abstraction. Saves results incrementally.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from src.core.providers import get_provider
from src.core.parsing import parse_json_response
from src.tracks.metacognition.prompts.step_a import build_prompt as build_step_a
from src.tracks.metacognition.prompts.step_b_private import build_prompt as build_step_b_private
from src.tracks.metacognition.prompts.step_b_social import build_prompt as build_step_b_social
from src.tracks.metacognition.prompts.step_b_minimal import build_prompt as build_step_b_minimal
from src.tracks.metacognition.scoring.measures import compute_all_computed_measures
from src.tracks.metacognition.scoring.aggregation import compute_tier_scores, compute_all_dimension_scores, compute_total_score
from src.tracks.metacognition.scoring.known_answer import score_known_answer_instance
from src.tracks.metacognition.scoring.judge import call_judge_v2
from src.tracks.metacognition.tasks import load_benchmark_data

logger = logging.getLogger(__name__)


class ProviderLLMShim:
    """Wraps an async LLMProvider into a sync .prompt() interface for run_instance."""

    def __init__(self, provider):
        self._provider = provider
        self.model_name = provider.model_name

    def prompt(self, text: str) -> str:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._provider.complete(text)).result()
        return asyncio.run(self._provider.complete(text))


async def run_instance_async(
    provider,
    instance: dict,
    consensus: dict,
    precomputed_judges: dict,
    precomputed_step_a: str | None = None,
    judge_client=None,
    judge_model: str | None = None,
) -> dict:
    """Run three-step benchmark on a single instance with an async provider."""
    instance_id = instance["instance_id"]
    model_name = provider.model_name

    # Step A — reuse analyst collection response if available (saves one API call)
    if precomputed_step_a:
        raw_a = precomputed_step_a
        # Ensure raw_a is a string (DB may store as JSON dict/list)
        if not isinstance(raw_a, str):
            raw_a = json.dumps(raw_a)
        step_a = parse_json_response(raw_a)
        logger.debug("[%s] %s: reusing precomputed Step A", model_name, instance_id)
    else:
        prompt_a = build_step_a(vignette=instance["vignette"])
        raw_a = await provider.complete(prompt_a)
        step_a = parse_json_response(raw_a)

    # Step B-Private
    prompt_bp = build_step_b_private(step_a_response=raw_a, vignette=instance["vignette"])
    raw_bp = await provider.complete(prompt_bp)
    step_b_private = parse_json_response(raw_bp)

    # Step B-Social
    analyst_text = json.dumps(instance.get("ensemble_outputs", []), indent=2)
    consensus_text = json.dumps(consensus, indent=2)
    prompt_bs = build_step_b_social(
        step_a_response=raw_a, vignette=instance["vignette"],
        analyst_outputs=analyst_text, jackknifed_consensus=consensus_text,
    )
    raw_bs = await provider.complete(prompt_bs)
    step_b_social = parse_json_response(raw_bs)

    # Optional probes
    step_b_minimal = None
    if instance.get("is_minimal_instruction"):
        prompt_bm = build_step_b_minimal(
            step_a_response=raw_a, vignette=instance["vignette"],
            analyst_outputs=analyst_text,
        )
        raw_bm = await provider.complete(prompt_bm)
        step_b_minimal = parse_json_response(raw_bm)

    # Compute measures
    computed = compute_all_computed_measures(
        step_a=step_a, step_b_private=step_b_private, step_b_social=step_b_social,
        instance=instance, consensus=consensus,
        analyst_outputs=instance.get("ensemble_outputs"),
        step_b_minimal=step_b_minimal,
    )

    # Judge scores — prefer precomputed, fall back to live judge if configured.
    judge_key = f"{instance_id}_{model_name}"
    judged = precomputed_judges.get(judge_key, {}).get("judged_measures", {})
    if not judged and judge_client is not None and judge_model:
        try:
            judged = await asyncio.to_thread(
                call_judge_v2,
                raw_a=raw_a,
                raw_bs=raw_bs,
                vignette=instance.get("vignette", ""),
                ensemble_outputs=instance.get("ensemble_outputs", []),
                key_claims=instance.get("key_claims", []),
                client=judge_client,
                model=judge_model,
                is_known_answer=instance.get("is_known_answer", False),
            )
        except Exception as e:
            logger.error("[%s] %s: live judge failed: %s", model_name, instance_id, e)
            judged = {}

    # Aggregate
    tier_scores = compute_tier_scores(computed, judged)
    dim_scores = compute_all_dimension_scores(computed, judged)
    total = compute_total_score(tier_scores)

    # Known-answer scoring
    ka_scoring = None
    if instance.get("is_known_answer"):
        ka_scoring = score_known_answer_instance(instance_id, step_a, step_b_social, instance)

    return {
        "instance_id": instance_id,
        "model": model_name,
        "domain": instance.get("domain"),
        "difficulty_tier": instance.get("difficulty_tier"),
        "total_score": total,
        "tier_scores": {k: v["score"] for k, v in tier_scores.items()},
        "dimension_scores": dim_scores,
        "private_vs_social_delta": computed.get("private_vs_social_delta"),
        "computed": {k: v for k, v in computed.items() if isinstance(v, (int, float))},
        "known_answer_scoring": ka_scoring,
        # Save raw responses for post-hoc judge scoring
        "raw_responses": {
            "step_a": raw_a[:5000] if isinstance(raw_a, str) else str(raw_a)[:5000],
            "step_b_private": raw_bp[:5000] if isinstance(raw_bp, str) else str(raw_bp)[:5000],
            "step_b_social": raw_bs[:5000] if isinstance(raw_bs, str) else str(raw_bs)[:5000],
        },
    }


def _load_analyst_responses(db_path: Path, model_id: str) -> dict[str, str]:
    """Load precomputed Step A responses from analyst collection in DB.

    Returns: {case_id: response_text}
    """
    import sqlite3
    if not db_path or not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT case_id, response FROM analyst_responses WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        conn.close()
        result = {row[0]: row[1] for row in rows}
        logger.info("Loaded %d precomputed Step A responses for %s", len(result), model_id)
        return result
    except Exception as e:
        logger.warning("Could not load analyst responses from DB: %s", e)
        return {}


async def run_benchmark(
    model_ids: list[str],
    data_dir: str | Path | None = None,
    output_dir: str | Path = "results",
    domains: list[str] | None = None,
    db_path: str | Path | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    n_instances: int | None = None,
) -> dict:
    """Run full benchmark for given models, saving results incrementally.

    If db_path is provided, reuses analyst collection responses as Step A
    (saves 130 API calls per model).

    If judge_model is provided, a live v2 judge is called on each instance
    whenever a precomputed judge score is unavailable. judge_base_url must
    point at an OpenAI-compatible endpoint (e.g. Gemini's OpenAI-compat
    endpoint, OpenRouter, or http://localhost:11434/v1 for Ollama).

    If n_instances is set, only the first N instances per domain are run
    (useful for smoke tests).

    Returns: {model_id: [instance_results]}
    """
    if data_dir is None:
        from data import get_default_data_dir
        data_dir = get_default_data_dir()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path) if db_path else None

    judge_client = None
    if judge_model:
        from openai import OpenAI
        judge_client = OpenAI(
            base_url=judge_base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=judge_api_key or "unused",
        )
        logger.info("Live judge enabled: %s @ %s", judge_model, judge_base_url or "(default Gemini endpoint)")

    data = load_benchmark_data(data_dir)
    if domains is None:
        domains = list(data["instances"].keys())

    all_instances = []
    for domain in domains:
        instances = data["instances"].get(domain, [])
        if n_instances is not None:
            instances = instances[:n_instances]
        for inst in instances:
            consensus = data["consensus"].get(domain, {}).get(inst["instance_id"], {})
            all_instances.append((inst, consensus))

    logger.info("Benchmark: %d models × %d instances", len(model_ids), len(all_instances))
    all_results = {}

    for model_id in model_ids:
        provider = get_provider(model_id)
        model_results = []
        output_file = output_dir / f"{model_id.replace('/', '_')}.json"

        # Load precomputed Step A from analyst collection (saves 130 API calls)
        step_a_cache = _load_analyst_responses(db_path, model_id) if db_path else {}

        # Load existing results (resume support)
        existing_ids = set()
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
            model_results = existing
            existing_ids = {r["instance_id"] for r in existing}
            logger.info("Resuming %s: %d already done", model_id, len(existing_ids))

        t0 = time.time()
        for i, (instance, consensus) in enumerate(all_instances):
            if instance["instance_id"] in existing_ids:
                continue

            instance_id = instance["instance_id"]
            # instance_id may be like "MED_001", case_id in DB is the same
            precomputed_a = step_a_cache.get(instance_id)
            if precomputed_a:
                logger.debug("[%s] %d/%d %s (Step A precomputed)", model_id, i + 1, len(all_instances), instance_id)
            else:
                logger.info("[%s] %d/%d %s", model_id, i + 1, len(all_instances), instance_id)
            try:
                result = await run_instance_async(
                    provider, instance, consensus, data.get("precomputed_judges", {}),
                    precomputed_step_a=precomputed_a,
                    judge_client=judge_client,
                    judge_model=judge_model,
                )
                model_results.append(result)

                # Save incrementally
                with open(output_file, "w") as f:
                    json.dump(model_results, f, indent=2)

            except Exception as e:
                logger.error("[%s] %s failed: %s", model_id, instance["instance_id"], e)
                model_results.append({
                    "instance_id": instance["instance_id"],
                    "model": model_id,
                    "error": str(e),
                })

        elapsed = time.time() - t0
        logger.info("[%s] Done: %d instances in %.1f min", model_id, len(model_results), elapsed / 60)
        all_results[model_id] = model_results

    return all_results


def build_leaderboard_from_results(results_dir: Path) -> list[dict]:
    """Build leaderboard from result JSON files in a directory."""
    import numpy as np

    entries = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results = json.load(fh)
        if not results:
            continue

        model = results[0].get("model", f.stem)
        valid = [r for r in results if "error" not in r and r.get("total_score") is not None]
        if not valid:
            continue

        entries.append({
            "model": model,
            "total": round(float(np.mean([r["total_score"] for r in valid])), 4),
            "pvsd": round(float(np.mean([r.get("private_vs_social_delta", 0.5) for r in valid])), 4),
            "n_instances": len(valid),
            "n_errors": len(results) - len(valid),
            "tiers": {
                tier: round(float(np.mean([
                    r.get("tier_scores", {}).get(tier, 0.5) for r in valid
                ])), 4)
                for tier in ["reflective_updating", "social_robustness", "epistemic_articulation"]
            },
        })

    entries.sort(key=lambda e: e["total"], reverse=True)
    return entries
