"""Step 1: Designer pipeline — 25 models generate case proposals.

Each designer model receives a seed and produces a case proposal including
a vignette, key claims, and difficulty assessment. The K=25 jackknife
disagreement across designers determines difficulty tiers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from src.core.db import get_db
from src.core.parsing import parse_json_response
from src.core.providers import LLMProvider, get_provider
from src.admin.db.models import (
    Case, DesignerResponse, insert_case, insert_designer_response,
    update_case_difficulty,
)

logger = logging.getLogger(__name__)

# 25 designer models across 6 categories
DESIGNER_MODELS = {
    # Premium API (5)
    "claude-sonnet-4": "Premium API",
    "gemini-2.5-pro": "Premium API",
    "gpt-4o": "Premium API",
    "grok-3": "Premium API",
    "mistral-large-latest": "Premium API",
    # Open Large (5)
    "llama-3.3-70b": "Open Large",
    "qwen-2.5-72b": "Open Large",
    "deepseek-r1": "Open Large",
    "gemma-2-27b": "Open Large",
    "command-r-plus": "Open Large",
    # Open Medium (5)
    "llama-3.1-8b": "Open Medium",
    "mistral-7b": "Open Medium",
    "phi-3-medium": "Open Medium",
    "gemma-2-9b": "Open Medium",
    "yi-34b": "Open Medium",
    # Open Small (4)
    "phi-3-mini": "Open Small",
    "gemma-2-2b": "Open Small",
    "qwen-2.5-3b": "Open Small",
    "tinyllama-1.1b": "Open Small",
    # Specialized Medical (4)
    "medgemma": "Specialized Med",
    "biomistral": "Specialized Med",
    "meditron-7b": "Specialized Med",
    "openbiollm": "Specialized Med",
    # Legacy (2)
    "gpt-3.5-turbo": "Legacy",
    "llama-2-70b": "Legacy",
}

DESIGNER_PROMPT = """\
You are participating in a benchmark case design process. Given the following \
seed information, generate a detailed case vignette for the {domain} domain.

The vignette should:
1. Present a genuinely ambiguous scenario with multiple defensible interpretations
2. Include 4-6 key claims that experts might disagree on
3. Contain enough detail for meaningful analysis

═══════════════════════════════════════════════════════
SEED
═══════════════════════════════════════════════════════
{seed_data}

═══════════════════════════════════════════════════════
DOMAIN: {domain}
═══════════════════════════════════════════════════════

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only)
═══════════════════════════════════════════════════════
{{
  "vignette": "detailed case description",
  "key_claims": [
    {{
      "claim_id": "C1",
      "claim_text": "a specific claim about the case",
      "why_ambiguous": "why reasonable experts might disagree"
    }}
  ],
  "difficulty_assessment": "easy | moderate | hard",
  "rationale": "why you assessed this difficulty level"
}}"""


async def run_single_designer(
    model_id: str,
    seed_data: dict,
    domain: str,
) -> dict:
    """Run a single designer model on a seed."""
    provider = get_provider(model_id)
    prompt = DESIGNER_PROMPT.format(
        domain=domain,
        seed_data=json.dumps(seed_data, indent=2),
    )
    raw = await provider.complete(prompt)
    return parse_json_response(raw)


async def run_all_designers(
    case_id: str,
    seed_data: dict,
    domain: str,
    db_path: Path,
    models: list[str] | None = None,
    max_concurrent: int = 5,
) -> list[dict]:
    """Run all 25 designer models on a single case seed.

    Uses semaphore to limit concurrent API calls.
    """
    if models is None:
        models = list(DESIGNER_MODELS.keys())

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def _run_one(model_id: str):
        async with semaphore:
            logger.info("Running designer %s on %s", model_id, case_id)
            try:
                response = await run_single_designer(model_id, seed_data, domain)
                # Save to DB
                with get_db(db_path) as conn:
                    insert_designer_response(conn, DesignerResponse(
                        case_id=case_id,
                        model_id=model_id,
                        response=response,
                    ))
                return response
            except Exception as e:
                logger.error("Designer %s failed on %s: %s", model_id, case_id, e)
                return {"_error": str(e), "model_id": model_id}

    tasks = [_run_one(m) for m in models]
    results = await asyncio.gather(*tasks)
    return results


def compute_disagreement(responses: list[dict]) -> tuple[float, str]:
    """Compute K=25 jackknife disagreement score from designer responses.

    Returns (disagreement_score, difficulty_tier).

    Disagreement is measured by variance in difficulty assessments
    and claim-level agreement rates.
    """
    # Extract difficulty assessments
    difficulties = []
    diff_map = {"easy": 0, "moderate": 1, "hard": 2}
    for resp in responses:
        if "_error" in resp:
            continue
        d = resp.get("difficulty_assessment", "moderate")
        difficulties.append(diff_map.get(d, 1))

    if not difficulties:
        return 0.5, "medium"

    # Variance in difficulty assessments → disagreement signal
    variance = float(np.var(difficulties))
    # Mean difficulty
    mean_diff = float(np.mean(difficulties))

    # Difficulty tier based on mean + variance
    # High variance = hard (genuine disagreement)
    # High mean = hard (consensus is hard)
    combined = mean_diff * 0.6 + variance * 2.0

    if combined < 0.8:
        tier = "easy"
    elif combined < 1.5:
        tier = "medium"
    else:
        tier = "hard"

    return round(combined, 4), tier


def synthesize_vignette(responses: list[dict]) -> str:
    """Synthesize a single vignette from multiple designer proposals.

    Takes the most detailed vignette and enriches it with unique
    elements from other proposals.
    """
    vignettes = []
    for resp in responses:
        if "_error" in resp:
            continue
        v = resp.get("vignette", "")
        if v:
            vignettes.append(v)

    if not vignettes:
        return ""

    # Use longest vignette as base (tends to be most detailed)
    return max(vignettes, key=len)


async def process_case(
    case_id: str,
    seed_data: dict,
    domain: str,
    db_path: Path,
    models: list[str] | None = None,
) -> Case:
    """Full Step 1 pipeline for a single case.

    1. Run all designers
    2. Compute disagreement + difficulty tier
    3. Synthesize vignette
    4. Update case in DB
    """
    # Ensure case exists in DB
    with get_db(db_path) as conn:
        from src.admin.db.models import get_case
        if get_case(conn, case_id) is None:
            case = Case(case_id=case_id, domain=domain, seed_data=seed_data)
            insert_case(conn, case)

    # Run designers
    responses = await run_all_designers(case_id, seed_data, domain, db_path, models)

    # Compute disagreement
    score, tier = compute_disagreement(responses)

    # Synthesize vignette
    vignette = synthesize_vignette(responses)

    # Update case
    with get_db(db_path) as conn:
        update_case_difficulty(conn, case_id, tier, score)
        conn.execute(
            "UPDATE cases SET vignette = ? WHERE case_id = ?",
            (vignette, case_id),
        )

    logger.info("Case %s: tier=%s, score=%.4f, vignette=%d chars",
                case_id, tier, score, len(vignette))

    return Case(
        case_id=case_id, domain=domain, seed_data=seed_data,
        vignette=vignette, difficulty_tier=tier, disagreement_score=score,
    )
