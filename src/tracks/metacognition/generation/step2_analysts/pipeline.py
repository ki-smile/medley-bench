"""Step 2: Analyst pipeline — 5 curated models analyze each case.

The curated ensemble of 5 maximally differentiated models produces
independent analyses. One model per instance is held out for jackknife
judging (not shown to benchmarked model, acts as judge later).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.core.db import get_db
from src.core.parsing import parse_json_response, get_claim_conf
from src.core.providers import get_provider
from src.admin.db.models import (
    AnalystResponse, Claim,
    insert_analyst_response, insert_claim,
    get_case,
)

logger = logging.getLogger(__name__)

# 5 curated analyst models, selected for maximum differentiation
ANALYST_MODELS = [
    {
        "model_id": "claude-sonnet-4",
        "profile": "Premium US · conservative clinical reasoning",
        "region": "US",
    },
    {
        "model_id": "gemini-2.5-pro",
        "profile": "Premium US · novel hypotheses, slow-burn patterns",
        "region": "US",
    },
    {
        "model_id": "qwen-2.5-72b",
        "profile": "Open CN · Chinese medical priors, different prevalence",
        "region": "CN",
    },
    {
        "model_id": "meditron-7b",
        "profile": "Medical specialist · protocol-driven, small scale",
        "region": "US",
    },
    {
        "model_id": "llama-3.3-70b",
        "profile": "Open US · broad generalist baseline",
        "region": "US",
    },
]

ANALYST_PROMPT = """\
You are an expert analyst. Carefully analyze the following case and provide \
your independent assessment.

═══════════════════════════════════════════════════════
CASE
═══════════════════════════════════════════════════════
{vignette}

═══════════════════════════════════════════════════════
KEY CLAIMS TO EVALUATE
═══════════════════════════════════════════════════════
{claims}

═══════════════════════════════════════════════════════
CONFIDENCE SCALE
═══════════════════════════════════════════════════════
  very_high (90-100%) | high (70-89%) | moderate (50-69%) |
  low (30-49%) | very_low (0-29%)

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only)
═══════════════════════════════════════════════════════
{{
  "overall_assessment": "your assessment",
  "claim_level_assessments": [
    {{
      "claim_id": "C1",
      "confidence": "very_high | high | moderate | low | very_low",
      "reasoning": "your reasoning"
    }}
  ]
}}"""


def _get_held_out_index(case_id: str, n_analysts: int = 5) -> int:
    """Deterministic held-out model index based on case_id hash."""
    return hash(case_id) % n_analysts


async def run_single_analyst(
    model_id: str,
    vignette: str,
    claims: list[dict],
) -> dict:
    """Run a single analyst model on a case."""
    provider = get_provider(model_id)
    claims_text = "\n".join(
        f"  {c['claim_id']}: {c['claim_text']}" for c in claims
    )
    prompt = ANALYST_PROMPT.format(vignette=vignette, claims=claims_text)
    raw = await provider.complete(prompt)
    return parse_json_response(raw)


async def run_all_analysts(
    case_id: str,
    vignette: str,
    claims: list[dict],
    db_path: Path,
    analysts: list[dict] | None = None,
) -> list[dict]:
    """Run all 5 analysts on a case with jackknife hold-out."""
    if analysts is None:
        analysts = ANALYST_MODELS

    held_out_idx = _get_held_out_index(case_id, len(analysts))
    results = []

    for i, analyst in enumerate(analysts):
        is_held_out = (i == held_out_idx)
        model_id = analyst["model_id"]

        logger.info(
            "Running analyst %s on %s%s",
            model_id, case_id, " (held out)" if is_held_out else "",
        )

        try:
            response = await run_single_analyst(model_id, vignette, claims)

            with get_db(db_path) as conn:
                insert_analyst_response(conn, AnalystResponse(
                    case_id=case_id,
                    model_id=model_id,
                    response=response,
                    jackknife_left_out=is_held_out,
                ))

            results.append({
                "model_id": model_id,
                "profile": analyst["profile"],
                "response": response,
                "held_out": is_held_out,
            })
        except Exception as e:
            logger.error("Analyst %s failed on %s: %s", model_id, case_id, e)
            results.append({"model_id": model_id, "_error": str(e)})

    return results


def extract_claims(analyst_results: list[dict], claims: list[dict]) -> list[dict]:
    """Extract claim-level statistics from analyst responses.

    Computes majority_strength (how many analysts support each claim)
    and Jensen-Shannon divergence across analysts per claim.
    """
    import numpy as np

    enriched_claims = []
    for claim in claims:
        cid = claim["claim_id"]
        ct = claim.get("claim_text", "")

        # Collect confidences from non-held-out analysts
        confidences = []
        for result in analyst_results:
            if result.get("held_out") or "_error" in result:
                continue
            resp = result.get("response", {})
            conf = get_claim_conf(resp, claim_id=cid, claim_text=ct)
            if conf is not None:
                confidences.append(conf)

        # Majority strength: count of analysts with confidence > 0.5
        majority_strength = sum(1 for c in confidences if c > 0.5)

        # JSD: variance-based proxy (true JSD requires distributions)
        jsd_score = float(np.std(confidences)) if len(confidences) > 1 else 0.0

        enriched_claims.append({
            **claim,
            "majority_strength": majority_strength,
            "jsd_score": round(jsd_score, 4),
        })

    return enriched_claims


def save_claims(db_path: Path, case_id: str, claims: list[dict]) -> None:
    """Save extracted claims to database."""
    with get_db(db_path) as conn:
        for claim in claims:
            insert_claim(conn, Claim(
                case_id=case_id,
                claim_id=claim["claim_id"],
                claim_text=claim["claim_text"],
                majority_strength=claim.get("majority_strength"),
                jsd_score=claim.get("jsd_score"),
            ))


async def process_case(
    case_id: str,
    db_path: Path,
    analysts: list[dict] | None = None,
) -> list[dict]:
    """Full Step 2 pipeline for a single case.

    1. Load case from DB
    2. Run all analysts
    3. Extract claims with statistics
    4. Save claims to DB
    """
    with get_db(db_path) as conn:
        case = get_case(conn, case_id)
        if case is None:
            raise ValueError(f"Case {case_id} not found")
        if not case.vignette:
            raise ValueError(f"Case {case_id} has no vignette (run Step 1 first)")

    # Get initial claims from designer responses
    with get_db(db_path) as conn:
        from src.admin.db.models import get_claims
        existing_claims = get_claims(conn, case_id)

    claims = [
        {"claim_id": c.claim_id, "claim_text": c.claim_text}
        for c in existing_claims
    ]

    # If no claims extracted from Step 1 yet, use placeholder
    if not claims:
        claims = [
            {"claim_id": f"C{i+1}", "claim_text": f"Claim {i+1}"}
            for i in range(5)
        ]

    # Run analysts
    results = await run_all_analysts(case_id, case.vignette, claims, db_path, analysts)

    # Extract claim statistics
    enriched = extract_claims(results, claims)
    save_claims(db_path, case_id, enriched)

    return results
