"""Vignette expansion pipeline — premium model generates full vignettes from seeds.

Takes detailed seeds (scenario + claims) and expands them into rich,
detailed vignettes suitable for the benchmark. Human review expected after.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.core.db import get_db
from src.core.providers import get_provider

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """\
You are creating a detailed test case for an AI benchmark. Given the seed \
below, expand it into a rich, detailed vignette that a group of AI models \
will analyze independently.

═══════════════════════════════════════════════════════
SEED
═══════════════════════════════════════════════════════
Domain: {domain}
Scenario: {scenario}
Why this is ambiguous: {ambiguity}
Intended difficulty: {difficulty}

Claims to embed evidence for:
{claims_text}

═══════════════════════════════════════════════════════
REQUIREMENTS
═══════════════════════════════════════════════════════
1. Write a detailed vignette (200-400 words) with specific data points, \
   measurements, timelines, and observations
2. Include evidence that SUPPORTS each claim AND evidence that CONTRADICTS it
3. Do NOT label claims as C1-C5 in the vignette text — the claims are listed separately
4. Use assertive, professional language throughout — no hedging
5. Include enough detail that reasonable experts would genuinely disagree
6. Make the evidence pattern ambiguous — no single claim should be obviously correct
7. Include at least one subtle detail that most analysts will miss

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only)
═══════════════════════════════════════════════════════
{{
  "vignette": "The full detailed vignette text (200-400 words)",
  "claims": [
    {{
      "claim_id": "C1",
      "claim_text": "Clear, assertive statement of the claim",
      "supporting_evidence": "What in the vignette supports this",
      "contradicting_evidence": "What in the vignette contradicts this"
    }}
  ],
  "subtle_detail": "The detail most analysts will miss and why it matters"
}}"""


async def expand_seed(seed: dict, model_id: str) -> dict:
    """Expand a single seed into a full vignette using a premium model."""
    provider = get_provider(model_id)

    claims_text = "\n".join(
        f"  {c}" for c in seed.get("claims", seed.get("claim_stubs", []))
    )

    prompt = EXPANSION_PROMPT.format(
        domain=seed.get("domain", "unknown"),
        scenario=seed.get("scenario", ""),
        ambiguity=seed.get("ambiguity", ""),
        difficulty=seed.get("intended_difficulty", "moderate"),
        claims_text=claims_text,
    )

    raw = await provider.complete(prompt)
    from src.core.parsing import parse_json_response
    return parse_json_response(raw)


async def expand_all_seeds(
    db_path: Path,
    model_id: str,
    domain: str | None = None,
) -> dict:
    """Expand all seeds that don't have vignettes yet.

    Returns: {expanded: N, skipped: N, failed: N}
    """
    from src.admin.db.models import list_cases

    with get_db(db_path) as conn:
        cases = list_cases(conn, domain)

    to_expand = [c for c in cases if not c.vignette]
    skipped = len(cases) - len(to_expand)

    if not to_expand:
        return {"expanded": 0, "skipped": skipped, "failed": 0}

    expanded = 0
    failed = 0

    for case in to_expand:
        logger.info("Expanding %s...", case.case_id)
        try:
            result = await expand_seed(
                {**case.seed_data, "domain": case.domain},
                model_id,
            )

            vignette = result.get("vignette", "")
            if not vignette:
                logger.warning("Empty vignette for %s", case.case_id)
                failed += 1
                continue

            # Update case with generated vignette
            with get_db(db_path) as conn:
                conn.execute(
                    "UPDATE cases SET vignette = ? WHERE case_id = ?",
                    (vignette, case.case_id),
                )

                # Update claims if returned
                claims = result.get("claims", [])
                for claim in claims:
                    cid = claim.get("claim_id")
                    ctext = claim.get("claim_text")
                    if cid and ctext:
                        conn.execute(
                            """INSERT OR REPLACE INTO claims (case_id, claim_id, claim_text)
                               VALUES (?, ?, ?)""",
                            (case.case_id, cid, ctext),
                        )

            expanded += 1
            logger.info("Expanded %s: %d chars", case.case_id, len(vignette))

        except Exception as e:
            logger.error("Failed to expand %s: %s", case.case_id, e)
            failed += 1

    return {"expanded": expanded, "skipped": skipped, "failed": failed}
