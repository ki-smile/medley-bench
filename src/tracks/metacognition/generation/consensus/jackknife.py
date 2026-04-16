"""Jackknife consensus building for MEDLEY-BENCH.

Builds consensus from analyst responses using leave-one-out jackknife.
For each claim, the consensus is the median confidence across non-held-out analysts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.core.db import get_db
from src.core.parsing import get_claim_conf

logger = logging.getLogger(__name__)


def build_consensus(
    analyst_results: list[dict],
    claims: list[dict],
) -> dict:
    """Build jackknifed consensus from analyst responses.

    For each claim, computes:
    - Median confidence across non-held-out analysts
    - Confidence range (min, max)
    - Agreement strength (1 - normalized std)

    Returns a consensus dict suitable for export.
    """
    claim_confidences = {}
    claim_details = {}

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

        if not confidences:
            claim_confidences[cid] = 0.5
            claim_details[cid] = {
                "median": 0.5, "mean": 0.5,
                "min": 0.5, "max": 0.5,
                "std": 0.0, "n_analysts": 0,
            }
            continue

        median_conf = float(np.median(confidences))
        claim_confidences[cid] = round(median_conf, 4)
        claim_details[cid] = {
            "median": round(median_conf, 4),
            "mean": round(float(np.mean(confidences)), 4),
            "min": round(float(min(confidences)), 4),
            "max": round(float(max(confidences)), 4),
            "std": round(float(np.std(confidences)), 4),
            "n_analysts": len(confidences),
        }

    return {
        "method": "jackknife_median",
        "claim_confidences": claim_confidences,
        "claim_details": claim_details,
    }


def save_consensus(db_path: Path, case_id: str, consensus: dict) -> None:
    """Save consensus to database."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO consensus (case_id, consensus_data, method)
               VALUES (?, ?, ?)""",
            (case_id, json.dumps(consensus), consensus.get("method", "jackknife")),
        )


async def build_and_save_consensus(
    case_id: str,
    db_path: Path,
) -> dict:
    """Full consensus pipeline: load analyst responses, build consensus, save.

    Loads analyst responses from DB, builds consensus, saves to DB.
    """
    from src.admin.db.models import get_analyst_responses, get_claims

    with get_db(db_path) as conn:
        analyst_responses = get_analyst_responses(conn, case_id)
        claims = get_claims(conn, case_id)

    # Convert to the format expected by build_consensus
    results = [
        {
            "model_id": ar.model_id,
            "response": ar.response,
            "held_out": ar.jackknife_left_out,
        }
        for ar in analyst_responses
    ]
    claim_dicts = [
        {"claim_id": c.claim_id, "claim_text": c.claim_text}
        for c in claims
    ]

    consensus = build_consensus(results, claim_dicts)
    save_consensus(db_path, case_id, consensus)

    logger.info("Consensus built for %s: %d claims", case_id, len(claim_dicts))
    return consensus
