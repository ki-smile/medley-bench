"""Incremental model response collection for MEDLEY-BENCH.

Allows adding model responses one model at a time, skipping already-done
combinations. Supports gradual dataset building with local models (ollama).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.core.db import get_db
from src.core.parsing import parse_json_response
from src.core.providers import get_provider
from src.admin.db.models import (
    DesignerResponse, AnalystResponse,
    insert_designer_response, insert_analyst_response,
    list_cases,
)

def _log_failure(db_path: Path, case_id: str, model_id: str, role: str, error: str):
    """Log a collection failure for later retry."""
    with get_db(db_path) as conn:
        conn.execute("""
            INSERT INTO collection_failures (case_id, model_id, role, error, attempt_count, last_attempt)
            VALUES (?, ?, ?, ?, 1, datetime('now'))
            ON CONFLICT(case_id, model_id, role) DO UPDATE SET
                error = excluded.error,
                attempt_count = attempt_count + 1,
                last_attempt = datetime('now')
        """, (case_id, model_id, role, str(error)[:500]))

def _mark_resolved(db_path: Path, case_id: str, model_id: str, role: str):
    """Mark a previously failed collection as resolved (succeeded on retry)."""
    with get_db(db_path) as conn:
        conn.execute("""
            UPDATE collection_failures SET resolved = 1
            WHERE case_id = ? AND model_id = ? AND role = ?
        """, (case_id, model_id, role))

def get_failure_summary(db_path: Path) -> dict:
    """Get summary of collection failures for monitoring."""
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT model_id, role, COUNT(*) as failures,
                   SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved
            FROM collection_failures
            GROUP BY model_id, role
            ORDER BY model_id, role
        """).fetchall()
    return [{"model_id": r["model_id"], "role": r["role"],
             "failures": r["failures"], "resolved": r["resolved"]} for r in rows]

def get_unresolved_failures(db_path: Path, model_id: str = None, role: str = None) -> list[dict]:
    """Get unresolved failures for retry."""
    query = "SELECT case_id, model_id, role, error, attempt_count FROM collection_failures WHERE resolved = 0"
    params = []
    if model_id:
        query += " AND model_id = ?"
        params.append(model_id)
    if role:
        query += " AND role = ?"
        params.append(role)
    query += " ORDER BY attempt_count ASC"
    with get_db(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]
from src.admin.generation.step1_designers.pipeline import DESIGNER_PROMPT
from src.admin.generation.prompts.step_a_prompt import build_prompt as build_step_a

logger = logging.getLogger(__name__)


def get_collection_status(db_path: Path) -> dict:
    """Get a matrix of which models have responded to which cases.

    Returns: {
        "designers": {case_id: [model_ids...]},
        "analysts": {case_id: [model_ids...]},
        "summary": {model_id: {"designer": N, "analyst": N}},
    }
    """
    with get_db(db_path) as conn:
        # Designer responses
        d_rows = conn.execute(
            "SELECT case_id, model_id FROM designer_responses"
        ).fetchall()
        designers = {}
        for r in d_rows:
            designers.setdefault(r["case_id"], []).append(r["model_id"])

        # Analyst responses
        a_rows = conn.execute(
            "SELECT case_id, model_id FROM analyst_responses"
        ).fetchall()
        analysts = {}
        for r in a_rows:
            analysts.setdefault(r["case_id"], []).append(r["model_id"])

        # Summary by model
        summary = {}
        for r in d_rows:
            summary.setdefault(r["model_id"], {"designer": 0, "analyst": 0})
            summary[r["model_id"]]["designer"] += 1
        for r in a_rows:
            summary.setdefault(r["model_id"], {"designer": 0, "analyst": 0})
            summary[r["model_id"]]["analyst"] += 1

        total_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    return {
        "designers": designers,
        "analysts": analysts,
        "summary": summary,
        "total_cases": total_cases,
    }


def _get_existing_pairs(db_path: Path, table: str) -> set[tuple[str, str]]:
    """Get set of (case_id, model_id) pairs already in a response table."""
    with get_db(db_path) as conn:
        rows = conn.execute(f"SELECT case_id, model_id FROM {table}").fetchall()
    return {(r["case_id"], r["model_id"]) for r in rows}


async def collect_designer_responses(
    db_path: Path,
    model_id: str,
    domain: str | None = None,
    max_concurrent: int = 4,
) -> dict:
    """Collect designer responses for a single model across all cases.

    Skips cases where this model already has a response.
    Returns: {completed: N, skipped: N, failed: N}
    """
    existing = _get_existing_pairs(db_path, "designer_responses")

    with get_db(db_path) as conn:
        cases = list_cases(conn, domain)

    to_run = [c for c in cases if (c.case_id, model_id) not in existing]
    skipped = len(cases) - len(to_run)

    if not to_run:
        return {"completed": 0, "skipped": skipped, "failed": 0, "total": len(cases)}

    provider = get_provider(model_id)

    # Pre-warm model
    logger.info("Pre-warming model %s...", model_id)
    try:
        await provider.complete("Say OK")
        logger.info("Model %s is warm", model_id)
    except Exception as e:
        logger.warning("Pre-warm failed for %s: %s (continuing anyway)", model_id, e)

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0
    total_to_run = len(to_run)

    async def _run_one(case):
        nonlocal completed, failed
        async with semaphore:
            try:
                prompt = DESIGNER_PROMPT.format(
                    domain=case.domain,
                    seed_data=json.dumps(case.seed_data, indent=2),
                )
                raw = await provider.complete(prompt)
                response = parse_json_response(raw)
                with get_db(db_path) as conn:
                    insert_designer_response(conn, DesignerResponse(
                        case_id=case.case_id, model_id=model_id, response=response,
                    ))
                completed += 1
                _mark_resolved(db_path, case.case_id, model_id, "designer")
                logger.info("[%d/%d] Designer %s on %s: OK", completed, total_to_run, model_id, case.case_id)
            except Exception as e:
                failed += 1
                _log_failure(db_path, case.case_id, model_id, "designer", str(e))
                logger.error("[%d/%d] Designer %s on %s FAILED: %s", completed+failed, total_to_run, model_id, case.case_id, e)

    await asyncio.gather(*[_run_one(c) for c in to_run])
    return {"completed": completed, "skipped": skipped, "failed": failed, "total": len(cases)}


async def collect_analyst_responses(
    db_path: Path,
    model_id: str,
    domain: str | None = None,
    max_concurrent: int = 4,
    max_tokens: int = 10000,
) -> dict:
    """Collect analyst responses for a single model across all cases.

    Skips cases where this model already has a response.
    """
    existing = _get_existing_pairs(db_path, "analyst_responses")

    with get_db(db_path) as conn:
        cases = list_cases(conn, domain)

    # Only run on cases that have a vignette
    cases = [c for c in cases if c.vignette]
    to_run = [c for c in cases if (c.case_id, model_id) not in existing]
    skipped = len(cases) - len(to_run)

    if not to_run:
        return {"completed": 0, "skipped": skipped, "failed": 0, "total": len(cases)}

    provider = get_provider(model_id)

    # Pre-warm: send a tiny request to ensure model is loaded into memory
    # This prevents Cloudflare 524 timeouts on the first real request
    logger.info("Pre-warming model %s...", model_id)
    try:
        await provider.complete("Say OK")
        logger.info("Model %s is warm", model_id)
    except Exception as e:
        logger.warning("Pre-warm failed for %s: %s (continuing anyway)", model_id, e)

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0
    total_to_run = len(to_run)

    async def _run_one(case):
        nonlocal completed, failed
        async with semaphore:
            try:
                prompt = build_step_a(vignette=case.vignette)
                raw = await provider.complete(prompt, max_tokens=max_tokens)
                response = parse_json_response(raw)
                with get_db(db_path) as conn:
                    insert_analyst_response(conn, AnalystResponse(
                        case_id=case.case_id, model_id=model_id, response=response,
                    ))
                completed += 1
                _mark_resolved(db_path, case.case_id, model_id, "analyst")
                logger.info("[%d/%d] Analyst %s on %s: OK", completed, total_to_run, model_id, case.case_id)
            except Exception as e:
                failed += 1
                _log_failure(db_path, case.case_id, model_id, "analyst", str(e))
                logger.error("[%d/%d] Analyst %s on %s FAILED: %s", completed+failed, total_to_run, model_id, case.case_id, e)

    await asyncio.gather(*[_run_one(c) for c in to_run])
    return {"completed": completed, "skipped": skipped, "failed": failed, "total": len(cases)}
