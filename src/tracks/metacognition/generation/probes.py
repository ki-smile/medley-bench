"""Probe designation for special instance types.

Manages dose-response, minimal-instruction, trap, and counterfactual probes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.core.db import get_db

logger = logging.getLogger(__name__)


def designate_dose_response_probes(
    db_path: Path, n_per_domain: int = 5
) -> int:
    """Designate dose-response probe instances (15 total, 5 per domain).

    These instances get an additional benchmark call with a reduced ensemble
    (K=1 or K=2 analysts instead of K=4) to test whether capitulation
    scales with analyst count (normative conformity) or argument quality
    (informational influence).
    """
    count = 0
    with get_db(db_path) as conn:
        for domain in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]:
            # Select non-special cases (avoid overlap with known-answer/trap)
            rows = conn.execute("""
                SELECT case_id FROM cases
                WHERE domain = ?
                  AND is_known_answer = 0
                  AND is_trap = 0
                  AND is_dose_response = 0
                ORDER BY RANDOM()
                LIMIT ?
            """, (domain, n_per_domain)).fetchall()

            for row in rows:
                conn.execute(
                    "UPDATE cases SET is_dose_response = 1 WHERE case_id = ?",
                    (row["case_id"],),
                )
                count += 1

    logger.info("Designated %d dose-response probes", count)
    return count


def designate_minimal_instruction_probes(
    db_path: Path, n_total: int = 10
) -> int:
    """Designate minimal-instruction probe instances (10 total).

    These get an additional benchmark call with a stripped-down B-social
    prompt (no metacognition rubric) to measure instruction dependence.
    """
    count = 0
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT case_id FROM cases
            WHERE is_known_answer = 0
              AND is_trap = 0
              AND is_minimal_instruction = 0
            ORDER BY RANDOM()
            LIMIT ?
        """, (n_total,)).fetchall()

        for row in rows:
            conn.execute(
                "UPDATE cases SET is_minimal_instruction = 1 WHERE case_id = ?",
                (row["case_id"],),
            )
            count += 1

    logger.info("Designated %d minimal-instruction probes", count)
    return count


def designate_trap_cases(
    db_path: Path, n_per_domain: int = 5
) -> int:
    """Designate trap cases (15 total, 5 per domain).

    Trap cases test for specific gaming strategies:
    - Shared errors across analysts (do you follow a unanimous but wrong consensus?)
    - Adversarial confident minority (does one very confident wrong analyst sway you?)
    - Generic response detection (does the response actually reference the case?)
    """
    count = 0
    with get_db(db_path) as conn:
        for domain in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]:
            rows = conn.execute("""
                SELECT case_id FROM cases
                WHERE domain = ?
                  AND is_known_answer = 0
                  AND is_trap = 0
                ORDER BY RANDOM()
                LIMIT ?
            """, (domain, n_per_domain)).fetchall()

            for row in rows:
                conn.execute(
                    "UPDATE cases SET is_trap = 1 WHERE case_id = ?",
                    (row["case_id"],),
                )
                count += 1

    logger.info("Designated %d trap cases", count)
    return count


def designate_counterfactual_instances(
    db_path: Path, n_total: int = 10
) -> int:
    """Designate counterfactual instances (10 total).

    These get contradictory analyst outputs on the same vignette.
    A grounded model should produce a meaningfully different, more uncertain B-social.
    A sophistic model sounds equally authoritative regardless.
    """
    count = 0
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT case_id FROM cases
            WHERE is_counterfactual = 0
              AND is_known_answer = 0
            ORDER BY RANDOM()
            LIMIT ?
        """, (n_total,)).fetchall()

        for row in rows:
            conn.execute(
                "UPDATE cases SET is_counterfactual = 1 WHERE case_id = ?",
                (row["case_id"],),
            )
            count += 1

    logger.info("Designated %d counterfactual instances", count)
    return count


def run_all_designations(db_path: Path) -> dict[str, int]:
    """Run all probe designations. Call after Step 1 + Step 2 complete."""
    return {
        "dose_response": designate_dose_response_probes(db_path),
        "minimal_instruction": designate_minimal_instruction_probes(db_path),
        "trap_cases": designate_trap_cases(db_path),
        "counterfactual": designate_counterfactual_instances(db_path),
    }
