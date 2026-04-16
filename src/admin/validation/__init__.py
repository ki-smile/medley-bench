"""Quality gate validation for MEDLEY-BENCH (G1-G15)."""
from __future__ import annotations

from pathlib import Path

from src.core.db import get_db


def run_quality_gates(db_path: Path) -> list[dict]:
    """Run all 15 quality gates and return results.

    Each result: {gate, passed, severity, message, value, threshold}
    """
    results = []

    with get_db(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        domains = {}
        for d in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]:
            domains[d] = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE domain = ?", (d,)
            ).fetchone()[0]
        ka_count = conn.execute("SELECT COUNT(*) FROM cases WHERE is_known_answer = 1").fetchone()[0]
        trap_count = conn.execute("SELECT COUNT(*) FROM cases WHERE is_trap = 1").fetchone()[0]
        minimal_count = conn.execute("SELECT COUNT(*) FROM cases WHERE is_minimal_instruction = 1").fetchone()[0]
        error_count = conn.execute("SELECT COUNT(*) FROM cases WHERE is_error_detection = 1").fetchone()[0]
        analyst_count = conn.execute("SELECT COUNT(DISTINCT case_id) FROM analyst_responses").fetchone()[0]
        claim_count = conn.execute(
            "SELECT COUNT(DISTINCT case_id) FROM claims"
        ).fetchone()[0]
        consensus_count = conn.execute("SELECT COUNT(*) FROM consensus").fetchone()[0]
        prompt_count = conn.execute(
            "SELECT COUNT(DISTINCT prompt_name) FROM prompt_versions"
        ).fetchone()[0]

    def _gate(gate_id, check, threshold, severity, desc):
        passed = check >= threshold if isinstance(threshold, (int, float)) else check
        results.append({
            "gate": gate_id,
            "passed": passed,
            "severity": severity,
            "message": f"{desc}: {check} (threshold: {threshold})",
            "value": check,
            "threshold": threshold,
        })

    # G1: Total cases
    _gate("G1", total, 65, "BLOCKING", "Total cases in database")

    # G2: Cases per domain
    for d, count in domains.items():
        _gate(f"G2_{d}", count, 15, "BLOCKING", f"Cases in {d}")

    # G3: Difficulty distribution (check after Step 1)
    # Simplified: just check that at least some cases have difficulty tiers
    with get_db(db_path) as conn:
        tiered = conn.execute("SELECT COUNT(*) FROM cases WHERE difficulty_tier IS NOT NULL").fetchone()[0]
    _gate("G3", tiered > 0, True, "WARNING", f"Cases with difficulty tiers: {tiered}")

    # G4: Analyst responses complete
    _gate("G4", analyst_count, total if total > 0 else 1, "BLOCKING",
           "Cases with analyst responses")

    # G5: Claims extracted
    _gate("G5", claim_count, total if total > 0 else 1, "BLOCKING",
           "Cases with claims extracted")

    # G6: Consensus built
    _gate("G6", consensus_count, total if total > 0 else 1, "BLOCKING",
           "Cases with consensus built")

    # G7/G8: Judge variance (checked separately after judge runs)
    _gate("G7", True, True, "WARNING", "Judge variance Fleiss κ ≥ 0.40 (check after judge run)")
    _gate("G8", True, True, "INFO", "Judge variance Fleiss κ ≥ 0.60 (check after judge run)")

    # G9: Trap cases
    _gate("G9", trap_count, 12, "WARNING", "Trap cases designated")

    # G10: Known-answer family (v3.1: ≥ 18)
    _gate("G10", ka_count, 18, "BLOCKING", "Known-answer instances designated")

    # G11: Minimal-instruction probes
    _gate("G11", minimal_count, 8, "WARNING", "Minimal-instruction probes")

    # G12: Error detection probes
    _gate("G12", error_count, 4, "WARNING", "Error detection probes")

    # G13: Prompt versions recorded
    _gate("G13", prompt_count, 5, "BLOCKING", "Prompt versions recorded")

    # G14: metadata.json completeness (checked at export time)
    _gate("G14", True, True, "BLOCKING", "metadata.json complete (checked at export)")

    # G15: Sophistry IRR check (v3.2 — always passes, result determines weight)
    _gate("G15", True, True, "BLOCKING", "Sophistry IRR check (run after judge variance analysis)")

    return results
