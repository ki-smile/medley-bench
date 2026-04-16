"""Export database contents for Part 2 benchmark deployment."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.db import get_db


_ENSEMBLE_SELECTION = None

def _get_ensemble_selection(domain: str) -> list[str] | None:
    """Load curated ensemble selection for a domain. Returns model_id list or None."""
    global _ENSEMBLE_SELECTION
    if _ENSEMBLE_SELECTION is None:
        sel_path = Path(__file__).parent.parent.parent.parent / "data" / "ensemble_selection.json"
        if sel_path.exists():
            with open(sel_path) as f:
                _ENSEMBLE_SELECTION = json.load(f)
        else:
            _ENSEMBLE_SELECTION = {}
    return _ENSEMBLE_SELECTION.get(domain, {}).get("analysts")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _export_instances(conn, domain: str) -> list[dict]:
    """Export instances for a domain. Strips known_answer field (security boundary)."""
    rows = conn.execute(
        "SELECT * FROM cases WHERE domain = ?", (domain,)
    ).fetchall()

    instances = []
    for row in rows:
        instance = {
            "instance_id": row["case_id"],
            "domain": row["domain"],
            "vignette": row["vignette"],
            "difficulty_tier": row["difficulty_tier"],
            "is_known_answer": bool(row["is_known_answer"]),
            "is_trap": bool(row["is_trap"]),
            "is_dose_response": bool(row["is_dose_response"]),
            "is_minimal_instruction": bool(row["is_minimal_instruction"]),
            "is_error_detection": bool(row["is_error_detection"]),
            "is_counterfactual": bool(row["is_counterfactual"]),
        }

        # Add claims
        claims = conn.execute(
            "SELECT * FROM claims WHERE case_id = ?", (row["case_id"],)
        ).fetchall()
        instance["key_claims"] = [
            {
                "claim_id": c["claim_id"],
                "claim_text": c["claim_text"],
                "majority_strength": c["majority_strength"],
                "jsd_score": c["jsd_score"],
            }
            for c in claims
        ]

        # Add analyst outputs — use curated ensemble if available
        ensemble_selection = _get_ensemble_selection(row["domain"] if "domain" in row.keys() else "")
        if ensemble_selection:
            # Use only the selected analysts for this domain
            placeholders = ",".join("?" for _ in ensemble_selection)
            analysts = conn.execute(
                f"""SELECT model_id, response FROM analyst_responses
                   WHERE case_id = ? AND model_id IN ({placeholders})""",
                (row["case_id"], *ensemble_selection),
            ).fetchall()
        else:
            # Fallback: all analysts
            analysts = conn.execute(
                """SELECT model_id, response FROM analyst_responses
                   WHERE case_id = ?""",
                (row["case_id"],),
            ).fetchall()
        instance["ensemble_outputs"] = [
            {"model_id": a["model_id"], "response": json.loads(a["response"])}
            for a in analysts
        ]

        # Dose-response: also include reduced ensemble
        if row["is_dose_response"]:
            instance["probe_ensemble_outputs"] = instance["ensemble_outputs"][:2]

        instances.append(instance)

    return instances


def _export_consensus(conn, domain: str) -> dict:
    rows = conn.execute(
        """SELECT c.case_id, con.consensus_data
           FROM cases c JOIN consensus con ON c.case_id = con.case_id
           WHERE c.domain = ?""",
        (domain,),
    ).fetchall()
    return {row["case_id"]: json.loads(row["consensus_data"]) for row in rows}


def _export_known_answers(conn) -> dict:
    """Export known answers separately — used for scoring only, NEVER shown to model."""
    rows = conn.execute(
        "SELECT case_id, known_answer FROM cases WHERE is_known_answer = 1"
    ).fetchall()
    return {
        row["case_id"]: json.loads(row["known_answer"])
        for row in rows
        if row["known_answer"]
    }


def _export_metadata(conn) -> dict:
    domains = ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]
    counts = {}
    for d in domains:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM cases WHERE domain = ?", (d,)
        ).fetchone()
        counts[d] = row["n"]

    ka_count = conn.execute(
        "SELECT COUNT(*) as n FROM cases WHERE is_known_answer = 1"
    ).fetchone()["n"]

    prompts = conn.execute(
        "SELECT prompt_name, MAX(version) as v, content_hash FROM prompt_versions GROUP BY prompt_name"
    ).fetchall()

    return {
        "three_step_design": True,
        "domains": domains,
        "instance_counts": counts,
        "total_instances": sum(counts.values()),
        "known_answer_count": ka_count,
        "prompt_versions": {
            p["prompt_name"]: {"version": p["v"], "hash": p["content_hash"]}
            for p in prompts
        },
    }


def export_for_benchmark(db_path: Path | str, output_dir: Path | str) -> None:
    """Generate the full export directory for Part 2 benchmark deployment."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with get_db(db_path) as conn:
        # Instances per domain
        for domain in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]:
            instances = _export_instances(conn, domain)
            _write_json(output / "instances" / f"{domain}.json", instances)

            consensus = _export_consensus(conn, domain)
            _write_json(output / "consensus" / f"{domain}.json", consensus)

        # Known answers (separate, scoring only)
        known = _export_known_answers(conn)
        if known:
            _write_json(output / "known_answers.json", known)

        # Metadata
        metadata = _export_metadata(conn)
        _write_json(output / "metadata.json", metadata)
