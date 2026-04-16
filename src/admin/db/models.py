"""Data models and CRUD operations for MEDLEY-BENCH SQLite database."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection

from src.core.db import get_db


@dataclass
class Case:
    case_id: str
    domain: str
    seed_data: dict
    vignette: str | None = None
    difficulty_tier: str | None = None
    disagreement_score: float | None = None
    is_known_answer: bool = False
    known_answer: dict | None = None
    is_trap: bool = False
    is_dose_response: bool = False
    is_minimal_instruction: bool = False
    is_error_detection: bool = False
    is_counterfactual: bool = False

    @classmethod
    def from_row(cls, row) -> Case:
        return cls(
            case_id=row["case_id"],
            domain=row["domain"],
            seed_data=json.loads(row["seed_data"]),
            vignette=row["vignette"],
            difficulty_tier=row["difficulty_tier"],
            disagreement_score=row["disagreement_score"],
            is_known_answer=bool(row["is_known_answer"]),
            known_answer=json.loads(row["known_answer"]) if row["known_answer"] else None,
            is_trap=bool(row["is_trap"]),
            is_dose_response=bool(row["is_dose_response"]),
            is_minimal_instruction=bool(row["is_minimal_instruction"]),
            is_error_detection=bool(row["is_error_detection"]),
            is_counterfactual=bool(row["is_counterfactual"]),
        )


@dataclass
class DesignerResponse:
    case_id: str
    model_id: str
    response: dict
    id: int | None = None

    @classmethod
    def from_row(cls, row) -> DesignerResponse:
        return cls(
            id=row["id"],
            case_id=row["case_id"],
            model_id=row["model_id"],
            response=json.loads(row["response"]),
        )


@dataclass
class AnalystResponse:
    case_id: str
    model_id: str
    response: dict
    jackknife_left_out: bool = False
    id: int | None = None

    @classmethod
    def from_row(cls, row) -> AnalystResponse:
        return cls(
            id=row["id"],
            case_id=row["case_id"],
            model_id=row["model_id"],
            response=json.loads(row["response"]),
            jackknife_left_out=bool(row["jackknife_left_out"]),
        )


@dataclass
class Claim:
    case_id: str
    claim_id: str
    claim_text: str
    majority_strength: int | None = None
    jsd_score: float | None = None

    @classmethod
    def from_row(cls, row) -> Claim:
        return cls(
            case_id=row["case_id"],
            claim_id=row["claim_id"],
            claim_text=row["claim_text"],
            majority_strength=row["majority_strength"],
            jsd_score=row["jsd_score"],
        )


@dataclass
class PromptVersion:
    prompt_name: str
    version: int
    content_hash: str
    content: str

    @classmethod
    def from_row(cls, row) -> PromptVersion:
        return cls(
            prompt_name=row["prompt_name"],
            version=row["version"],
            content_hash=row["content_hash"],
            content=row["content"],
        )


# ── CRUD Operations ────────────────────────────────────────────

def insert_case(conn: Connection, case: Case) -> None:
    conn.execute(
        """INSERT INTO cases (case_id, domain, seed_data, vignette, difficulty_tier,
           disagreement_score, is_known_answer, known_answer, is_trap,
           is_dose_response, is_minimal_instruction, is_error_detection, is_counterfactual)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            case.case_id, case.domain, json.dumps(case.seed_data),
            case.vignette, case.difficulty_tier, case.disagreement_score,
            int(case.is_known_answer),
            json.dumps(case.known_answer) if case.known_answer else None,
            int(case.is_trap), int(case.is_dose_response),
            int(case.is_minimal_instruction), int(case.is_error_detection),
            int(case.is_counterfactual),
        ),
    )


def get_case(conn: Connection, case_id: str) -> Case | None:
    row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return Case.from_row(row) if row else None


def list_cases(conn: Connection, domain: str | None = None) -> list[Case]:
    if domain:
        rows = conn.execute("SELECT * FROM cases WHERE domain = ?", (domain,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM cases").fetchall()
    return [Case.from_row(r) for r in rows]


def get_known_answer_cases(conn: Connection) -> list[Case]:
    rows = conn.execute("SELECT * FROM cases WHERE is_known_answer = 1").fetchall()
    return [Case.from_row(r) for r in rows]


def insert_designer_response(conn: Connection, resp: DesignerResponse) -> None:
    conn.execute(
        """INSERT INTO designer_responses (case_id, model_id, response)
           VALUES (?, ?, ?)""",
        (resp.case_id, resp.model_id, json.dumps(resp.response)),
    )


def insert_analyst_response(conn: Connection, resp: AnalystResponse) -> None:
    conn.execute(
        """INSERT INTO analyst_responses (case_id, model_id, response, jackknife_left_out)
           VALUES (?, ?, ?, ?)""",
        (resp.case_id, resp.model_id, json.dumps(resp.response),
         int(resp.jackknife_left_out)),
    )


def get_analyst_responses(conn: Connection, case_id: str) -> list[AnalystResponse]:
    rows = conn.execute(
        "SELECT * FROM analyst_responses WHERE case_id = ?", (case_id,)
    ).fetchall()
    return [AnalystResponse.from_row(r) for r in rows]


def insert_claim(conn: Connection, claim: Claim) -> None:
    conn.execute(
        """INSERT INTO claims (case_id, claim_id, claim_text, majority_strength, jsd_score)
           VALUES (?, ?, ?, ?, ?)""",
        (claim.case_id, claim.claim_id, claim.claim_text,
         claim.majority_strength, claim.jsd_score),
    )


def get_claims(conn: Connection, case_id: str) -> list[Claim]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE case_id = ?", (case_id,)
    ).fetchall()
    return [Claim.from_row(r) for r in rows]


def insert_prompt_version(conn: Connection, pv: PromptVersion) -> None:
    conn.execute(
        """INSERT INTO prompt_versions (prompt_name, version, content_hash, content)
           VALUES (?, ?, ?, ?)""",
        (pv.prompt_name, pv.version, pv.content_hash, pv.content),
    )


def update_case_difficulty(conn: Connection, case_id: str, tier: str, score: float) -> None:
    conn.execute(
        "UPDATE cases SET difficulty_tier = ?, disagreement_score = ? WHERE case_id = ?",
        (tier, score, case_id),
    )
