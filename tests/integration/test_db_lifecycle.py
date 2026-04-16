"""Integration tests for database lifecycle: init → insert → query → export."""
import json
import tempfile
from pathlib import Path

import pytest

from src.core.db import init_db, get_db
from src.admin.db.models import (
    Case, Claim, DesignerResponse, AnalystResponse,
    insert_case, insert_claim, insert_designer_response, insert_analyst_response,
    get_case, list_cases, get_known_answer_cases, get_claims, get_analyst_responses,
)
from src.admin.db.export import export_for_benchmark


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database."""
    path = tmp_path / "test.db"
    schema = Path(__file__).parent.parent.parent / "src" / "admin" / "db" / "schema.sql"
    init_db(path, schema)
    return path


class TestDbLifecycle:
    def test_init_creates_tables(self, db_path):
        with get_db(db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {t["name"] for t in tables}
            assert "cases" in names
            assert "claims" in names
            assert "consensus" in names

    def test_insert_and_retrieve_case(self, db_path):
        case = Case(
            case_id="TEST_001", domain="medical",
            seed_data={"test": True}, vignette="Test vignette",
            is_known_answer=True,
            known_answer={"correct_answer": "X", "target_wrong_claim": "C1"},
        )
        with get_db(db_path) as conn:
            insert_case(conn, case)
            retrieved = get_case(conn, "TEST_001")

        assert retrieved is not None
        assert retrieved.domain == "medical"
        assert retrieved.is_known_answer is True
        assert retrieved.known_answer["correct_answer"] == "X"

    def test_known_answer_query(self, db_path):
        with get_db(db_path) as conn:
            insert_case(conn, Case("KA_001", "medical", {}, is_known_answer=True,
                                   known_answer={"correct_answer": "Y", "target_wrong_claim": "C2"}))
            insert_case(conn, Case("STD_001", "medical", {}))
            ka = get_known_answer_cases(conn)

        assert len(ka) == 1
        assert ka[0].case_id == "KA_001"

    def test_claims_insert_and_retrieve(self, db_path):
        with get_db(db_path) as conn:
            insert_case(conn, Case("C_001", "medical", {}))
            insert_claim(conn, Claim("C_001", "C1", "Test claim", majority_strength=3, jsd_score=0.4))
            insert_claim(conn, Claim("C_001", "C2", "Another claim", majority_strength=1, jsd_score=0.7))
            claims = get_claims(conn, "C_001")

        assert len(claims) == 2
        assert claims[0].majority_strength == 3

    def test_export_strips_known_answer(self, db_path, tmp_path):
        """Export must NOT include known_answer field in instance data."""
        with get_db(db_path) as conn:
            insert_case(conn, Case(
                "KA_EXP_001", "medical", {"seed": True},
                vignette="Test", is_known_answer=True,
                known_answer={"correct_answer": "Z", "target_wrong_claim": "C1"},
            ))
            insert_claim(conn, Claim("KA_EXP_001", "C1", "Claim"))
            # Need consensus for export
            conn.execute(
                "INSERT INTO consensus (case_id, consensus_data) VALUES (?, ?)",
                ("KA_EXP_001", json.dumps({"claim_confidences": {"C1": 0.7}})),
            )

        export_dir = tmp_path / "export"
        export_for_benchmark(db_path, export_dir)

        # Check instances don't contain known_answer field
        instances = json.loads((export_dir / "instances" / "medical.json").read_text())
        for inst in instances:
            assert "known_answer" not in inst or inst.get("known_answer") is None

        # Check known_answers.json exists separately
        ka_path = export_dir / "known_answers.json"
        assert ka_path.exists()
        ka_data = json.loads(ka_path.read_text())
        assert "KA_EXP_001" in ka_data

        # Check metadata
        meta = json.loads((export_dir / "metadata.json").read_text())
        assert meta["three_step_design"] is True
        assert meta["known_answer_count"] == 1
