"""MEDLEY-BENCH Admin Web Panel — FastAPI + Jinja2 + htmx."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.db import get_db
from src.admin.db.models import list_cases, get_case, get_claims, get_analyst_responses, get_known_answer_cases

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="MEDLEY-BENCH Admin")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def _db():
        return get_db(db_path)

    # ── Dashboard ──────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        with _db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            by_domain = {}
            for d in ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]:
                by_domain[d] = conn.execute("SELECT COUNT(*) FROM cases WHERE domain=?", (d,)).fetchone()[0]

            ka = conn.execute("SELECT COUNT(*) FROM cases WHERE is_known_answer=1").fetchone()[0]
            analysts_done = conn.execute("SELECT COUNT(DISTINCT case_id) FROM analyst_responses").fetchone()[0]
            consensus_done = conn.execute("SELECT COUNT(*) FROM consensus").fetchone()[0]
            claims_done = conn.execute("SELECT COUNT(DISTINCT case_id) FROM claims").fetchone()[0]

            # Model counts
            d_models = conn.execute("SELECT model_id, COUNT(*) as n FROM designer_responses GROUP BY model_id").fetchall()
            a_models = conn.execute("SELECT model_id, COUNT(*) as n FROM analyst_responses GROUP BY model_id").fetchall()

            probes = {
                "trap": conn.execute("SELECT COUNT(*) FROM cases WHERE is_trap=1").fetchone()[0],
                "dose_response": conn.execute("SELECT COUNT(*) FROM cases WHERE is_dose_response=1").fetchone()[0],
                "minimal": conn.execute("SELECT COUNT(*) FROM cases WHERE is_minimal_instruction=1").fetchone()[0],
                "counterfactual": conn.execute("SELECT COUNT(*) FROM cases WHERE is_counterfactual=1").fetchone()[0],
                "error_detection": conn.execute("SELECT COUNT(*) FROM cases WHERE is_error_detection=1").fetchone()[0],
            }

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "total": total, "by_domain": by_domain, "ka": ka,
            "analysts_done": analysts_done, "consensus_done": consensus_done,
            "claims_done": claims_done, "probes": probes,
            "d_models": [dict(r) for r in d_models],
            "a_models": [dict(r) for r in a_models],
        })

    # ── Cases List ─────────────────────────────────────────
    @app.get("/cases", response_class=HTMLResponse)
    async def cases_list(request: Request, domain: str = None):
        with _db() as conn:
            cases = list_cases(conn, domain)
        return templates.TemplateResponse("cases.html", {
            "request": request, "cases": cases, "domain_filter": domain,
        })

    # ── Case Detail ────────────────────────────────────────
    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    async def case_detail(request: Request, case_id: str):
        with _db() as conn:
            case = get_case(conn, case_id)
            claims = get_claims(conn, case_id)
            analysts = get_analyst_responses(conn, case_id)
            designers = conn.execute(
                "SELECT model_id, response FROM designer_responses WHERE case_id=?", (case_id,)
            ).fetchall()
        return templates.TemplateResponse("case_detail.html", {
            "request": request, "case": case, "claims": claims,
            "analysts": analysts, "designers": [dict(r) for r in designers],
        })

    # ── Models Matrix ──────────────────────────────────────
    @app.get("/models", response_class=HTMLResponse)
    async def models_view(request: Request):
        from src.admin.generation.collect import get_collection_status
        status = get_collection_status(db_path)
        return templates.TemplateResponse("models.html", {
            "request": request, "status": status,
        })

    # ── API Endpoints ──────────────────────────────────────
    @app.get("/api/status")
    async def api_status():
        with _db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            ka = conn.execute("SELECT COUNT(*) FROM cases WHERE is_known_answer=1").fetchone()[0]
            analysts = conn.execute("SELECT COUNT(DISTINCT case_id) FROM analyst_responses").fetchone()[0]
            consensus = conn.execute("SELECT COUNT(*) FROM consensus").fetchone()[0]
        return {"total_cases": total, "known_answer": ka,
                "analyst_coverage": analysts, "consensus_built": consensus}

    @app.get("/api/validate")
    async def api_validate():
        from src.admin.validation import run_quality_gates
        return run_quality_gates(db_path)

    return app
