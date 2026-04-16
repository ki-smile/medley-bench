"""Prompt version registry — saves all prompt templates to the database."""
from __future__ import annotations

from pathlib import Path
from sqlite3 import Connection

from src.admin.db.models import PromptVersion, insert_prompt_version
from src.admin.generation.prompts import content_hash
from src.admin.generation.prompts import step_a_prompt
from src.admin.generation.prompts import step_b_private_prompt
from src.admin.generation.prompts import step_b_social_prompt
from src.admin.generation.prompts import step_b_minimal_prompt
from src.admin.generation.prompts import judge_prompt

ALL_PROMPTS = {
    "step_a": step_a_prompt,
    "step_b_private": step_b_private_prompt,
    "step_b_social": step_b_social_prompt,
    "step_b_minimal": step_b_minimal_prompt,
    "judge": judge_prompt,
}


def register_all_prompts(conn: Connection) -> int:
    """Register all prompt templates in the database.

    Checks if the current version already exists (by hash).
    Returns count of newly registered prompts.
    """
    count = 0
    for name, module in ALL_PROMPTS.items():
        template = module.PROMPT_TEMPLATE
        version = module.PROMPT_VERSION
        h = content_hash(template)

        # Check if this hash already registered
        existing = conn.execute(
            "SELECT 1 FROM prompt_versions WHERE prompt_name = ? AND content_hash = ?",
            (name, h),
        ).fetchone()

        if existing is None:
            insert_prompt_version(conn, PromptVersion(
                prompt_name=name,
                version=version,
                content_hash=h,
                content=template,
            ))
            count += 1

    return count
