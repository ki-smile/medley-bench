"""SQLite connection management for MEDLEY-BENCH."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def get_db(db_path: Path | str):
    """Context manager for SQLite connections.

    Configures WAL mode, Row factory, and foreign key enforcement.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str, schema_path: Path | str | None = None):
    """Initialize the database by executing the schema SQL file.

    If schema_path is None, looks for schema.sql relative to this package.
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "admin" / "db" / "schema.sql"

    schema_sql = Path(schema_path).read_text()

    with get_db(db_path) as conn:
        conn.executescript(schema_sql)
