"""Database initialization and connection layer."""

import sqlite3
from pathlib import Path


def create_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create and return a database connection."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the database schema from schema.sql. Idempotent — skips existing tables."""
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
    # Triggers need separate handling since SQLite <3.37 lacks IF NOT EXISTS
    for trigger_name, trigger_sql in _FTS5_TRIGGERS.items():
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name=?",
            (trigger_name,),
        ).fetchone()
        if existing is None:
            conn.executescript(trigger_sql)
    conn.commit()


_FTS5_TRIGGERS = {
    "content_items_ai": """
        CREATE TRIGGER content_items_ai AFTER INSERT ON content_items BEGIN
            INSERT INTO content_items_fts(rowid, name, description)
            VALUES (NEW.rowid, NEW.name, NEW.description);
        END;
    """,
    "content_items_ad": """
        CREATE TRIGGER content_items_ad AFTER DELETE ON content_items BEGIN
            INSERT INTO content_items_fts(content_items_fts, rowid, name, description)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.description);
        END;
    """,
    "content_items_au": """
        CREATE TRIGGER content_items_au AFTER UPDATE ON content_items BEGIN
            INSERT INTO content_items_fts(content_items_fts, rowid, name, description)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.description);
            INSERT INTO content_items_fts(rowid, name, description)
            VALUES (NEW.rowid, NEW.name, NEW.description);
        END;
    """,
}


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize database: create connection and apply schema if needed."""
    conn = create_connection(db_path)
    apply_schema(conn)
    return conn
