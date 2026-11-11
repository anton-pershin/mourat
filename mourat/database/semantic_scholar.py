"""Durable state for Semantic Scholar publication-date collection."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Iterable


def load_query_state(conn: sqlite3.Connection, query: str) -> dict | None:
    """Return the last successful cursor for one exact query string."""
    row = conn.execute(
        "SELECT last_completed_end_date, last_backfill_completed_on "
        "FROM semantic_scholar_collection_state WHERE query = ?",
        (query,),
    ).fetchone()
    return dict(row) if row is not None else None


def get_seen_paper_ids(conn: sqlite3.Connection, paper_ids: Iterable[str]) -> set[str]:
    """Return the subset of provider IDs already committed to the ledger."""
    unique_ids = sorted(set(paper_ids))
    seen: set[str] = set()
    # Keep each SQLite statement below the common bound-parameter limit.
    for offset in range(0, len(unique_ids), 500):
        chunk = unique_ids[offset : offset + 500]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT paper_id FROM semantic_scholar_seen_papers "
            f"WHERE paper_id IN ({placeholders})",
            chunk,
        ).fetchall()
        seen.update(row["paper_id"] for row in rows)
    return seen


def _isoformat(value: date | str | None) -> str | None:
    return value.isoformat() if isinstance(value, date) else value


def record_successful_collection(
    conn: sqlite3.Connection,
    paper_ids: Iterable[str],
    query_states: Iterable[dict],
) -> None:
    """Commit provider IDs and query cursors as one durable unit.

    The caller invokes this only after every downstream pipeline stage has
    succeeded. ``INSERT OR IGNORE`` makes a repeated commit idempotent.
    """
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO semantic_scholar_seen_papers (paper_id) VALUES (?)",
            ((paper_id,) for paper_id in sorted(set(paper_ids))),
        )
        for state in query_states:
            conn.execute(
                "INSERT INTO semantic_scholar_collection_state "
                "(query, last_completed_end_date, last_backfill_completed_on) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(query) DO UPDATE SET "
                "last_completed_end_date = excluded.last_completed_end_date, "
                "last_backfill_completed_on = excluded.last_backfill_completed_on",
                (
                    state["query"],
                    _isoformat(state["completed_end_date"]),
                    _isoformat(state.get("backfill_completed_on")),
                ),
            )
