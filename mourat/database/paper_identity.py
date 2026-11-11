"""Durable newest-paper ledger and canonical paper identity operations."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from comourat.models import PaperIdentity

_ID_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ARXIV_PREFIX = "paper_arxiv-"


@dataclass(frozen=True)
class PaperIdentityResolution:
    """The selected row and whether arXiv metadata must remain unchanged."""

    content_item_id: str
    preserve_existing: bool = False


def canonical_identity(
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> PaperIdentity:
    """Normalize identifiers and apply the repository's arXiv-DOI rule."""
    identity = PaperIdentity.from_values(arxiv_id=arxiv_id, doi=doi)
    if identity.doi is not None:
        identity.doi = identity.doi.lower()
    return identity


def paper_content_item_id(
    title: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> str:
    """Derive a content-item ID with arXiv, DOI, then title precedence."""
    identity = canonical_identity(arxiv_id=arxiv_id, doi=doi)
    if identity.arxiv_id:
        return f"{_ARXIV_PREFIX}{identity.arxiv_id}"
    if identity.doi:
        return f"paper_doi-{identity.doi}"
    slug = _ID_NON_ALNUM.sub("_", title.strip().lower()).strip("_")
    return f"paper_{slug}"


def list_semantic_scholar_ledger_ids(conn: sqlite3.Connection) -> set[str]:
    """Return provider IDs that completed a newest-paper run."""
    rows = conn.execute(
        "SELECT paper_id FROM semantic_scholar_collection_ledger"
    ).fetchall()
    return {row[0] for row in rows}


def record_semantic_scholar_processed(
    conn: sqlite3.Connection,
    paper_ids: Iterable[str],
) -> None:
    """Record one completed batch in a single transaction."""
    unique_ids = sorted({paper_id for paper_id in paper_ids if paper_id})
    if not unique_ids:
        return
    processed_at = datetime.now(timezone.utc).isoformat()
    with conn:
        for paper_id in unique_ids:
            conn.execute(
                "INSERT INTO semantic_scholar_collection_ledger "
                "(paper_id, processed_at) VALUES (?, ?) "
                "ON CONFLICT(paper_id) DO UPDATE SET "
                "processed_at = excluded.processed_at",
                (paper_id, processed_at),
            )


def find_paper_content_item_ids(
    conn: sqlite3.Connection,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> list[str]:
    """Find existing content rows through canonical aliases."""
    identity = canonical_identity(arxiv_id=arxiv_id, doi=doi)
    aliases = []
    if identity.arxiv_id:
        aliases.append(("arxiv", identity.arxiv_id))
    if identity.doi:
        aliases.append(("doi", identity.doi))

    result: list[str] = []
    for identity_type, identity_value in aliases:
        rows = conn.execute(
            "SELECT pi.content_item_id FROM paper_identities pi "
            "JOIN content_items ci ON ci.id = pi.content_item_id "
            "WHERE pi.identity_type = ? AND pi.identity_value = ? "
            "ORDER BY pi.content_item_id",
            (identity_type, identity_value),
        ).fetchall()
        for row in rows:
            if row[0] not in result:
                result.append(row[0])
    return result


def find_existing_content_item_id(
    conn: sqlite3.Connection,
    title: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> str | None:
    """Find a row through aliases or the deterministic fallback ID."""
    candidates = find_paper_content_item_ids(conn, arxiv_id=arxiv_id, doi=doi)
    derived_id = paper_content_item_id(title, arxiv_id=arxiv_id, doi=doi)
    if derived_id not in candidates:
        candidates.append(derived_id)
    for content_item_id in candidates:
        row = conn.execute(
            "SELECT id FROM content_items WHERE id = ?", (content_item_id,)
        ).fetchone()
        if row is not None:
            return row[0]
    return None


def _move_links(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    link_tables = {
        "item_technical_challenges": ("challenge_id, justification, relevance_score"),
        "item_research_questions": "question_id, justification, relevance_score",
        "item_research_topics": "topic_id, justification, relevance_score",
        "item_constraints": "constraint_id, justification, relevance_score",
    }
    for table, value_columns in link_tables.items():
        conn.execute(
            f"INSERT OR IGNORE INTO {table} "
            f"(content_id, {value_columns}) "
            f"SELECT ?, {value_columns} FROM {table} WHERE content_id = ?",
            (target_id, source_id),
        )
        conn.execute(f"DELETE FROM {table} WHERE content_id = ?", (source_id,))

    conn.execute(
        "INSERT OR IGNORE INTO content_item_references (source_id, target_id) "
        "SELECT CASE WHEN source_id = ? THEN ? ELSE source_id END, "
        "CASE WHEN target_id = ? THEN ? ELSE target_id END "
        "FROM content_item_references WHERE source_id = ? OR target_id = ?",
        (source_id, target_id, source_id, target_id, source_id, source_id),
    )
    conn.execute(
        "DELETE FROM content_item_references WHERE source_id = ? OR target_id = ?",
        (source_id, source_id),
    )


def _merge_existing_row(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
) -> None:
    if source_id == target_id:
        return
    _move_links(conn, source_id, target_id)
    conn.execute(
        "INSERT OR IGNORE INTO paper_identities "
        "(identity_type, identity_value, content_item_id) "
        "SELECT identity_type, identity_value, ? FROM paper_identities "
        "WHERE content_item_id = ?",
        (target_id, source_id),
    )
    conn.execute("DELETE FROM paper_identities WHERE content_item_id = ?", (source_id,))
    conn.execute("DELETE FROM content_items WHERE id = ?", (source_id,))


def _promote_existing_row(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
) -> None:
    source = conn.execute(
        "SELECT * FROM content_items WHERE id = ?", (source_id,)
    ).fetchone()
    if source is None:
        return
    conn.execute(
        "INSERT INTO content_items "
        "(id, name, description, source_type_id, platform_id, url, published_at, "
        "authors, influence_score, influence_metric_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            target_id,
            source["name"],
            source["description"],
            source["source_type_id"],
            source["platform_id"],
            source["url"],
            source["published_at"],
            source["authors"],
            source["influence_score"],
            source["influence_metric_id"],
        ),
    )
    _merge_existing_row(conn, source_id, target_id)


def resolve_paper_content_item_id(
    conn: sqlite3.Connection,
    title: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> PaperIdentityResolution:
    """Resolve aliases and merge or promote existing rows when required."""
    identity = canonical_identity(arxiv_id=arxiv_id, doi=doi)
    preferred_id = paper_content_item_id(
        title,
        arxiv_id=identity.arxiv_id,
        doi=identity.doi,
    )
    existing_ids = find_paper_content_item_ids(
        conn,
        arxiv_id=identity.arxiv_id,
        doi=identity.doi,
    )
    preferred_exists = conn.execute(
        "SELECT 1 FROM content_items WHERE id = ?", (preferred_id,)
    ).fetchone()
    if preferred_exists and preferred_id not in existing_ids:
        existing_ids.append(preferred_id)

    arxiv_rows = sorted(
        content_id
        for content_id in existing_ids
        if content_id.startswith(_ARXIV_PREFIX)
    )
    preserve_existing = identity.arxiv_id is None and bool(arxiv_rows)
    target_id = arxiv_rows[0] if preserve_existing else preferred_id

    with conn:
        target_exists = conn.execute(
            "SELECT 1 FROM content_items WHERE id = ?", (target_id,)
        ).fetchone()
        if target_exists is None and existing_ids:
            _promote_existing_row(conn, existing_ids[0], target_id)
        for source_id in existing_ids:
            if source_id != target_id:
                _merge_existing_row(conn, source_id, target_id)

    return PaperIdentityResolution(
        content_item_id=target_id,
        preserve_existing=preserve_existing,
    )


def register_paper_aliases(
    conn: sqlite3.Connection,
    content_item_id: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> None:
    """Store all known aliases for one persisted paper."""
    identity = canonical_identity(arxiv_id=arxiv_id, doi=doi)
    aliases = []
    if identity.arxiv_id:
        aliases.append(("arxiv", identity.arxiv_id))
    if identity.doi:
        aliases.append(("doi", identity.doi))
    with conn:
        for identity_type, identity_value in aliases:
            conn.execute(
                "INSERT INTO paper_identities "
                "(identity_type, identity_value, content_item_id) VALUES (?, ?, ?) "
                "ON CONFLICT(identity_type, identity_value) DO UPDATE SET "
                "content_item_id = excluded.content_item_id",
                (identity_type, identity_value, content_item_id),
            )
