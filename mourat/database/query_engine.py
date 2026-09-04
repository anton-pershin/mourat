"""Query engine for content retrieval."""

from __future__ import annotations

import sqlite3


def search_by_keywords(
    conn: sqlite3.Connection, query: str, min_relevance: int | None = None
) -> list[dict]:
    """Full-text search via FTS5 with optional relevance score filter.

    Supports boolean operators (AND, OR, NOT) and prefix matching (term*).
    """
    sql = """
        SELECT ci.*
        FROM content_items ci
        JOIN content_items_fts fts ON ci.rowid = fts.rowid
        WHERE content_items_fts MATCH ?
    """
    params: list = [query]

    if min_relevance is not None:
        sql += " AND ci.influence_score >= ?"
        params.append(min_relevance)

    sql += " ORDER BY fts.rank"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_research_question(
    conn: sqlite3.Connection, question_id: str, min_score: int | None = None
) -> list[dict]:
    """Retrieve content items linked to a research question."""
    sql = """
        SELECT ci.*
        FROM content_items ci
        JOIN item_research_questions irq ON ci.id = irq.content_id
        WHERE irq.question_id = ?
    """
    params: list = [question_id]

    if min_score is not None:
        sql += " AND irq.relevance_score >= ?"
        params.append(min_score)

    sql += " ORDER BY irq.relevance_score DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_technical_challenge(
    conn: sqlite3.Connection, challenge_id: str, min_score: int | None = None
) -> list[dict]:
    """Retrieve content items linked to a technical challenge."""
    sql = """
        SELECT ci.*
        FROM content_items ci
        JOIN item_technical_challenges itc ON ci.id = itc.content_id
        WHERE itc.challenge_id = ?
    """
    params: list = [challenge_id]

    if min_score is not None:
        sql += " AND itc.relevance_score >= ?"
        params.append(min_score)

    sql += " ORDER BY itc.relevance_score DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_research_topic(
    conn: sqlite3.Connection, topic_id: str, min_score: int | None = None
) -> list[dict]:
    """Retrieve content items linked to a research topic."""
    sql = """
        SELECT ci.*
        FROM content_items ci
        JOIN item_research_topics irt ON ci.id = irt.content_id
        WHERE irt.topic_id = ?
    """
    params: list = [topic_id]

    if min_score is not None:
        sql += " AND irt.relevance_score >= ?"
        params.append(min_score)

    sql += " ORDER BY irt.relevance_score DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_constraint(
    conn: sqlite3.Connection, constraint_id: str, min_score: int | None = None
) -> list[dict]:
    """Retrieve content items linked to a constraint."""
    sql = """
        SELECT ci.*, ic.relevance_score
        FROM content_items ci
        JOIN item_constraints ic ON ci.id = ic.content_id
        WHERE ic.constraint_id = ?
    """
    params: list = [constraint_id]

    if min_score is not None:
        sql += " AND ic.relevance_score >= ?"
        params.append(min_score)

    sql += " ORDER BY ic.relevance_score DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_influence_score(
    conn: sqlite3.Connection, min_score: int, max_score: int = 100
) -> list[dict]:
    """Retrieve content items filtered by influence score range."""
    rows = conn.execute(
        "SELECT * FROM content_items "
        "WHERE influence_score BETWEEN ? AND ? "
        "ORDER BY influence_score DESC",
        (min_score, max_score),
    ).fetchall()
    return [dict(r) for r in rows]
