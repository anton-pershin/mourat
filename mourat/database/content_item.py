"""Content item CRUD operations."""

from __future__ import annotations

import sqlite3

# -- Content items --


def create_content_item(
    conn: sqlite3.Connection,
    id: str,
    name: str,
    source_type_id: str,
    platform_id: str,
    influence_metric_id: str,
    description: str = "",
    url: str | None = None,
    published_at: str | None = None,
    authors: str | None = None,
    influence_score: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO content_items "
        "(id, name, description, source_type_id, platform_id, url, published_at, "
        "authors, influence_score, influence_metric_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            id,
            name,
            description,
            source_type_id,
            platform_id,
            url,
            published_at,
            authors,
            influence_score,
            influence_metric_id,
        ),
    )
    conn.commit()


def get_content_item(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM content_items WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_content_item(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
    source_type_id: str | None = None,
    platform_id: str | None = None,
    influence_metric_id: str | None = None,
    url: str | None = None,
    published_at: str | None = None,
    authors: str | None = None,
    influence_score: int | None = None,
) -> None:
    fields = []
    values = []
    for col, val in [
        ("name", name),
        ("description", description),
        ("source_type_id", source_type_id),
        ("platform_id", platform_id),
        ("influence_metric_id", influence_metric_id),
        ("url", url),
        ("published_at", published_at),
        ("authors", authors),
        ("influence_score", influence_score),
    ]:
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE content_items SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_content_item(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM content_items WHERE id = ?", (id,))
    conn.commit()


def list_content_items(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM content_items ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Source types --


def create_source_type(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO source_types (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def list_source_types(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM source_types ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Platforms --


def create_platform(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO platforms (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def list_platforms(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM platforms ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Influence metrics --


def create_influence_metric(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO influence_metrics (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def list_influence_metrics(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM influence_metrics ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Content item references --


def add_content_item_reference(
    conn: sqlite3.Connection, source_id: str, target_id: str
) -> None:
    conn.execute(
        "INSERT INTO content_item_references (source_id, target_id) VALUES (?, ?)",
        (source_id, target_id),
    )
    conn.commit()


def remove_content_item_reference(
    conn: sqlite3.Connection, source_id: str, target_id: str
) -> None:
    conn.execute(
        "DELETE FROM content_item_references WHERE source_id = ? AND target_id = ?",
        (source_id, target_id),
    )
    conn.commit()


def list_content_item_references(
    conn: sqlite3.Connection, content_id: str
) -> list[str]:
    rows = conn.execute(
        "SELECT target_id FROM content_item_references WHERE source_id = ?",
        (content_id,),
    ).fetchall()
    return [r["target_id"] for r in rows]


# -- Content item ↔ technical challenges --


def add_item_technical_challenge(
    conn: sqlite3.Connection,
    content_id: str,
    challenge_id: str,
    justification: str = "",
    relevance_score: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO item_technical_challenges "
        "(content_id, challenge_id, justification, relevance_score) "
        "VALUES (?, ?, ?, ?)",
        (content_id, challenge_id, justification, relevance_score),
    )
    conn.commit()


def remove_item_technical_challenge(
    conn: sqlite3.Connection, content_id: str, challenge_id: str
) -> None:
    conn.execute(
        "DELETE FROM item_technical_challenges "
        "WHERE content_id = ? AND challenge_id = ?",
        (content_id, challenge_id),
    )
    conn.commit()


def list_item_technical_challenges(
    conn: sqlite3.Connection, content_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT tc.*, itc.justification, itc.relevance_score "
        "FROM technical_challenges tc "
        "JOIN item_technical_challenges itc ON tc.id = itc.challenge_id "
        "WHERE itc.content_id = ? ORDER BY tc.id",
        (content_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# -- Content item ↔ research questions --


def add_item_research_question(
    conn: sqlite3.Connection,
    content_id: str,
    question_id: str,
    justification: str = "",
    relevance_score: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO item_research_questions "
        "(content_id, question_id, justification, relevance_score) "
        "VALUES (?, ?, ?, ?)",
        (content_id, question_id, justification, relevance_score),
    )
    conn.commit()


def remove_item_research_question(
    conn: sqlite3.Connection, content_id: str, question_id: str
) -> None:
    conn.execute(
        "DELETE FROM item_research_questions WHERE content_id = ? AND question_id = ?",
        (content_id, question_id),
    )
    conn.commit()


def list_item_research_questions(
    conn: sqlite3.Connection, content_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT rq.*, irq.justification, irq.relevance_score "
        "FROM research_questions rq "
        "JOIN item_research_questions irq ON rq.id = irq.question_id "
        "WHERE irq.content_id = ? ORDER BY rq.id",
        (content_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# -- Content item ↔ research topics --


def add_item_research_topic(
    conn: sqlite3.Connection,
    content_id: str,
    topic_id: str,
    justification: str = "",
    relevance_score: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO item_research_topics "
        "(content_id, topic_id, justification, relevance_score) "
        "VALUES (?, ?, ?, ?)",
        (content_id, topic_id, justification, relevance_score),
    )
    conn.commit()


def remove_item_research_topic(
    conn: sqlite3.Connection, content_id: str, topic_id: str
) -> None:
    conn.execute(
        "DELETE FROM item_research_topics WHERE content_id = ? AND topic_id = ?",
        (content_id, topic_id),
    )
    conn.commit()


def list_item_research_topics(conn: sqlite3.Connection, content_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT rt.*, irt.justification, irt.relevance_score "
        "FROM research_topics rt "
        "JOIN item_research_topics irt ON rt.id = irt.topic_id "
        "WHERE irt.content_id = ? ORDER BY rt.id",
        (content_id,),
    ).fetchall()
    return [dict(r) for r in rows]
