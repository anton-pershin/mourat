"""Research domain CRUD operations."""

from __future__ import annotations

import sqlite3


# -- Research domains --

def create_research_domain(conn: sqlite3.Connection, id: str, name: str, description: str = "") -> None:
    conn.execute(
        "INSERT INTO research_domains (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def get_research_domain(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM research_domains WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_research_domain(conn: sqlite3.Connection, id: str, name: str | None = None, description: str | None = None) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if description is not None:
        fields.append("description = ?"); values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE research_domains SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_research_domain(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM research_domains WHERE id = ?", (id,))
    conn.commit()


def list_research_domains(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM research_domains ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Research directions --

def create_research_direction(conn: sqlite3.Connection, id: str, name: str, domain_id: str, description: str = "") -> None:
    conn.execute(
        "INSERT INTO research_directions (id, name, description, domain_id) VALUES (?, ?, ?, ?)",
        (id, name, description, domain_id),
    )
    conn.commit()


def get_research_direction(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM research_directions WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_research_direction(conn: sqlite3.Connection, id: str, name: str | None = None, description: str | None = None) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if description is not None:
        fields.append("description = ?"); values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE research_directions SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_research_direction(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM research_directions WHERE id = ?", (id,))
    conn.commit()


def list_research_directions(conn: sqlite3.Connection, domain_id: str | None = None) -> list[dict]:
    if domain_id:
        rows = conn.execute("SELECT * FROM research_directions WHERE domain_id = ? ORDER BY id", (domain_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM research_directions ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Research objects --

def create_research_object(conn: sqlite3.Connection, id: str, name: str, direction_id: str, description: str = "") -> None:
    conn.execute(
        "INSERT INTO research_objects (id, name, description, direction_id) VALUES (?, ?, ?, ?)",
        (id, name, description, direction_id),
    )
    conn.commit()


def get_research_object(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM research_objects WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_research_object(conn: sqlite3.Connection, id: str, name: str | None = None, description: str | None = None) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if description is not None:
        fields.append("description = ?"); values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE research_objects SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_research_object(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM research_objects WHERE id = ?", (id,))
    conn.commit()


def list_research_objects(conn: sqlite3.Connection, direction_id: str | None = None) -> list[dict]:
    if direction_id:
        rows = conn.execute("SELECT * FROM research_objects WHERE direction_id = ? ORDER BY id", (direction_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM research_objects ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Research questions --

def create_research_question(conn: sqlite3.Connection, id: str, name: str, object_id: str, description: str = "") -> None:
    conn.execute(
        "INSERT INTO research_questions (id, name, description, object_id) VALUES (?, ?, ?, ?)",
        (id, name, description, object_id),
    )
    conn.commit()


def get_research_question(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM research_questions WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_research_question(conn: sqlite3.Connection, id: str, name: str | None = None, description: str | None = None) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if description is not None:
        fields.append("description = ?"); values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE research_questions SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_research_question(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM research_questions WHERE id = ?", (id,))
    conn.commit()


def list_research_questions(conn: sqlite3.Connection, object_id: str | None = None) -> list[dict]:
    if object_id:
        rows = conn.execute("SELECT * FROM research_questions WHERE object_id = ? ORDER BY id", (object_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM research_questions ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Research topics --

def create_research_topic(conn: sqlite3.Connection, id: str, name: str, description: str = "") -> None:
    conn.execute(
        "INSERT INTO research_topics (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def get_research_topic(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM research_topics WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_research_topic(conn: sqlite3.Connection, id: str, name: str | None = None, description: str | None = None) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if description is not None:
        fields.append("description = ?"); values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE research_topics SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_research_topic(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM research_topics WHERE id = ?", (id,))
    conn.commit()


def list_research_topics(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM research_topics ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Junction: topic ↔ technical challenges --

def add_topic_technical_challenge(conn: sqlite3.Connection, topic_id: str, challenge_id: str) -> None:
    conn.execute(
        "INSERT INTO topic_technical_challenges (topic_id, challenge_id) VALUES (?, ?)",
        (topic_id, challenge_id),
    )
    conn.commit()


def remove_topic_technical_challenge(conn: sqlite3.Connection, topic_id: str, challenge_id: str) -> None:
    conn.execute(
        "DELETE FROM topic_technical_challenges WHERE topic_id = ? AND challenge_id = ?",
        (topic_id, challenge_id),
    )
    conn.commit()


def list_topic_technical_challenges(conn: sqlite3.Connection, topic_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT tc.* FROM technical_challenges tc "
        "JOIN topic_technical_challenges ttc ON tc.id = ttc.challenge_id "
        "WHERE ttc.topic_id = ? ORDER BY tc.id",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# -- Junction: topic ↔ research questions --

def add_topic_research_question(conn: sqlite3.Connection, topic_id: str, question_id: str) -> None:
    conn.execute(
        "INSERT INTO topic_research_questions (topic_id, question_id) VALUES (?, ?)",
        (topic_id, question_id),
    )
    conn.commit()


def remove_topic_research_question(conn: sqlite3.Connection, topic_id: str, question_id: str) -> None:
    conn.execute(
        "DELETE FROM topic_research_questions WHERE topic_id = ? AND question_id = ?",
        (topic_id, question_id),
    )
    conn.commit()


def list_topic_research_questions(conn: sqlite3.Connection, topic_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT rq.* FROM research_questions rq "
        "JOIN topic_research_questions trq ON rq.id = trq.question_id "
        "WHERE trq.topic_id = ? ORDER BY rq.id",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]
