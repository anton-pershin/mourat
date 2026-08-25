"""Business domain CRUD operations."""

from __future__ import annotations

import sqlite3

# -- Business domains --


def create_business_domain(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO business_domains (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def get_business_domain(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM business_domains WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_business_domain(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(
        f"UPDATE business_domains " f"SET {', '.join(fields)} " f"WHERE id = ?", values
    )
    conn.commit()


def delete_business_domain(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM business_domains WHERE id = ?", (id,))
    conn.commit()


def list_business_domains(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM business_domains ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Products --


def create_product(
    conn: sqlite3.Connection, id: str, name: str, domain_id: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO products (id, name, description, domain_id) "
        "VALUES (?, ?, ?, ?)",
        (id, name, description, domain_id),
    )
    conn.commit()


def get_product(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM products WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_product(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE products SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_product(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM products WHERE id = ?", (id,))
    conn.commit()


def list_products(conn: sqlite3.Connection, domain_id: str | None = None) -> list[dict]:
    if domain_id:
        rows = conn.execute(
            "SELECT * FROM products WHERE domain_id = ? ORDER BY id", (domain_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM products ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- High-level technologies --


def create_technology(
    conn: sqlite3.Connection, id: str, name: str, product_id: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO high_level_technologies (id, name, description, product_id) "
        "VALUES (?, ?, ?, ?)",
        (id, name, description, product_id),
    )
    conn.commit()


def get_technology(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM high_level_technologies WHERE id = ?", (id,)
    ).fetchone()
    return dict(row) if row else None


def update_technology(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(
        f"UPDATE high_level_technologies " f"SET {', '.join(fields)} " f"WHERE id = ?",
        values,
    )
    conn.commit()


def delete_technology(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM high_level_technologies WHERE id = ?", (id,))
    conn.commit()


def list_technologies(
    conn: sqlite3.Connection, product_id: str | None = None
) -> list[dict]:
    if product_id:
        rows = conn.execute(
            "SELECT * FROM high_level_technologies "
            "WHERE product_id = ? "
            "ORDER BY id",
            (product_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM high_level_technologies ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


# -- Technical challenges --


def create_technical_challenge(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO technical_challenges (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def get_technical_challenge(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM technical_challenges WHERE id = ?", (id,)
    ).fetchone()
    return dict(row) if row else None


def update_technical_challenge(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(
        f"UPDATE technical_challenges SET {', '.join(fields)} WHERE id = ?", values
    )
    conn.commit()


def delete_technical_challenge(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM technical_challenges WHERE id = ?", (id,))
    conn.commit()


def list_technical_challenges(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM technical_challenges ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Constraints --


def create_constraint(
    conn: sqlite3.Connection, id: str, name: str, description: str = ""
) -> None:
    conn.execute(
        "INSERT INTO constraints (id, name, description) VALUES (?, ?, ?)",
        (id, name, description),
    )
    conn.commit()


def get_constraint(conn: sqlite3.Connection, id: str) -> dict | None:
    row = conn.execute("SELECT * FROM constraints WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else None


def update_constraint(
    conn: sqlite3.Connection,
    id: str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    fields = []
    values = []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    if not fields:
        return
    values.append(id)
    conn.execute(f"UPDATE constraints SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def delete_constraint(conn: sqlite3.Connection, id: str) -> None:
    conn.execute("DELETE FROM constraints WHERE id = ?", (id,))
    conn.commit()


def list_constraints(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM constraints ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# -- Junction: technology ↔ challenges --


def add_technology_challenge(
    conn: sqlite3.Connection, technology_id: str, challenge_id: str
) -> None:
    conn.execute(
        "INSERT INTO technology_challenges (technology_id, challenge_id) "
        "VALUES (?, ?)",
        (technology_id, challenge_id),
    )
    conn.commit()


def remove_technology_challenge(
    conn: sqlite3.Connection, technology_id: str, challenge_id: str
) -> None:
    conn.execute(
        "DELETE FROM technology_challenges "
        "WHERE technology_id = ? AND challenge_id = ?",
        (technology_id, challenge_id),
    )
    conn.commit()


def list_technology_challenges(
    conn: sqlite3.Connection, technology_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT tc.* FROM technical_challenges tc "
        "JOIN technology_challenges ttc ON tc.id = ttc.challenge_id "
        "WHERE ttc.technology_id = ? ORDER BY tc.id",
        (technology_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# -- Junction: technology ↔ constraints --


def add_technology_constraint(
    conn: sqlite3.Connection, technology_id: str, constraint_id: str
) -> None:
    conn.execute(
        "INSERT INTO technology_constraints (technology_id, constraint_id) "
        "VALUES (?, ?)",
        (technology_id, constraint_id),
    )
    conn.commit()


def remove_technology_constraint(
    conn: sqlite3.Connection, technology_id: str, constraint_id: str
) -> None:
    conn.execute(
        "DELETE FROM technology_constraints "
        "WHERE technology_id = ? AND constraint_id = ?",
        (technology_id, constraint_id),
    )
    conn.commit()


def list_technology_constraints(
    conn: sqlite3.Connection, technology_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT c.* FROM constraints c "
        "JOIN technology_constraints ttc ON c.id = ttc.constraint_id "
        "WHERE ttc.technology_id = ? ORDER BY c.id",
        (technology_id,),
    ).fetchall()
    return [dict(r) for r in rows]
