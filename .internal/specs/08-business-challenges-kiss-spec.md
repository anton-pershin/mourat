## Business challenges

### 1. Requirement analysis

**Context.** The DB layer currently models the business side as: `business_domains` → `products` → `high_level_technologies`, plus `technical_challenges` and `constraints` linked to technologies via `technology_challenges` / `technology_constraints` junctions. The "business challenge" concept — the business-level motivation that technical challenges and technologies serve — is absent. This spec adds it at the DB layer only.

**Functional requirements:**

1. **R1 — Business challenges entity.** New table `business_challenges` with the same shape as `constraints` and `technical_challenges`: `id TEXT PRIMARY KEY`, `name TEXT NOT NULL`, `description TEXT` (nullable).
2. **R2 — Technology ↔ business challenge link.** Many-to-many: each high-level technology can be linked to zero or more business challenges, and each business challenge to zero or more technologies. Junction table `technology_business_challenges (technology_id, business_challenge_id)` with composite PK and FKs to `high_level_technologies` and `business_challenges`.
3. **R3 — Technical challenge ↔ business challenge link.** Many-to-many, same shape: junction table `technical_challenge_business_challenges (challenge_id, business_challenge_id)` with composite PK and FKs to `technical_challenges` and `business_challenges`.
4. **R4 — CRUD + link API.** In `mourat/database/business_domain.py`, mirroring the existing `constraints` block exactly:
   - `create/get/update/delete/list_business_challenges`
   - `add/remove/list_technology_business_challenges(conn, technology_id)` — required forward-direction parameter, matching the existing junction listers (no reverse-direction or unfiltered listing).
   - `add/remove/list_technical_challenge_business_challenges(conn, challenge_id)` — same.
5. **R5 — Automatic migration.** No separate migration scripts: adding the tables to `schema.sql` with `CREATE TABLE IF NOT EXISTS` makes `apply_schema` migrate existing DBs on next open. Existing data must remain intact after migration.

**Non-requirements (explicitly out of scope):**

- No changes to `content_items` junctions — business challenges are not scored against content items.
- No scorer/prompt/monitoring changes, no Hydra config changes.
- No retrieval/query-engine support (nothing reads these tables in the pipeline yet; CRUD API only).
- No seed data or scripts that populate business challenges.

**Expected variants:**
- `get_business_challenge` returns `dict | None`; `list_business_challenges` returns `list[dict]` ordered by id, like all existing entity functions.
- Deleting a business challenge cascades via FK default behavior (consistent with existing junction semantics — SQLite default, no ON DELETE clauses, same as `technology_constraints`).

### 2. Tests

All tests go in the existing database test module, following the existing per-entity/junction test patterns. Test list:

**T1 — Business challenge CRUD.**
- `create_business_challenge` inserts; `get_business_challenge` returns the dict; `get` on a missing id returns `None`.
- `update_business_challenge` changes name/description; `get` reflects the change.
- `list_business_challenges` returns all rows ordered by id.
- `delete_business_challenge` removes it; subsequent `get` returns `None`.

**T2 — Technology ↔ business challenge junction.**
- Create a business domain → product → technology (existing helpers), create two business challenges.
- `add_technology_business_challenge` twice; `list_technology_business_challenges(conn, technology_id)` returns both, ordered by id.
- `remove_technology_business_challenge` removes one; listing reflects it.
- Duplicate `add` raises (composite PK / IntegrityError), consistent with existing junctions.

**T3 — Technical challenge ↔ business challenge junction.**
- Same pattern with a technical challenge: add, list (by `challenge_id`), remove, duplicate-add raises.

**T4 — FK enforcement.**
- Adding a junction row with a nonexistent technology_id / challenge_id / business_challenge_id raises IntegrityError (SQLite FK enforcement on, as in existing tests).

### 3. Implementation plan

#### 3.1 Implementation repos

- `/home/tony/reps/github/anton-pershin/mourat` (management repo = implementation repo).

#### 3.2. Solution design

- **Schema** (`mourat/database/schema.sql`): append a `business_challenges` table (id/name/description, same shape as `constraints`) plus two junction tables `technology_business_challenges` and `technical_challenge_business_challenges` (composite PKs, FKs to `high_level_technologies` / `technical_challenges` / `business_challenges`), placed next to the existing business-domain tables. No triggers, no indexes beyond PKs — consistent with existing junctions.
- **API** (`mourat/database/business_domain.py`): a new `business_challenges` CRUD block and two junction function blocks, each mirroring the existing `constraints` / `technology_constraints` block verbatim (same signatures, `conn.commit()` per mutation, joins ordered by target id).
- **Migration**: none — `apply_schema` idempotently applies `schema.sql`; new tables appear on next DB open, and pre-existing data stays intact (R5 holds as a property of the `IF NOT EXISTS`-based design). Explicit old-DB migration testing is out of scope per user decision; migration correctness is covered indirectly by T1–T3 running against a freshly applied schema.
- **Tests**: extend the existing database test module with T1–T4 from Section 2.

#### 3.3 Todo list

1. [ ] Write the tests (T1–T4)
2. [ ] Run all the tests and ensure that they fail
3. [ ] Add the three tables to `schema.sql`
4. [ ] Add `business_challenges` CRUD functions to `business_domain.py`
5. [ ] Add the two junction function blocks to `business_domain.py`
6. [ ] Run all the tests and ensure they pass
7. [ ] Run linters (black, isort, pylint E-only, mypy on changed files) and fix findings

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `mourat/database/schema.sql` | Modified: add `business_challenges`, `technology_business_challenges`, `technical_challenge_business_challenges` tables |
| `mourat/database/business_domain.py` | Modified: add business challenge CRUD block + two junction function blocks |
| `tests/test_database.py` | Modified: add T1–T4 tests |
