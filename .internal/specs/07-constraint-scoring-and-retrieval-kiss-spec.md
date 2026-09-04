## Constraint scoring and retrieval (adapting DB layer + collect_posts to constraint-aware content items)

### 1. Requirement analysis

**Context**: The constitution spec was updated (commit 9b0852b) so that each content item has filled relevance attributes and a relevance score for each related research question, technical challenge, **constraint** and research topic; the content item database must support retrieval by constraints in addition to keywords, RQs, TCs and topics.

**Requirements**:

- **R1 — Schema**: Add an `item_constraints` junction table (content_id, constraint_id, justification, relevance_score) mirroring the existing `item_research_topics` table, with a schema migration for existing databases.
- **R2 — Database layer**: Add `add_item_constraint` / `remove_item_constraint` / `list_item_constraints` CRUD functions in `mourat/database/content_item.py`, mirroring the research-topic functions (same signature shape: justification + optional relevance_score).
- **R3 — Retrieval**: Add `search_by_constraint(conn, constraint_id, min_score=None)` in `mourat/database/query_engine.py`, mirroring `search_by_research_topic` (join through the junction, optional score threshold).
- **R4 — Scoring**: `PostScorer` must score content items against **all constraints** in the database:
  - Add a `constraint_list` parameter (id, name, description, type="constraint") alongside rq_list / tc_list / topic_list.
  - Constraints are included in the scoring prompt and in `valid_id_type_pairs`.
  - `ScoreEntry.type` `Literal` must be extended with `"constraint"`.
- **R5 — max_score semantics**: `PostScorer.max_score` must be computed over RQ/TC/topic scores **only**, excluding constraint scores. Rationale: a content item may be relevant to a technical challenge while failing constraints (low constraint scores); the constraint scores must not affect `max_score` (used for monitoring and downstream filtering). This is intentional: it is acceptable for a saved content item to be relevant to some TC without satisfying any constraint.
- **R6 — Save path**: `save_posts_to_db` in `collect_posts.py` must dispatch `type == "constraint"` score entries to `add_item_constraint`. Constraint entries follow the same threshold behavior as other types: only entries surviving the score filter are saved. No additional constraint-specific filtering is introduced.
- **R7 — Collection wiring**: `collect_posts_main` must load all constraints from the DB (via existing business-domain CRUD, analogous to `list_research_topics`) and pass them to the scorer as `constraint_list`.

**Expected variants**:
- A post relevant to a TC but failing all constraints → saved with TC link and no (or below-threshold) constraint links; `max_score` reflects the TC score.
- Constraint scores behave exactly like topic scores wrt thresholding and saving.
- Empty constraint table → scoring proceeds with no constraint targets; behavior identical to today.

### 2. Tests

All tests go into the existing files, following current conventions (`tests/test_database.py`, `tests/test_processors.py`).

**Database layer (tests/test_database.py)**:

- `TestContentItem`:
  - `test_add_list_remove_item_constraint` — add a constraint link with justification + relevance_score; `list_item_constraints` returns it joined with constraint fields; remove and verify empty. Mirrors the topic-link test.
  - `test_add_item_constraint_duplicate` — adding the same (content_id, constraint_id) twice behaves like other junctions (INSERT fails / handled the same way as topic duplicates).
- `TestQueryEngine`:
  - `test_search_by_constraint` — items linked to a constraint are returned with relevance_score; `min_score` filters correctly; unlinked items excluded. Mirrors `test_search_by_research_topic`.
- Schema migration:
  - `test_migrate_add_item_constraints` — open a database created with the old schema (without `item_constraints`), run migration, verify table exists and existing data is intact; opening a fresh DB creates the table via schema.sql.

**Scorer (tests/test_processors.py)** — extend `TestPostScorerAdditionalContextAndMaxScore` (or sibling class, using the existing fake/mocked LLM agent pattern):

- `test_constraint_scores_returned` — with a `constraint_list`, the scorer returns `ScoreEntry`s of `type="constraint"` for valid constraint ids.
- `test_max_score_excludes_constraints` — post scored against a TC (score 90) and a constraint (score 20, or any value) → `max_score == 90`, proving constraint scores don't affect it.
- `test_max_score_constraint_only` — post with scores only for constraints → `max_score == 0` (no RQ/TC/topic scores), while constraint entries are still present in `relevance_scores`.
- `test_invalid_constraint_ids_filtered` — constraint entries with unknown ids are dropped by `valid_id_type_pairs` filtering, as for other types.

**Save path / wiring**: no tests (per user decision — skip `save_posts_to_db` tests); save path correctness is covered indirectly by the DB and scorer tests.

### 3. Implementation plan

#### 3.1 Implementation repos

- `mourat` (management repo = implementation repo; single repo).

#### 3.2 Solution design

Three layers, each mirroring an existing pattern:

1. **Storage**: `item_constraints` junction in `schema.sql` (content_id FK → content_items, constraint_id FK → constraints, justification, relevance_score; PK (content_id, constraint_id)). Existing DBs are migrated automatically because `apply_schema` is idempotent (`CREATE TABLE IF NOT EXISTS`) — no migration code.
2. **Database API**: constraint-link CRUD in `content_item.py` + `search_by_constraint` in `query_engine.py`, copied from the research-topic counterparts.
3. **Scoring pipeline**: `ScoreEntry.type` Literal extended with `"constraint"`; `PostScorer` gains `constraint_list` (prompt, valid pairs) but computes `max_score` over non-constraint scores only; `collect_posts.py` loads all constraints (`bd.list_constraints`) into `scoring_constraint_list` (same formatting as topics) and adds the `constraint` save dispatch branch.

#### 3.3 Todo list

1. [ ] Write the tests (section 2)
2. [ ] Run all the tests and ensure that they fail
3. [ ] Add `item_constraints` table to `schema.sql`
4. [ ] Implement `add_item_constraint` / `remove_item_constraint` / `list_item_constraints` in `mourat/database/content_item.py`
5. [ ] Implement `search_by_constraint` in `mourat/database/query_engine.py`
6. [ ] Extend `ScoreEntry.type` Literal with `"constraint"` in data models
7. [ ] Extend `PostScorer`: `constraint_list` param, prompt section, `valid_id_type_pairs`; exclude constraint scores from `max_score`
8. [ ] Wire `collect_posts.py`: load constraints, build `scoring_constraint_list`, pass to scorer; add `constraint` branch in `save_posts_to_db`
9. [ ] Run all tests and ensure they pass

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `mourat/database/schema.sql` | Modified: add `item_constraints` junction table |
| `mourat/database/content_item.py` | Modified: add item↔constraint CRUD functions |
| `mourat/database/query_engine.py` | Modified: add `search_by_constraint` |
| `mourat/data_models` (ScoreEntry) | Modified: extend `type` Literal with `"constraint"` |
| `mourat/processors/post_scorer.py` | Modified: `constraint_list` param, prompt, valid pairs, max_score exclusion |
| `mourat/scripts/collect_posts.py` | Modified: load constraints, scorer wiring, save dispatch branch |
| `tests/test_database.py` | Modified: constraint CRUD, search, schema tests |
| `tests/test_processors.py` | Modified: scorer constraint tests |
