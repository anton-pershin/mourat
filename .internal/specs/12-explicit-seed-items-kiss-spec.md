# Explicit seed content items for from-seeds collection

## 1. Requirement analysis

`collect_influential_papers_from_seeds` currently seeds its expansion from content items
retrieved via links to configured research attributes (`seed_research_question_ids`,
`seed_technical_challenge_ids`). This spec adds a way to restrict that seed set to
explicitly named content items, selected by their database ids.

### Requirements

- **R1.** The script gains a new config option `seed_content_item_ids: []` — a list of
  DB content-item ids (the same ids the junction tables key on; `retrieve_content.py`
  is where these ids are shown to the user).

- **R2.** Non-empty `seed_content_item_ids` acts as a **filter** over the seed items
  retrieved by the attribute sources: attribute retrieval runs exactly as today over
  `seed_research_question_ids` / `seed_technical_challenge_ids`, and only retrieved items
  whose content-item id appears in `seed_content_item_ids` become seeds. An empty list
  imposes no restriction (backward-compatible). The filter is a set-membership test:
  the explicit list never causes items to be fetched by id, and duplicate ids within
  the list are harmless.

- **R3.** If any explicit id in `seed_content_item_ids` matches no retrieved seed item,
  the run **aborts before expansion** with an error listing the unmatched ids. There is
  no tolerance path: every id in the list must match or the run stops.

- **R4.** `seed_content_item_ids` is seeds-only. It does not contribute scoring
  attributes: the scorer still scores candidates against `seed_research_question_ids` /
  `seed_technical_challenge_ids` plus the optional `research_topic_ids` / `constraint_ids`,
  exactly as today.

### Expected variants of the new behaviour

- **B1.** All-empty seed configuration (no attribute ids, no explicit ids) — unchanged:
  the run aborts with the existing "no seed content items resolved" error.
- **B2.** A filtered-in seed item that is not a paper (e.g. a Reddit post) — not
  special-cased: it flows to `SeedResolver` and is skip-and-reported like any other
  unresolvable seed.
- **B3.** An explicit id naming an item that exists in the DB but is not linked to any
  configured attribute is treated exactly like any other unmatched id: it matches no
  retrieved item, so R3 aborts the run. (This is the intended cost of filter semantics:
  to use an item as a seed, the attribute lists must retrieve it.)

### Non-requirements

- No new components, no pipeline-stage changes, no schema changes. The change is
  confined to seed retrieval/filtering logic and configuration in the from-seeds script.
- No way to add seeds by id on top of attribute retrieval (override semantics) — the
  filter semantics of R2 were chosen deliberately.
- No changes to the other collection scripts (from-scratch, posts, newest papers).

## 2. Tests

All new tests live in `tests/test_paper_collection.py`, extending the existing
`TestSeedItemRetrieval` class (which covers `_retrieve_seed_items`) and reusing its
`_seed_db` fixture (`item1` linked to `q1`, `item2` linked to `tc1`) and `_make_cfg`
helper — the helper gains a `seed_content_item_ids: []` default. Scoring-attribute
isolation is tested via the existing `TestLoadScoringAttributes` fixture. No test
performs a real network request.

- **T1 (R2, filter keeps the subset).** `seed_technical_challenge_ids=["tc1"]`,
  `seed_content_item_ids=["item2"]` → collection contains exactly `item2`; `item1`
  (linked to `q1` only) is filtered out.
- **T2 (R2, empty list = no restriction).** Attribute-only config with
  `seed_content_item_ids=[]` → both items returned. Guards against the filter being
  applied as an empty intersection when the list is empty.
- **T3 (R2, both sources merged before filtering).** `seed_research_question_ids=["q1"]`,
  `seed_technical_challenge_ids=["tc1"]`, `seed_content_item_ids=["item1", "item2"]` →
  both items present, each exactly once (no duplicate despite the list overlapping two
  retrieval sources).
- **T4 (R2, duplicate ids within the list).** `seed_content_item_ids=["item1", "item1"]`
  → exactly one `item1` in the result; no error.
- **T5 (R3, unmatched id aborts before expansion).**
  `seed_technical_challenge_ids=["tc1"]`, `seed_content_item_ids=["ghost-id"]` → raises
  the R3 `ValueError` (the one from `_retrieve_seed_items`, not the script main's
  "no seed content items resolved" abort) with `ghost-id` in the message; the abort
  happens in seed retrieval, so no expansion or resolution is reached.
- **T6 (R3, partial match still aborts).** `seed_content_item_ids=["item2", "ghost-id"]`
  → raises with `ghost-id` named. Guards the "every id must match" rule against a
  partial-match implementation that would silently drop unmatched ids.
- **T7 (R4, explicit list does not touch scoring attributes).**
  `_load_scoring_attributes` called with `seed_content_item_ids=["item1"]` and no other
  attribute ids → all four returned lists (rq, tc, topic, constraint) are empty; the
  explicit id appears in none of them.
- **T8 (B1, all-empty configuration).** Every seed-source list empty → empty collection
  from `_retrieve_seed_items`, same as today (the abort then comes from the existing
  zero-seeds check in the script main).
- **T9 (B2, non-paper item passes the filter).** A Reddit-style content item linked to
  `tc1`, with `seed_content_item_ids` naming only that item → the filter passes it into
  the seed collection. (Its later skip-and-report by `SeedResolver` is out of scope
  here; it is covered by the resolver's own tests.)

## 3. Implementation plan

### 3.1 Implementation repos

Only `mourat` (`/home/tony/reps/github/anton-pershin/mourat`).

### 3.2 Solution design

The change is confined to seed retrieval. `_retrieve_seed_items(conn, cfg)`
(`mourat/scripts/collect_influential_papers_from_seeds.py:79`) gains a post-retrieval
filtering step:

1. Read `seed_content_item_ids` from cfg (default `[]`, read via `cfg.get` like
   `research_topic_ids` — so older configs and tests without the key keep working).
2. After the existing two-source merge and dedup, if the list is non-empty:
   - `kept = [item for item in items if item.id in id_set]`
   - `unmatched = id_set - {item.id for item in items}`; if `unmatched` is non-empty →
     raise `ValueError` listing the unmatched ids (R3). The check is on the explicit
     list *against the retrieved set* — it deliberately does not consult the DB again,
     because under filter semantics an id naming an unlinked existing item must also
     abort (B3).
3. Return the filtered `ContentItemCollection`.

`_load_scoring_attributes` is untouched (R4). The config file gains one commented
option block:

```yaml
# Optional explicit seed selection: when non-empty, only the retrieved seed
# items whose content-item id is listed here become seeds. Every listed id
# must match a retrieved item, otherwise the run aborts.
seed_content_item_ids: []
```

The error path reuses the existing "no seed content items resolved" abort in the script
main only when the whole set is empty; R3's abort is raised inside
`_retrieve_seed_items` with its own message naming the unmatched ids, so the two
failure reasons stay distinguishable.

### 3.3 Todo list

1. [x] Write the tests (T1–T9 in `tests/test_paper_collection.py`; extend `_make_cfg`
       with the new default)
2. [x] Run all the tests and ensure that they fail
3. [x] Add `seed_content_item_ids: []` to
       `config/config_collect_influential_papers_from_seeds.yaml` with the explanatory
       comment
4. [x] Implement the filter + unmatched-id abort in `_retrieve_seed_items`
5. [x] Run the test suite until green (pytest, targeted at `test_paper_collection.py`
       first, then full)
6. [x] Run linters on touched files (black, isort, pylint E, mypy scoped)

### 3.4 Modification summary

| File | Action |
|------|--------|
| `mourat/scripts/collect_influential_papers_from_seeds.py` | Modified: `_retrieve_seed_items` gains the explicit-id filter and unmatched-id abort (R2, R3); docstring updated |
| `config/config_collect_influential_papers_from_seeds.yaml` | Modified: new `seed_content_item_ids: []` option with comment |
| `tests/test_paper_collection.py` | Modified: `_make_cfg` default + new tests T1–T9 in `TestSeedItemRetrieval` |

