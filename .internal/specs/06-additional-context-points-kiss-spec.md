## Additional context points and validated max_score

### 1. Requirement analysis

This spec redesigns the enrichment data flow and fixes one related bug. The common seam is `WebEnricher` → `EnrichedRedditPost` → `PostScorer`.

**Additional context points (redesign of the enrichment output).**
Today `WebEnricher` receives `EnrichmentResult` with `enriched_text` (a full rewritten post text) and `enrichment_summary` (a one-line note). Problems: the LLM regurgitates content already present in the post (wasted output tokens); the rewrite is lossy and unauditable; only `enrichment_summary` is stored on `EnrichedRedditPost` (mourat/data_models.py:271-276) while `enriched_text` is written to monitoring (web_enricher.py:201) and discarded; and `PostScorer` passes `ep.enrichment_summary` where the enriched content was meant to go (mourat/processors/post_scorer.py:114-120), so scoring reads a one-line description instead of the enrichment.

Redesign: the enrichment agent's job becomes extraction of *additional context* — facts obtained from the web that are NOT already evident in the post — emitted as a list of short self-contained points, each phrased like "X is ...". The cap of 5 points exists to bound the agent's output and reduce execution time; it is a prompt-level instruction only.

Requirements:
- R1. `EnrichmentResult` becomes a list of context points: `additional_context: list[str]` (with a max of 5 per post). The fields `enriched_text` and `enrichment_summary` are removed.
- R2. `EnrichedRedditPost` carries `additional_context: list[str] = []` in place of `enrichment_summary`. No other fields added or kept. The field name `additional_context` is used consistently across `EnrichmentResult`, `EnrichedRedditPost`, and `ScoredRedditPost`.
- R3. `WebEnricher` stores the returned list in `additional_context`. The monitoring text for each post includes the original post text and the full list of additional context points (e.g. a numbered list under the post heading).
- R4. The enrichment system prompt is updated to instruct the agent to: (a) use its tools to gather context, (b) return only facts not already evident in the post, each as one short self-contained point, (c) return an empty list when the post needs no enrichment, (d) return at most 5 points. (The user will tune the prompt wording himself; the spec pins the behavioral contract, not the wording.)
- R5. The scorer composition: `_build_scoring_prompt` currently takes `enriched_text: str` and composes `text = enriched_text or post_info.text or "(no text)"`. After the redesign there is no enriched text: the prompt is built from the original post text plus the additional context points. Concretely, `PostScorer._run` passes `ep.additional_context` (a list of strings) and the raw `ep.post.text` to `_build_scoring_prompt`; the prompt contains the original text followed by the numbered context points. If the list is empty, the prompt is identical in shape to today's raw-text case.

**B2. `max_score` computed from unvalidated LLM output.**
`PostScorer._run` validates `result.scores` against `valid_id_type_pairs` — entries whose `(id, type)` pair was not passed to the scorer are dropped from `relevance_scores` (mourat/processors/post_scorer.py:131-135). But `max_score` is computed from the *unvalidated* list (post_scorer.py:136-139): `max((e.score for e in result.scores), default=0)`. A hallucinated id with a high score inflates `max_score`, and `PostScoreFilter` (mourat/filters.py:50+) filters on `p.max_score`, so a phantom entry can push an irrelevant post through the score threshold into the database.

Requirements:
- R6. `max_score` is computed from `relevance_scores` (the validated entries), never from `result.scores`.
- R7. With no valid entries remaining, `max_score` is `0.0` (unchanged `default=0` semantics, now applied to the validated list).

Non-functional requirements:
- NFR1. No monitoring output *format* changes other than the required inclusion of the context points (R3); log lines are untouched.
- NFR2. No Hydra config schema changes: no new config keys, no new LLM aliases, no new defaults files.
- NFR3. Existing tests keep passing except where they assert the removed fields or the buggy `max_score` behavior.

### 2. Tests

Test tooling: LLM-backed components are exercised with `pydantic_ai.models.function.FunctionModel` — the model fn returns `ModelResponse(parts=[TextPart(json_string)])`, and prompts are read via `messages[-1].parts[-1].content`. The enricher's tool closures are not exercised (the mock model never requests tools), so no HTTP mocking is needed.

New module `tests/test_enrichers.py`:

1. `test_enriched_post_carries_additional_context` — `FunctionModel` returns `EnrichmentResult(additional_context=["Point one.", "Point two."])`; assert output `EnrichedRedditPost.additional_context == ["Point one.", "Point two."]`. (R1, R3)
2. `test_empty_additional_context_retained` — `FunctionModel` returns `additional_context=[]`; assert `additional_context == []` and the post is present in the output collection. (R1, R4)
3. `test_monitoring_contains_additional_context` — capture monitoring text via the dummy handler; assert it contains the original post text and both points. (R3)

Existing module `tests/test_processors.py` (PostScorer):

4. `test_scorer_prompt_contains_text_then_points` — `EnrichedRedditPost` with `additional_context=["ctx alpha", "ctx beta"]`, `post.text="RAW TEXT"`; capture the prompt via `FunctionModel`; assert it contains `"RAW TEXT"`, `"ctx alpha"`, `"ctx beta"`, and that the points appear after the raw text. (R5)
5. `test_scorer_prompt_without_context` — `additional_context=[]`; assert the prompt contains `"RAW TEXT"` and no point block. (R5)
6. `test_max_score_ignores_unknown_ids` — scorer configured with one valid RQ id; `FunctionModel` returns two entries: the valid id with score 40 and a hallucinated id with score 95. Assert `relevance_scores` contains only the valid entry, `max_score == 40.0`, and the monitoring line reports `Max score: 40`. (R6)
7. `test_max_score_zero_when_no_valid_entries` — `FunctionModel` returns only a hallucinated id with score 95. Assert `relevance_scores == []`, `max_score == 0.0`, and the post is still present in the output collection (not dropped). (R6, R7)

Regression guard: tests in the existing suite that construct `EnrichedRedditPost(enrichment_summary=...)` or assert on `enriched_text` are updated to the new model shape; all other existing tests pass unchanged (NFR3). The `EnrichmentResult` JSON schema change is exercised implicitly by every `FunctionModel`-driven test above (the mock returns the new shape).

### 3. Implementation plan

#### 3.1 Implementation repos

- `mourat` (management repo = implementation repo, this repo).

#### 3.2 Solution design

Five coordinated edits around the enrich→score seam:

1. **Data models** (mourat/data_models.py):
   - `EnrichmentResult`: replace `enriched_text` + `enrichment_summary` with `additional_context: list[str] = Field(description=...)`.
   - `EnrichedRedditPost`: replace `enrichment_summary` with `additional_context: list[str] = []`.
   - `ScoredRedditPost`: replace `enrichment_summary: str` with `additional_context: list[str] = []` so the scored pipeline keeps carrying the points (it already forwards the field from `EnrichedRedditPost`).
2. **Enricher** (mourat/enrichers/web_enricher.py): `SYSTEM_PROMPT` updated to the extraction contract (only facts not already evident in the post, short self-contained points, at most 5, empty list when no enrichment needed — wording to be tuned by the user after implementation). In `_run`, construct `EnrichedRedditPost(post=post_info, additional_context=result.additional_context)`; the monitoring line per post prints the original text followed by the numbered points. Per-item DEBUG timing lines unchanged.
3. **Scorer** (mourat/processors/post_scorer.py):
   - `_build_scoring_prompt(post_info, additional_context: list[str] | None)`: text block is `post_info.text or "(no text)"`; when `additional_context` is non-empty, a numbered "Additional context:" block is appended after the text block.
   - `_run` passes `ep.additional_context`; `max_score` computed from `relevance_scores` instead of `result.scores`; `ScoredRedditPost` constructed with `additional_context=ep.additional_context`.
4. **Script plumbing** (mourat/scripts/collect_posts.py): the pipeline log line and `save_posts_to_db` reference `enrichment_summary` only implicitly via the collection models — the only required touch is that nothing in the script reads the removed fields (verified; `sp.post` fields and `sp.relevance_scores` only). No change expected; if compilation reveals otherwise, the fix is mechanical field renaming.
5. **Monitoring output** (within `WebEnricher._run`): per-post block becomes title/URL/original text/numbered additional context points. `text_for_monitoring` composition unchanged otherwise.

Pitfalls respected: no step-timing or logging blocks move (timing stays in `Function.__call__` and the existing per-item DEBUG lines); `Function.__call__` returns only the output; `FunctionModel` tests follow the `ModelResponse(parts=[TextPart(json_string)])` pattern; pytest runs before linters since linters miss NameErrors.

#### 3.3 Todo list

1. [ ] Write tests 1-7 (new `tests/test_enrichers.py` + additions to `tests/test_processors.py`)
2. [ ] Run all the tests and ensure that the new ones fail and the suite is otherwise green
3. [ ] Update `EnrichmentResult`, `EnrichedRedditPost`, `ScoredRedditPost` in `mourat/data_models.py`
4. [ ] Update `WebEnricher` (system prompt contract, output construction, monitoring text with points)
5. [ ] Update `PostScorer` (`_build_scoring_prompt` signature, prompt composition, `max_score` from `relevance_scores`)
6. [ ] Verify `collect_posts.py` compiles and needs no changes (field removal fallout check)
7. [ ] Run all the tests and ensure that they pass (`/home/tony/venvs/mourat/bin/python -m pytest tests/ -v --ignore-glob='*import*'`)
8. [ ] Run linters on the touched files: `python -m black`, `isort`, `pylint` (E only), and `mypy` scoped to the modified files (mypy baseline: 3 pre-existing errors in web_enricher.py on main — confirm via `git stash` + rerun before treating any finding as a regression)

#### 3.4 Modification summary

| File | Action |
|------|--------|
| mourat/data_models.py | Modified: `EnrichmentResult` → `additional_context: list[str]`; `EnrichedRedditPost.enrichment_summary` → `additional_context: list[str]`; `ScoredRedditPost` same field swap |
| mourat/enrichers/web_enricher.py | Modified: system prompt (extraction contract, ≤5 points, empty-list case), output construction with `additional_context`, monitoring block with numbered points |
| mourat/processors/post_scorer.py | Modified: `_build_scoring_prompt(post, additional_context)`; prompt composition original-text-then-points; `max_score` from validated entries |
| tests/test_enrichers.py | New: WebEnricher tests 1-3 via FunctionModel |
| tests/test_processors.py | Modified: PostScorer tests 4-7 via FunctionModel; existing tests aligned to the new model shape |
