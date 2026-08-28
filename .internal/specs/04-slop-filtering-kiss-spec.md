## Slop filtering

### 1. Requirement analysis

#### 1.1 Motivation

In the current `collect_posts` pipeline every collected post is passed to `WebEnricher`,
which is the most expensive component (LLM call plus web fetches per post). Empirically,
the vast majority of collected Reddit posts bear no meaningful content ("slop") and do not
deserve enrichment. Two cheap filtering stages inserted between collection and enrichment
are expected to remove most of them.

Slop is a property of the post alone: a post is slop if it carries no substantive
information whatsoever. It is orthogonal to relevance, which is already handled downstream
by `PostScorer` + `PostScoreFilter` and stays topic-aware. The new stages must remain
topic-agnostic so they can be reused by other collectors.

#### 1.2 Functional requirements

- **FR1.** A heuristic slop filter is added as a separate pipeline step operating on
  `RedditPostCollection` and returning `RedditPostCollection`. It drops posts using
  deterministic rules only, without any LLM call.
- **FR2.** The heuristic rules are: minimum text length; deleted/removed text markers;
  link-only text (the body consists of nothing but a URL); minimum post score; author
  denylist (exact names and regular expressions); title regular expression denylist.
- **FR3.** Every heuristic rule is individually configurable via Hydra and individually
  disableable. No rule has an implicit hardcoded default in the code; all thresholds and
  lists come from the config.
- **FR4.** An LLM-based binary slop classifier is added as a separate pipeline step
  operating on `RedditPostCollection` and returning a collection in which every post is
  marked with a binary verdict (slop / not slop) and a justification.
- **FR5.** The classifier submits posts to the LLM in batches. The batch size is
  configurable. Each post in a batch is identified by its `submission_id` so verdicts can
  be matched back to posts.
- **FR6.** The classifier fails open: a post for which no verdict is returned by the LLM
  (missing id, malformed batch response, failed request) is marked as not slop and is
  retained.
- **FR7.** A slop filter is added as a separate pipeline step which consumes the marked
  collection and returns a `RedditPostCollection` containing only the posts marked as not
  slop, so that `WebEnricher` requires no change.
- **FR8.** The pipeline order in `collect_posts` becomes: collect → heuristic slop filter →
  slop classification → slop filter → enrich → score → score filter → save.
- **FR9.** Slop verdicts are not persisted to the database. Dropped posts are discarded.
- **FR10.** The classifier uses a dedicated LLM instance configured via its own Hydra
  config alias, so it can be changed independently of the enrichment and scoring LLMs.
- **FR11.** The monitoring text of each new step reports, in this order: number of posts in,
  number kept, number dropped, and then the full details of the **dropped** posts (title,
  URL, and the reason for dropping — the rule name for the heuristic filter, the LLM
  justification for the slop filter).

#### 1.3 Non-functional requirements

- **NFR1.** All three new components are `Function[InputT, OutputT]` subclasses from
  `mourat.base`, with Pydantic collection models as input and output types.
- **NFR2.** The classifier LLM is `caila_glm_5_3_flash`, the same model as enrichment and
  scoring, wired through a separate `@classification_llm` alias.
- **NFR3.** The LLM agent uses `output_type=` with a Pydantic model; no raw string parsing.
- **NFR4.** The heuristic filter performs no network access and no LLM call.

#### 1.4 Expected behavioural variants

| Situation | Expected behaviour |
|---|---|
| Post text shorter than the configured minimum | Dropped by the heuristic filter |
| Post text is `[deleted]` or `[removed]` | Dropped by the heuristic filter |
| Post text is a bare URL only | Dropped by the heuristic filter |
| Post score below the configured minimum | Dropped by the heuristic filter |
| Post author matches the author denylist or a bot regex | Dropped by the heuristic filter |
| Post title matches a title denylist regex | Dropped by the heuristic filter |
| A rule is disabled in the config | That rule never drops a post |
| All rules disabled | The heuristic filter is a pass-through |
| LLM marks a post as slop | Post is marked slop and removed by the slop filter |
| LLM marks a post as not slop | Post is retained and reaches the enricher |
| LLM returns no verdict for a post in the batch | Post is marked not slop and retained (fail-open) |
| LLM returns a verdict for an unknown `submission_id` | Verdict is ignored |
| Number of posts exceeds the batch size | Posts are split into several batches, all classified |
| Input collection is empty | Both steps return an empty collection, no LLM call is made |

### 2. Tests

All tests are unit tests run with the project venv
(`/home/tony/venvs/mourat/bin/python -m pytest tests/ -v`). No test performs a real LLM
call or network access: the classifier tests use `pydantic_ai.models.function.FunctionModel`
so that the verdicts returned for each batch are controlled by the test.

#### 2.1 `tests/test_filters.py` — `TestHeuristicSlopFilter`

Covers FR1, FR2, FR3, FR11 and the heuristic rows of 1.4.

| Test | Requirement | Expectation |
|---|---|---|
| `test_drops_short_text` | FR2 | A post whose text is shorter than `min_text_chars` is dropped; a longer one is kept |
| `test_drops_deleted_and_removed` | FR2 | Posts with text `[deleted]`, `[removed]`, `[deleted by user]` are dropped |
| `test_drops_link_only_text` | FR2 | A post whose body is a bare URL is dropped; a post with a URL plus substantive prose is kept |
| `test_drops_below_min_score` | FR2 | A post with `score` below `min_score` is dropped |
| `test_drops_denylisted_author_exact` | FR2 | A post by an author listed in `author_denylist` is dropped |
| `test_drops_denylisted_author_regex` | FR2 | A post by an author matching an `author_regex_denylist` entry (e.g. `.*[Bb]ot$`) is dropped |
| `test_drops_denylisted_title_regex` | FR2 | A post whose title matches a `title_regex_denylist` entry is dropped |
| `test_disabled_rule_never_drops` | FR3 | With `min_text_chars` disabled, a very short post is kept while the other rules still apply |
| `test_all_rules_disabled_is_passthrough` | FR3 | With every rule disabled the output collection equals the input collection |
| `test_empty_collection` | 1.4 | An empty input produces an empty output without error |
| `test_monitoring_reports_counts_and_dropped_posts` | FR11 | The monitoring text contains the in/kept/dropped counts and, for each dropped post, its title, URL and the name of the rule that dropped it; it does not list kept posts |
| `test_returns_reddit_post_collection` | FR1, NFR1 | The output is a `RedditPostCollection` instance |

#### 2.2 `tests/test_filters.py` — `TestSlopFilter`

Covers FR7 and FR11.

| Test | Requirement | Expectation |
|---|---|---|
| `test_keeps_only_non_slop` | FR7 | Given a marked collection with a mix of verdicts, only the posts marked not slop are returned |
| `test_output_is_reddit_post_collection` | FR7 | The output type is `RedditPostCollection`, so it can be fed directly to `WebEnricher` |
| `test_all_slop_yields_empty` | FR7 | If every post is marked slop the output is empty |
| `test_none_slop_yields_all` | FR7 | If no post is marked slop every post is retained |
| `test_monitoring_reports_dropped_with_justification` | FR11 | The monitoring text contains the in/kept/dropped counts and the LLM justification for each dropped post |

#### 2.3 `tests/test_classifiers.py` — `TestPostSlopClassifier`

Covers FR4, FR5, FR6, FR10, NFR1 and NFR3.

| Test | Requirement | Expectation |
|---|---|---|
| `test_marks_each_post_with_binary_verdict` | FR4 | Every input post appears in the output with an `is_slop` boolean and a justification |
| `test_output_collection_preserves_all_posts` | FR4 | The classifier marks but does not drop: output length equals input length |
| `test_batching_splits_by_batch_size` | FR5 | With 5 posts and `batch_size=2`, the model is invoked 3 times; every post receives a verdict |
| `test_single_batch_when_below_batch_size` | FR5 | With 3 posts and `batch_size=10`, the model is invoked once |
| `test_verdicts_matched_by_submission_id` | FR5 | When the model returns verdicts in an order different from the input order, each verdict is attached to the post with the matching `submission_id` |
| `test_missing_verdict_fails_open` | FR6 | A post for which the model returns no verdict is marked not slop |
| `test_model_error_fails_open` | FR6 | If the model raises `UnexpectedModelBehavior` for a batch, every post of that batch is marked not slop and the remaining batches are still processed |
| `test_unknown_submission_id_is_ignored` | FR6 | A verdict whose `submission_id` matches no input post is discarded and does not raise |
| `test_empty_collection_makes_no_llm_call` | 1.4 | An empty input produces an empty output and the model is never invoked |
| `test_monitoring_reports_counts_and_slop_posts` | FR11 | The monitoring text contains the in/kept/dropped counts and the justification of each post marked slop |

#### 2.4 `tests/test_imports.py`

Covers NFR1.

| Test | Requirement | Expectation |
|---|---|---|
| `test_import_post_slop_classifier` | NFR1 | `mourat.classifiers` imports in a fresh subprocess with the new classifier present |

### 3. Implementation plan

#### 3.1 Implementation repos

`mourat` (`/home/tony/reps/github/anton-pershin/mourat`) is both the management repo and
the only implementation repo for this spec.

#### 3.2. Solution design

##### Pipeline

```
1 collect         RedditPostCollector    -> RedditPostCollection
2 heuristic       HeuristicSlopFilter    -> RedditPostCollection          (no LLM)
3 classify        PostSlopClassifier     -> ClassifiedRedditPostCollection (batched LLM)
4 slop filter     SlopFilter             -> RedditPostCollection
5 enrich          WebEnricher            -> EnrichedRedditPostCollection
6 score           PostScorer             -> ScoredRedditPostCollection
7 score filter    PostScoreFilter        -> ScoredRedditPostCollection
8 save            save_posts_to_db
```

Steps 5–8 are unchanged. Step 4 emits a plain `RedditPostCollection`, so `WebEnricher`
needs no modification.

##### Data models (`mourat/data_models.py`)

```python
class SlopVerdict(BaseModel):
    submission_id: str
    is_slop: bool
    justification: str

class SlopClassificationResult(BaseModel):   # LLM agent output_type
    verdicts: list[SlopVerdict]

class ClassifiedRedditPost(BaseModel):
    post: RedditPostInfo
    is_slop: bool = False
    justification: str = ""

class ClassifiedRedditPostCollection(BaseModel):
    posts: list[ClassifiedRedditPost]
```

##### `HeuristicSlopFilter` (`mourat/filters.py`)

`Function[RedditPostCollection, RedditPostCollection]`. Constructor parameters, one per
rule, each disabled by passing `None` (thresholds) or an empty list (denylists):

- `min_text_chars: int | None`
- `deleted_markers: list[str]` — matched against the stripped, lowercased text
- `drop_link_only: bool` — the stripped text matches a bare-URL regex
- `min_score: int | None`
- `author_denylist: list[str]` — exact match
- `author_regex_denylist: list[str]`
- `title_regex_denylist: list[str]`

Rules are evaluated in the order above; the first rule that matches drops the post and its
name is recorded as the drop reason. Regexes are compiled once in `__init__`.

##### `PostSlopClassifier` (`mourat/classifiers.py`)

`Function[RedditPostCollection, ClassifiedRedditPostCollection]`. Constructor takes
`monitoring_handler`, `model`, `system_prompt`, `batch_size`, `max_text_chars`
(per-post truncation in the prompt), `model_settings` and `retries`. The agent is built
with `output_type=SlopClassificationResult`.

`_run` slices the input into batches of `batch_size`, builds one prompt per batch
containing a JSON array of `{submission_id, subreddit, title, text}` (text truncated to
`max_text_chars`), and calls `agent.run_sync` once per batch. Returned verdicts are indexed
by `submission_id`; verdicts with an unknown id are ignored. A post with no verdict, and
every post of a batch whose call raised, defaults to `is_slop=False` — fail-open. An empty
input returns an empty collection without any model call.

##### `SlopFilter` (`mourat/filters.py`)

`Function[ClassifiedRedditPostCollection, RedditPostCollection]`. Keeps posts with
`is_slop=False` and unwraps them back to `RedditPostInfo`.

##### Monitoring

Each of the three new steps returns a monitoring text beginning with a summary line
(`in`, `kept`, `dropped`, percentage dropped) followed by a `### title` / `URL` / reason
block per **dropped** post — the rule name for `HeuristicSlopFilter`, the LLM justification
for `PostSlopClassifier` and `SlopFilter`. Kept posts are not listed.

##### Configuration

New config group file `config/slop_filter/default.yaml` holding the heuristic thresholds
and denylists. New `slop_classifier` and `slop_verdict_filter` sections in
`config/config_collect_posts.yaml`, all with `_partial_: true`. The classification LLM is
added to the defaults list as `- llm/caila_glm_5_3_flash@classification_llm`.

#### 3.3 Todo list

1. [ ] Write the tests listed in section 2 (`tests/test_filters.py`, new
   `tests/test_classifiers.py`, `tests/test_imports.py`)
2. [ ] Run all the tests and ensure that they fail
3. [ ] Add `SlopVerdict`, `SlopClassificationResult`, `ClassifiedRedditPost` and
   `ClassifiedRedditPostCollection` to `mourat/data_models.py`
4. [ ] Implement `HeuristicSlopFilter` in `mourat/filters.py`
5. [ ] Implement `PostSlopClassifier` in `mourat/classifiers.py`
6. [ ] Implement `SlopFilter` in `mourat/filters.py`
7. [ ] Create `config/slop_filter/default.yaml` with the heuristic rule settings
8. [ ] Add the `classification_llm` default, the `slop_classifier` and
   `slop_verdict_filter` sections to `config/config_collect_posts.yaml`, and renumber the
   pipeline step ids
9. [ ] Wire the three new steps into `mourat/scripts/collect_posts.py` between collection
   and enrichment, and extend the final summary print with the slop counts
10. [ ] Run all the tests and ensure that they pass
11. [ ] Run `./run_linters.sh` and fix any reported issue
12. [ ] Run `collect_posts` manually once and inspect the monitoring files to confirm FR8
    and the drop reporting

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `mourat/data_models.py` | Modified: add `SlopVerdict`, `SlopClassificationResult`, `ClassifiedRedditPost`, `ClassifiedRedditPostCollection` |
| `mourat/filters.py` | Modified: add `HeuristicSlopFilter` and `SlopFilter` |
| `mourat/classifiers.py` | Modified: add `PostSlopClassifier` |
| `mourat/scripts/collect_posts.py` | Modified: insert steps 2–4, renumber step ids, extend the summary print |
| `config/config_collect_posts.yaml` | Modified: add `classification_llm` and `slop_filter` to defaults, add `slop_classifier` and `slop_verdict_filter` sections |
| `config/slop_filter/default.yaml` | New |
| `tests/test_filters.py` | Modified: add `TestHeuristicSlopFilter` and `TestSlopFilter` |
| `tests/test_classifiers.py` | New |
| `tests/test_imports.py` | Modified: add the classifier import test |
