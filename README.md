# mourat
LLM-based pipeline for paper review

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n mourat python=3.13
conda activate mourat
```
2. Install necessary packages
```bash
pip install -r requirements.txt
```
3. Set up environment variables mentioned in `/config/user_settings/user_settings.yaml`. Currently, the config relies on Caila API but it is trivial to modify it to your needs
4. Run one of the scripts `/mourat/scripts/XXX.py` and do not forget to modify the corresponding config file in `/config/config_XXX.yaml'
```bash
python -m mourat.scripts.XXX
```

## Scripts

### `collect_influential_papers_from_scratch.py`

Discovers influential papers **from scratch**: an LLM agent with web search proposes
candidate papers for the configured research attributes, then the pipeline resolves,
assesses, verifies, scores, filters, and stores them.

Pipeline: discover (LLM agent with web search) → resolve (OpenAlex/arXiv metadata,
unresolved candidates are dropped) → assess influence (normalised fwci /
citations-per-year → 0-100) → verify arXiv PDF (records a verified PDF url or an empty
url with a reason) → score relevance (0-100 per research attribute, with constraints
contributing to the filtering score) → threshold filter → write.

#### Configuration (`config_collect_influential_papers_from_scratch.yaml`)

- Research attributes (research question / technical challenge / topic / constraints)
  are referenced by their DB ids; the pipeline loads their text from the database.
- Discovery, resolution, assessment, and the scoring threshold are configured via the
  corresponding component config groups (Hydra); all budgets are configurable.
- Output: `ContentItemDbWriter` (database) and/or `JsonlWriter` (one JSON line per
  paper, including relevance scores and justifications) — each independently enabled.

### `collect_influential_papers_from_seeds.py`

Expands papers **from seeds**: retrieves content items already stored as relevant to
the configured research attributes, resolves each seed, and generates candidates three
ways — forward citation expansion (works citing each seed), relevance-ranked search
built from seed titles, and backward expansion (the works each seed cites). The
candidates are merged and de-duplicated by OpenAlex work id, then pass through the same
resolve → assess → verify → score → filter → write tail as the from-scratch script,
with one addition: a **seed-derived influence floor** (a one-sided threshold derived
from the normalised influence of the seed set itself) applied before scoring. Each
candidate's provenance records which generator(s) produced it.

#### Configuration (`config_collect_influential_papers_from_seeds.yaml`)

- Same research-attribute references as the from-scratch script.
- Per-generator candidate budgets (forward / search / backward) and the influence-floor
  derivation rule come from Hydra config.
- `seed_content_item_ids` (optional): restricts seed retrieval to an explicit list of
  content-item ids (an intersection with the attribute-retrieved seeds). An empty list
  means no restriction; a listed id that matches no retrieved seed aborts the run.
- Writers are configured identically to the from-scratch script.

### `collect_posts.py`

Collects Reddit posts, enriches them with web-retrieved content, scores them against
the configured research attributes, and saves the results to the content database.
This is the replacement for the deprecated `print_reddit_summary.py`.

#### Configuration (`config_collect_posts.yaml`)

- Reddit client credentials come from `user_settings.yaml`.
- Research attributes for scoring are loaded from the database (no hardcoded lists).
- Scoring uses `PostContentItemScorer`; its `score_filter` threshold is configured
  inline in the config.

### `collect_newest_papers.py`

Collects recent arXiv papers, classifies them against your research topic with a
binary LLM classifier, scores the survivors, and filters by score. A lighter-weight
alternative to the influential-paper pipelines when you only care about new arXiv
submissions.

#### Configuration (`config_collect_newest_papers.yaml`)

- ArxivPaperCollector: `start_date`, `end_date`, `max_results`. ArXiv API has some
  stupid bug: sometimes, it outputs significantly less papers than specified by
  `max_results`. Incrementing `max_results` by 10 usually helps
- BinaryPaperClassifier: no changes needed in general
- PaperScorer: no changes needed in general
- ScoreBasedPaperFilter: set `score_threshold` (default: 4)

### `retrieve_content.py`

Query the content database from the CLI (not a pipeline — a read-only query tool):

```bash
python -m mourat.scripts.retrieve_content db_path=/path/to/mourat.db query_type=keywords keyword_query="transformer"
```

Query types: `keywords`, `research_question`, `technical_challenge`, `research_topic`,
`influence_score`.

### Deprecated / auxiliary scripts

- `collect_recent_influential_papers.py` — legacy Semantic Scholar-based pipeline; the
  S2 API is unreachable unauthenticated, so this script is effectively non-functional
  and superseded by `collect_influential_papers_from_scratch.py`.
- `generate_queries.py` — generates search queries for a candidate topic via LLM.
- `assess_candidate_topic.py` — assesses a candidate topic against the business
  hierarchy in the database.
- `print_reddit_summary.py` — **deprecated**, superseded by `collect_posts.py`.

## Storage layer

The project includes a SQLite-based content database for storing papers, posts, research metadata, and their relevance scores.

### Create the database

```python
from mourat.database import init_db

conn = init_db("/path/to/mourat.db")
conn.close()
```

This creates the file and applies the full schema (all tables, indexes, and FTS5 full-text search).

### Populate research metadata

Research metadata is organized into two hierarchies:

**Research hierarchy:** domain → direction → object → question

```python
from mourat.database import init_db
from mourat.database import research_domain as rd

conn = init_db("/path/to/mourat.db")

rd.create_research_domain(conn, "ai", "Artificial Intelligence", "Broad AI field")
rd.create_research_direction(conn, "ai-llm", "Large Language Models", "ai")
rd.create_research_object(conn, "ai-llm-rlhf", "RLHF", "ai-llm")
rd.create_research_question(conn, "ai-llm-rlhf-rq1", "Does RLHF improve safety?", "ai-llm-rlhf")

conn.close()
```

**Business hierarchy:** domain → product → technology → (challenges, constraints)

```python
from mourat.database import business_domain as bd

bd.create_business_domain(conn, "cloud", "Cloud Computing")
bd.create_product(conn, "cloud-storage", "Storage", "cloud")
bd.create_technology(conn, "cloud-storage-s3", "Object Storage", "cloud-storage")
bd.create_technical_challenge(conn, "ch-durability", "Data Durability")
bd.add_technology_challenge(conn, "cloud-storage-s3", "ch-durability")

conn.close()
```

Research topics can be linked to both technical challenges and research questions:

```python
rd.create_research_topic(conn, "topic-rlhf-safety", "RLHF for Safety")
rd.add_topic_technical_challenge(conn, "topic-rlhf-safety", "ch-durability")
rd.add_topic_research_question(conn, "topic-rlhf-safety", "ai-llm-rlhf-rq1")
```

### Retrieve content

Once the database is populated, query it via the CLI script:

```bash
python -m mourat.scripts.retrieve_content db_path=/path/to/mourat.db query_type=keywords keyword_query="transformer"
```

Query types: `keywords`, `research_question`, `technical_challenge`, `research_topic`, `influence_score`.
