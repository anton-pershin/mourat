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
3. Set up `/config/user_settings/user_settings.yaml`. Currently, the config relies on Caila API but it is trivial to modify it to your needs
4. Run one of the scripts `/mourat/scripts/XXX.py` and do not forget to modify the corresponding config file in `/config/config_XXX.yaml'
```bash
python kygs/scripts/XXX.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `run_pipeline.py`

Runs a configurable pipeline for collecting, filtering, and scoring arXiv papers based on their relevance to your research topic.

#### Configuration

1. In `user_settings.yaml`, set up your:
   ```yaml
   project_path: /path/to/your/project
   caila_api_key: "your-caila-api-key"  # Required for paper scoring
   ```

2. In `config_run_pipeline.yaml`, modify:
   - `paper_topic`: define your research area (e.g., "bio-inspired visual processing")
   - `problem_being_addressed`: specify your concrete research problem
   - Pipeline parameters:
     - ArxivPaperCollector: `start_date`, `end_date`, `max_results`. ArXiv API has some stupid bug: sometimes, it outputs significantly less papers than specified by `max_results`. Incrementing `max_results` by 10 usually helps
     - BinaryPaperClassifier: no changes needed in general
     - PaperScorer: no changes needed in general
     - ScoreBasedPaperFilter: set `score_threshold` (default: 4)

#### Output

The pipeline generates files in `${result_dir}` (default: `hydra_root/YYYY-MM-DD/HH-MM-SS/`) with step-by-step results (the filename and format are specified by `monitoring_handler`):

1. **ArXiv paper collection**
   - Title and abstract of each paper from arXiv
   - Direct link to the paper

2. **Classification results**
   - Filter out the papers irrelevant to `paper_topic`, the structure of the content remains the same

3. **Scoring Results**
   - Add score (0-5) and detailed justification to each paper

4. **Final Filtered Results**
   - Filter out the papers with scores ≥ threshold (default: 4)

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
