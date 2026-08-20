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

3. Set up `/config/user_settings/user_settings.yaml`. Specify `project_path` and
   any API keys required by the model configurations you use.
4. Run one of the scripts from `/mourat/scripts/` and modify the corresponding
   `/config/config_XXX.yaml` file when necessary.

```bash
python mourat/scripts/XXX.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `collect_newest_papers.py`

Collects recent papers from arXiv and selects papers relevant to the topics
configured in `config/paper_assigner/paper_assigner.yaml`.

The pipeline consists of three steps:

1. Collect recent papers from arXiv.
2. Assign potentially relevant papers to topics using a fast LLM.
3. Review the resulting shortlist using a slower and more accurate LLM.

The fast and slow models and the collector are selected in
`config/config_collect_newest_papers.yaml`. By default, both models use the
OpenAI-compatible local endpoint configured in `config/llm/local.yaml`.
Alternative model configurations are available in `config/llm/`.

Run the script from the repository root:

```bash
python mourat/scripts/collect_newest_papers.py
```

This default command requires an OpenAI-compatible model service running at
`http://localhost:9191/v1`. To run the pipeline with the included OpenRouter
configurations instead, export your API key and select the fast and slow models:

```bash
export OPENROUTER_API_KEY="your-api-key"

python mourat/scripts/collect_newest_papers.py \
  '+llm@fast_llm=openrouter_nemotron_3_nano' \
  '+llm@slow_llm=openrouter_owl_alpha' \
  'collector=arxiv_api_latest_cs_lg'
```

The API key is read from the environment and must not be committed.

#### Output

The pipeline stores its results in `${result_dir}` (by default,
`hydra/YYYY-MM-DD/HH-MM-SS/`) as JSONL files:

1. `step_1.jsonl`: papers collected from arXiv.
2. `step_2.jsonl`: papers accepted by the fast LLM.
3. `step_2_debug.jsonl`: the fast LLM response and acceptance decision for each
   processed paper.
4. `step_3.jsonl`: papers accepted by the slow LLM.
5. `step_3_debug.jsonl`: the slow LLM review and acceptance decision for each
   paper from the shortlist.

If no papers are accepted during step 2, both `step_2.jsonl` and
`step_3.jsonl` will be empty. Use `step_2_debug.jsonl` to inspect the fast LLM
decisions.
