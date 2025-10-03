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
