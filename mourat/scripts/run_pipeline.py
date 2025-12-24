import hydra
from omegaconf import DictConfig
from pydantic_ai.models.openai import OpenAIChatModel

from mourat.monitoring import MonitoringHandler
from mourat.pipeline import (ArxivPaperCollector, BinaryPaperClassifier,
                             PaperInfoCollection, PaperScorer, ScoreBasedPaperFilter,
                             ScoredPaperInfoCollection)
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_run_pipeline"


def run_pipeline(cfg: DictConfig) -> None:
    slow_llm: OpenAIChatModel = hydra.utils.instantiate(cfg.slow_llm)
    fast_llm: OpenAIChatModel = hydra.utils.instantiate(cfg.fast_llm)

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )
    arxiv_paper_collector: ArxivPaperCollector = hydra.utils.instantiate(
        cfg.functions.arxiv_paper_collector
    )(monitoring_handler)
    binary_paper_classifier: BinaryPaperClassifier = hydra.utils.instantiate(
        cfg.functions.binary_paper_classifier
    )(monitoring_handler, fast_llm)
    paper_scorer: PaperScorer = hydra.utils.instantiate(cfg.functions.paper_scorer)(
        monitoring_handler, slow_llm
    )
    score_based_paper_filter: ScoreBasedPaperFilter = hydra.utils.instantiate(
        cfg.functions.score_based_paper_filter
    )(monitoring_handler)

    # Step 1: collect papers from arxiv by keywords
    step_id = "1"
    paper_info: PaperInfoCollection = arxiv_paper_collector({}, step_id=step_id)

    # Step 2: drop obviously irrelevant papers
    step_id = "2"
    paper_info = binary_paper_classifier(paper_info, step_id=step_id)

    # Step 3: score the remaining papers with justifications
    step_id = "3"
    scored_paper_info: ScoredPaperInfoCollection = paper_scorer(
        paper_info, step_id=step_id
    )

    # Step 4: drop the papers with low scores
    step_id = "4"
    scored_paper_info = score_based_paper_filter(scored_paper_info, step_id=step_id)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_pipeline)()
