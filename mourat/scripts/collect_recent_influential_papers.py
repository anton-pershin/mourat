import hydra
from omegaconf import DictConfig
from pydantic_ai.models.openai import OpenAIChatModel
import httpx

from mourat.monitoring import MonitoringHandler
from mourat.pipeline import SemanticScholarPaperCollector, BinaryPaperClassifier, PaperScorer, ScoreBasedPaperFilter, ScoredPaperInfoCollection, PaperInfoCollection
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_collect_recent_influential_papers"


def collect_recent_influential_papers(cfg: DictConfig) -> None:
    slow_llm: OpenAiChatModel = hydra.utils.instantiate(cfg.slow_llm)
    fast_llm: OpenAiChatModel = hydra.utils.instantiate(cfg.fast_llm)

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(cfg.monitoring_handler)

    http_client = httpx.Client(
        verify=False,
        timeout=httpx.Timeout(
            timeout=600,
            connect=5,
        ),
    )

    ss_paper_collector: SemanticScholarPaperCollector = hydra.utils.instantiate(cfg.collector)(monitoring_handler, http_client)
    binary_paper_classifier: BinaryPaperClassifier = hydra.utils.instantiate(cfg.binary_paper_classifier)(monitoring_handler, fast_llm)
    paper_scorer: PaperScorer = hydra.utils.instantiate(cfg.paper_scorer)(monitoring_handler, slow_llm)
    score_based_paper_filter: ScoreBasedPaperFilter = hydra.utils.instantiate(cfg.paper_filter)(monitoring_handler)

    # Step 1: collect papers from semantic scholar by keywords
    step_id = "1"
    paper_info: PaperInfoCollection = ss_paper_collector({}, step_id=step_id)

    # Step 2: drop obviously irrelevant papers
    step_id = "2"
    paper_info = binary_paper_classifier(paper_info, step_id=step_id)

    # Step 3: score the remaining papers with justifications
    step_id = "3"
    scored_paper_info: ScoredPaperInfoCollection = paper_scorer(paper_info, step_id=step_id)

    # Step 4: drop the papers with low scores
    step_id = "4"
    scored_paper_info = score_based_paper_filter(scored_paper_info, step_id=step_id)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_recent_influential_papers)()
