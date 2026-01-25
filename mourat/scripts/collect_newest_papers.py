import hydra
from omegaconf import DictConfig
from pydantic_ai.models.openai import OpenAIChatModel
import httpx

from mourat.monitoring import MonitoringHandler
from mourat.pipeline import ArxivPaperCollector, PaperAssigner, PaperInfoCollection
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_collect_newest_papers"


def collect_newest_papers(cfg: DictConfig) -> None:
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

    arxiv_paper_collector: ArxivPaperCollector = hydra.utils.instantiate(cfg.collector)(monitoring_handler, http_client)
    paper_assigner: PaperAssigner = hydra.utils.instantiate(cfg.paper_assigner)(monitoring_handler, fast_llm)

    # Step 1: collect the recent feed from arxiv
    step_id = "1"
    paper_info: PaperInfoCollection = arxiv_paper_collector({}, step_id=step_id)

    # Step 2: assign papers to topics
    step_id = "2"
    paper_info = paper_assigner(paper_info, step_id=step_id)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_newest_papers)()
