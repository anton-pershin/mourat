import hydra
from omegaconf import DictConfig
from pydantic_ai.models.openai import OpenAIChatModel
import httpx

from mourat.monitoring import MonitoringHandler
from mourat.pipeline import QueryGeneratorViaLlm
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_generate_queries"


def generate_queries(cfg: DictConfig) -> None:
    slow_llm: OpenAiChatModel = hydra.utils.instantiate(cfg.slow_llm)

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(cfg.monitoring_handler)

    query_generator: QueryGeneratorViaLlm = hydra.utils.instantiate(cfg.query_generator)(monitoring_handler, slow_llm)

    # Step 1: generate queries
    step_id = "1"
    query_info: QueryInfo = query_generator({}, step_id=step_id)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(generate_queries)()
