import hydra
from omegaconf import DictConfig
from pydantic_ai.models.openai import OpenAIChatModel
import httpx

from mourat.monitoring import MonitoringHandler
from mourat.pipeline import CandidateTopicAssessor, CandidateTopicInfo, CandidateTopicAssessment, BusinessProductInfo
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_assess_candidate_topic"


def assess_candidate_topic(cfg: DictConfig) -> None:
    slow_llm: OpenAiChatModel = hydra.utils.instantiate(cfg.slow_llm)

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(cfg.monitoring_handler)

    assessor: CandidateTopicAssessor = hydra.utils.instantiate(cfg.candidate_topic_assessor)(monitoring_handler, slow_llm)

    input_data = CandidateTopicInfo(
        candidate_topic_name=cfg.candidate_topic_name,
        candidate_topic_description=cfg.candidate_topic_description,
        business_product={p: BusinessProductInfo(**kwargs) for p, kwargs in cfg.business_products.items()},
    )

    # Step 1: assess relevance
    step_id = "1"
    assessment: CandidateTopicAssessment = assessor(input_data, step_id=step_id)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(assess_candidate_topic)()
