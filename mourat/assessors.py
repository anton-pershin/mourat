import pydantic_ai
from pydantic_ai import Agent

from mourat.base import Function
from mourat.data_models import CandidateTopicAssessment, CandidateTopicInfo
from mourat.monitoring import MonitoringHandler


class CandidateTopicAssessor(Function[CandidateTopicInfo, CandidateTopicAssessment]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        system_prompt: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=CandidateTopicAssessment,
            system_prompt=system_prompt,
        )
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(
        self,
        data: CandidateTopicInfo,
    ) -> tuple[CandidateTopicAssessment, str]:
        result = self.agent.run_sync(
            self.user_prompt_template.format(
                candidate_topic_name=data.candidate_topic_name,
                candidate_topic_description=data.candidate_topic_description,
                business_prodict="\n\n".join(
                    [
                        "#### Product '{name}'\n"
                        + bp_info.model_dump_json(indent=2, ensure_ascii=False)
                        for name, bp_info in data.business_product.items()
                    ]
                ),
            )
        )

        text_for_monitoring = self.text_for_monitoring_template.format(
            candidate_topic_name=data.candidate_topic_name,
            relevance_info="\n\n".join(
                [
                    a.model_dump_json(indent=2, ensure_ascii=False)
                    for a in result.output.assessment
                ]
            ),
        )

        return result.output, text_for_monitoring
