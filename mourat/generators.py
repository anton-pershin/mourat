from typing import Any

import pydantic_ai
from pydantic_ai import Agent

from mourat.base import Function
from mourat.data_models import QueryInfo
from mourat.monitoring import MonitoringHandler


class QueryGeneratorViaLlm(Function[Any, QueryInfo]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        topic_name: str,
        topic_description: str,
        system_prompt: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
    ) -> None:
        self.agent = Agent(model, output_type=QueryInfo, system_prompt=system_prompt)
        self.topic_name = topic_name
        self.topic_description = topic_description
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[QueryInfo, str]:
        result = self.agent.run_sync(
            self.user_prompt_template.format(
                topic_name=self.topic_name, topic_description=self.topic_description
            )
        )

        text_for_monitoring = self.text_for_monitoring_template.format(
            topic_name=self.topic_name,
            topic_description=self.topic_description,
            general_queries="\n".join(result.output.general_queries),
            specific_queries="\n".join(result.output.specific_queries),
        )

        return result.output, text_for_monitoring
