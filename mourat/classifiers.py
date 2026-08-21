from typing import Optional

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.progress import track

from mourat.base import Function
from mourat.data_models import PaperInfoCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_text_description


class BinaryPaperClassifier(Function[PaperInfoCollection, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        topic_name: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(model, output_type=bool, system_prompt=system_prompt)
        self.topic_name = topic_name
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(self, data: PaperInfoCollection) -> tuple[PaperInfoCollection, str]:
        text_for_monitoring = ""
        output = data

        for p in track(output.papers[:], description="Classify papers"):
            try:
                result = self.agent.run_sync(
                    self.user_prompt_template.format(
                        title=p.title,
                        abstract=p.abstract,
                        topic_name=self.topic_name,
                    )
                )
            except UnexpectedModelBehavior:
                print(f"Failed to validate model answer. Remove paper '{p.title}'")
                output.papers.remove(p)
                continue

            relevant = result.output

            if relevant:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p,
                )
            else:
                output.papers.remove(p)

        return output, text_for_monitoring
