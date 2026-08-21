import json
from typing import Optional

import pydantic_ai
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior

from mourat.base import Function
from mourat.data_models import (
    AssignedPaperInfo,
    AssignedPaperInfoCollection,
    PaperInfoCollection,
)
from mourat.monitoring import MonitoringHandler


class PaperAssigner(Function[PaperInfoCollection, AssignedPaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        topics: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(model, output_type=str, system_prompt=system_prompt)
        self.topics = topics
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperInfoCollection
    ) -> tuple[AssignedPaperInfoCollection, str]:
        text_for_monitoring = ""
        output = AssignedPaperInfoCollection(papers=[])

        for p in data.papers:
            with capture_run_messages() as _:
                try:
                    result = self.agent.run_sync(
                        self.user_prompt_template.format(
                            topics=self.topics,
                            paper=json.dumps(
                                {"title": p.title, "abstract": p.abstract},
                                indent=2,
                                ensure_ascii=False,
                            ),
                        )
                    )
                except UnexpectedModelBehavior:
                    print(f"Failed to validate model answer. Remove paper '{p.title}'")
                    continue

            relevant_topic = result.output.strip(" \n\t\r*.#").lower()
            if relevant_topic != "none":
                ap = AssignedPaperInfo(
                    title=p.title,
                    link=p.link,
                    abstract=p.abstract,
                    citation_count=p.citation_count,
                    authors=p.authors,
                    publication_date=p.publication_date,
                    assigned_topics=[relevant_topic],
                )
                output.papers.append(ap)
                text_for_monitoring += self.text_for_monitoring_template.format(
                    title=p.title,
                    link=p.link,
                    abstract=p.abstract,
                    assigned_topics=relevant_topic,
                )

        return output, text_for_monitoring
