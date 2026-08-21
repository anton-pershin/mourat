import math
from typing import Optional

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.progress import track

from mourat.base import Function
from mourat.data_models import (
    PaperInfo,
    PaperInfoCollection,
    PaperScoredByAgent,
    ScoredPaperInfo,
    ScoredPaperInfoCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_text_description


class PaperScorer(Function[PaperInfoCollection, ScoredPaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        topic_name: str,
        topic_description: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(
            model, output_type=list[PaperScoredByAgent], system_prompt=system_prompt
        )
        self.topic_name = topic_name
        self.topic_description = topic_description
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(self, data: PaperInfoCollection) -> tuple[ScoredPaperInfoCollection, str]:
        output = ScoredPaperInfoCollection(papers=[])
        text_for_monitoring = ""
        n_papers_in_group = 10
        n_groups = math.ceil(len(data.papers) // n_papers_in_group) + 1
        for g_i in track(range(n_groups), description="Score papers"):
            p_i_start = g_i * n_papers_in_group
            p_i_end = min((g_i + 1) * n_papers_in_group, len(data.papers))
            papers = data.papers[p_i_start:p_i_end]

            try:
                result = self.agent.run_sync(
                    self.user_prompt_template.format(
                        topic_name=self.topic_name,
                        topic_description=self.topic_description,
                        papers_as_json=",\n".join(
                            [
                                p.model_dump_json(
                                    exclude_none=True, indent=2, ensure_ascii=False
                                )
                                for p in papers
                            ]
                        ),
                    )
                )
            except UnexpectedModelBehavior:
                print(
                    f"Failed to validate model answer. "
                    f"Skip scoring for {p_i_end - p_i_start} papers"
                )
                continue

            for paper_score_info in result.output:
                # TODO: this is an awful prompt design for checking the paper id
                p: PaperInfo | None = self._find_paper_by_title(
                    data.papers, paper_score_info.title
                )
                if p is None:
                    print(
                        f"Paper with title '{paper_score_info.title}' "
                        "not found in output. Have to skip it"
                    )
                    continue

                p_scored = ScoredPaperInfo(
                    title=p.title,
                    link=p.link,
                    abstract=p.abstract,
                    citation_count=p.citation_count,
                    authors=p.authors,
                    score=paper_score_info.score,
                    justification=paper_score_info.justification,
                )
                output.papers.append(p_scored)

                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p_scored,
                )

        return output, text_for_monitoring

    @staticmethod
    def _find_paper_by_title(papers: list[PaperInfo], title: str) -> PaperInfo | None:
        # TODO: absurdly slow method, should be replaced with dict-like structure
        for p in papers:
            if p.title.lower() == title.lower():
                return p

        return None
