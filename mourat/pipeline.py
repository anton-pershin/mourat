import json
import math
from abc import ABC, abstractmethod
from datetime import datetime
from textwrap import dedent
from typing import Any, Generic, Optional, TypeVar

import feedparser
import pydantic_ai
import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich.progress import track

from mourat.monitoring import MonitoringHandler

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Function(Generic[InputT, OutputT], ABC):
    def __init__(self, monitoring_handler: MonitoringHandler) -> None:
        self.monitoring_handler = monitoring_handler

    def __call__(self, data: InputT, step_id: str) -> OutputT:
        output, text_for_monitoring = self._run(data)
        self.monitoring_handler(step=step_id, text_for_monitoring=text_for_monitoring)
        return output

    @abstractmethod
    def _run(self, data: InputT) -> tuple[OutputT, str]:
        raise NotImplementedError()


class PaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="Arxiv URL of the paper")
    abstract: str = Field(description="Abstract of the paper")


class ScoredPaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="Arxiv URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


class PaperInfoCollection(BaseModel):
    papers: list[PaperInfo]


class ScoredPaperInfoCollection(BaseModel):
    papers: list[ScoredPaperInfo]


class ArxivPaperCollector(Function[Any, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        api_url: str,
        start_date: str,  # YYYY-MM-DD
        end_date: Optional[str],  # YYYY-MM-DD
        max_results: int,
        keywords: str,
    ) -> None:
        self.api_url = api_url
        self.start_date: datetime = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is not None:
            self.end_date: datetime = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = datetime.now()
        self.max_results = max_results
        self.keywords = keywords
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperInfoCollection, str]:
        """Input: none
        Output: papers which stores title, link and abstract, e.g.
        data["papers"][i]["title"]
        """
        output = PaperInfoCollection(papers=[])
        already_processed_titles = set()

        search_query = ""
        if self.keywords is not None:
            search_query += "all:"
            search_query += self.keywords.replace(" ", "+")
            search_query += "+AND+"

        date_range = "["
        date_range += self.start_date.strftime("%Y%m%d") + "0000"
        date_range += "+TO+"
        date_range += self.end_date.strftime("%Y%m%d") + "0000"
        date_range += "]"

        search_query += f"submittedDate:{date_range}"

        r = requests.get(
            f"{self.api_url}?"
            f"search_query={search_query}&"
            f"max_results={self.max_results}",
            timeout=5,
        )

        feed = feedparser.parse(r.text)
        for entry in feed.entries:
            link = entry.link
            title = entry.title.replace("\n", " ")
            if title not in already_processed_titles:
                abstract = entry.description.replace("\n", " ")

                output.papers.append(
                    PaperInfo(
                        title=title,
                        link=link,
                        abstract=abstract,
                    )
                )

                already_processed_titles.add(title)

        text_for_monitoring = (
            "# Keywords\n"
            f"{self.keywords}\n\n"
            "# Papers\n"
            f"Total {len(output.papers)}"
        )

        return output, text_for_monitoring


class BinaryPaperClassifier(Function[PaperInfoCollection, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        paper_topic: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(model, output_type=bool, system_prompt=system_prompt)
        self.paper_topic = paper_topic
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(self, data: PaperInfoCollection) -> tuple[PaperInfoCollection, str]:
        text_for_monitoring = ""
        output = data

        for p in track(output.papers[:], description="Classify papers"):
            result = self.agent.run_sync(
                self.user_prompt_template.format(
                    title=p.title,
                    abstract=p.abstract,
                    paper_topic=self.paper_topic,
                )
            )

            relevant = result.output

            if relevant:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p,
                )
            else:
                output.papers.remove(p)

        return output, text_for_monitoring


class PaperScoredByAgent(BaseModel):
    title: str = Field(description="Title of the paper")
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


class PaperScorer(Function[PaperInfoCollection, ScoredPaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        problem_being_addressed: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(
            model, output_type=list[PaperScoredByAgent], system_prompt=system_prompt
        )
        self.problem_being_addressed = problem_being_addressed
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(self, data: PaperInfoCollection) -> tuple[ScoredPaperInfoCollection, str]:
        output = ScoredPaperInfoCollection(papers=[])
        n_papers_in_group = 10
        n_groups = math.ceil(len(data.papers) // n_papers_in_group) + 1
        for g_i in track(range(n_groups), description="Score papers"):
            p_i_start = g_i * n_papers_in_group
            p_i_end = min((g_i + 1) * n_papers_in_group, len(data.papers))
            papers = data.papers[p_i_start:p_i_end]

            result = self.agent.run_sync(
                self.user_prompt_template.format(
                    problem_being_addressed=self.problem_being_addressed,
                    papers_as_json=",\n".join(
                        [
                            json.dumps(p.dict(), indent=2, ensure_ascii=False)
                            for p in papers
                        ]
                    ),
                )
            )

            text_for_monitoring = ""
            for paper_score_info in result.output:
                p: PaperInfo | None = self._find_paper_by_title(
                    data.papers, paper_score_info.title
                )
                if p is None:
                    raise ValueError(
                        f"Paper with title '{paper_score_info.title}' "
                        "not found in output"
                    )

                p_scored = ScoredPaperInfo(
                    title=p.title,
                    link=p.link,
                    abstract=p.abstract,
                    score=paper_score_info.score,
                    justification=paper_score_info.justification,
                )

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


class ScoreBasedPaperFilter(
    Function[ScoredPaperInfoCollection, ScoredPaperInfoCollection]
):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        score_threshold: int,
        text_for_monitoring_template: str,
    ) -> None:
        self.score_threshold = score_threshold
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(
        self, data: ScoredPaperInfoCollection
    ) -> tuple[ScoredPaperInfoCollection, str]:
        output = data
        text_for_monitoring = ""

        for p in output.papers[:]:
            if p.score < self.score_threshold:
                output.papers.remove(p)
            else:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p,
                )

        return output, text_for_monitoring


# class ProposalGenerator(Function):
#    def __init__(self) -> None:
#        pass
#
#    def _run(self, data: FunctionData) -> FunctionOutput:
#
#            responses = request_based_on_prompts(
#                llm_server_url=self.llm.url,
#                max_concurrent_requests=self.llm.max_concurrent_requests,
#                system_prompt=self.system_prompt,
#                user_prompts=[self.user_prompt_template.format(
#                    papers_as_json=json.dumps(papers, indent=2),
#                )],
#                authorization=self.llm.authorization,
#                model=self.llm.model,
#                progress_title=self.progress_title,
#            )
#
#        pass


def to_text_description(template: str, paper_info: PaperInfo | ScoredPaperInfo) -> str:
    return dedent(template).format(**(paper_info.dict())) + "\n"
