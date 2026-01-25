import json
import time
import math
from abc import ABC, abstractmethod
import datetime
from textwrap import dedent
from typing import Any, Generic, Optional, TypeVar, Literal, TypeAlias

import feedparser
import pydantic_ai
import requests
from pydantic import BaseModel, Field, NonNegativeInt
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.progress import track
import httpx

from mourat.monitoring import MonitoringHandler
from mourat.utils.common import normalize_author_name

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
    citation_count: NonNegativeInt | None = Field(description="Number of citations", default=None)
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(description="Date of publication", default=None)


class ScoredPaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="Arxiv URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    citation_count: NonNegativeInt | None = Field(description="Number of citations", default=None)
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(description="Date of publication", default=None)
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


class PaperInfoCollection(BaseModel):
    papers: list[PaperInfo]


class ScoredPaperInfoCollection(BaseModel):
    papers: list[ScoredPaperInfo]


ArxivSearchMode: TypeAlias = Literal["newest", "most_relevant"]


class ArxivPaperCollector(Function[Any, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        api_url: str,
        mode: ArxivSearchMode,
        start_date: str | None = None,  # YYYY-MM-DD
        end_date: str | None = None,  # YYYY-MM-DD
        max_results: int | None = None,
        keywords: str | None = None,
    ) -> None:
        self.http_client = http_client
        self.api_url = api_url
        if start_date is None:
            if mode == "most_relevant":
                raise ValueError(f"Mode '{mode}' implies that start_date is set")

        if end_date is None:
            self.end_date: datetime.date = datetime.date.today()
        else:
            self.end_date: datetime.date = datetime.date.fromisoformat(end_date)

        self.max_results = max_results
        self.keywords = keywords
        self.mode = mode
        self.mode_to_handler = {
            "newest": self._handle_newest_mode,
            "most_relevant": self._handle_most_relevant_mode,
        }

        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperInfoCollection, str]:
        output = self.mode_to_handler[self.mode]()
        text_for_monitoring = (
            "# URL;\n"
            f"{self.api_url}\n\n"
            "# Keywords\n"
            f"{self.keywords}\n\n"
            "# Papers\n"
            f"Total {len(output.papers)}"
        )

        return output, text_for_monitoring

    def _handle_newest_mode(self) -> PaperInfoCollection:
        output = PaperInfoCollection(papers=[])

        r: httpx.Response = self.http_client.get(self.api_url)

        feed = feedparser.parse(r.text)
        for entry in feed.entries:
            link = entry.link
            title = entry.title.replace("\n", " ")
            abstract = entry.description.split("\n")[1][10:]
            authors = [normalize_author_name(author_data["name"]) for author_data in entry.authors]
            publication_date = datetime.date(
                year=entry.published_parse.tm_year,
                month=entry.published_parse.tm_mon,
                day=entry.published_parse.tm_mday,
            )

            output.papers.append(
                PaperInfo(
                    title=title,
                    link=link,
                    abstract=abstract,
                    authors=authors,
                    publication_date=publication_date,
                )
            )

        return output

    def _handle_most_relevant_mode(self) -> PaperInfoCollection:
        output = PaperInfoCollection(papers=[])
        alread_processed_titles = set()

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

        serach_query += f"submittedDate:{date_range}"

        request_data = {
            "search_query": search_query,
            "max_results": self.max_results,
        }

        r: httpx.Response = self.http_client.get(
            self.api_url,
            params=request_data,
        )

        feed = feedparser.parse(r.text)
        for entry in feed.entries:
            link = entry.link
            title = entry.title.replace("\n", " ")
            abstract = entry.description.split("\n")[1][10:]
            authors = [normalize_author_name(author_data["name"]) for author_data in entry.authors]
            publication_date = datetime.date(
                year=entry.published_parse.tm_year,
                month=entry.published_parse.tm_mon,
                day=entry.published_parse.tm_mday,
            )

            output.papers.append(
                PaperInfo(
                    title=title,
                    link=link,
                    abstract=abstract,
                    authors=authors,
                    publication_date=publication_date,
                )
            )

        return output


SemanticScholarSearchMode: TypeAlias = Literal["newest", "most_relevant", "most_influential"]


class SemanticScholarPaperCollector(Function[Any, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        api_url: str,
        mode: SemanticScholarSearchMode,
        start_date: str | None,  # YYYY-MM-DD
        end_date: str | None,  # YYYY-MM-DD
        max_results: int,
        strict_keyword_query: str,
    ) -> None:
        self.http_client = http_client
        self.api_url = api_url

        self.start_date: datetime.date | None = None
        if start_date is not None:
            self.start_date = datetime.date.fromisoformat(start_date)

        self.end_date: datetime.date | None = None
        if end_date is not None:
            self.end_date = datetime.date.fromisoformat(end_date)

        self.max_results = max_results
        self.strict_keyword_query = strict_keyword_query
        self.mode = mode
        self.mode_to_handler = {
            "newest": self._handle_newest_mode,
            "most_relevant": self._handle_most_relevant_mode,
            "most_influential": self._handle_most_influential_mode,
        }

        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperInfoCollection, str]:
        output: PaperInfoCollection = self.mode_to_handler[self.mode]()

        text_for_monitoring = (
            "# Query\n"
            f"{self.strict_keyword_query}\n\n"
            "# Papers\n"
            f"Total {len(output.papers)}"
        )

        return output, text_for_monitoring

    def _handle_newest_mode(self) -> PaperInfoCollection:
        request_data = {
            "query": self.strict_keyword_query,
            "sort": "publicationDate:desc",
        }

        return self._get_papers_via_bulk_search(request_data)

    def _handle_most_relevant_mode(self) -> PaperInfoCollection:
        output = PaperInfoCollection(papers=[])

        n_papers_processed = 0
        request_data = {
            "query": self.strict_keyword_query,
        }
        next_offset: int | None = None

        while n_papers_processed < self.max_results:
            if next_offset is not None:
                request_data["offset"] = next_offset

            papers, r_data = self._run_one_paper_request_via_api(
                full_api_url=self.api_url + "/paper/search",
                request_data=request_data,
                n_papers_processed=n_papers_processed,
            )
            output.papers.extend(papers)
            n_papers_processed = len(output.papers)

            if r_data["next"] is None:  # present only if we can fetch more results
                break

            next_offset = r_data["next"]

        return output

    def _handle_most_influential_mode(self) -> PaperInfoCollection:
        request_data = {
            "query": self.strict_keyword_query,
            "sort": "citationCount:desc",
        }

        return self._get_papers_via_bulk_search(request_data)

    def _get_papers_via_bulk_search(
        self,
        request_data: dict[str, Any],
    ) -> PaperInfoCollection:
        output = PaperInfoCollection(papers=[])

        # Add standard fields to the request params
        request_data["fields"] = "title,url,abstract,citationCount,publicationDate,authors"

        # Add publication date range to the request params
        publication_date_range = self.start_date.isoformat() if self.start_date is not None else ""
        publication_date_range += ":"
        publication_date_range += self.end_date.isoformat() if self.end_date is not None else ""
        if publication_date_range != ":":
            request_data["publicationDateOrYear"] = publication_date_range

        n_papers_processed = 0
        token: str | None = None

        while n_papers_processed < self.max_results:
            if token is not None:
                request_data["token"] = token

            papers, r_data = self._run_one_paper_request_via_api(
                full_api_url=self.api_url + "/paper/search/bulk",
                request_data=request_data,
                n_papers_processed=n_papers_processed,
            )
            output.papers.extend(papers)
            n_papers_processed = len(output.papers)

            if r_data["token"] is None:  # token is present only if we can fetch more results
                break

            token = r_data["token"]

        return output

    def _run_one_paper_request_via_api(
        self,
        full_api_url: str,
        request_data: dict[str, Any],
        n_papers_processed: int,
    ) -> tuple[list[PaperInfo], dict[str, Any]]:
        received = False
        while not received:
            r: httpx.Response = self.http_client.get(
                full_api_url,
                params=request_data,
            )
            r_data = json.loads(r.text)
            if "code" not in r_data:
                received = True
            else:
                error_code = int(r_data["code"])
                sleep_s = 5
                print(f"Got error code {error_code}: '{r_data['message']}'. Will retry in {sleep_s} seconds")
                time.sleep(sleep_s)

        papers: list[PaperInfo] = []

        for ss_paper_data in r_data["data"]:
            if self._mandatory_fields_absent(ss_paper_data):
                continue

            publication_date: datetime.date | None = None
            if ss_paper_data["publicationDate"] is not None:
                publication_date = datetime.date.fromisoformat(ss_paper_data["publicationDate"])

            papers.append(
                PaperInfo(
                    title=ss_paper_data["title"],
                    link=ss_paper_data["url"],
                    abstract=ss_paper_data["abstract"],
                    citation_count=ss_paper_data["citationCount"],
                    authors=[author_data["name"] for author_data in ss_paper_data["authors"]],
                    publication_date=publication_date,
                )
            )
            n_papers_processed += 1
            if n_papers_processed >= self.max_results:
                break

        return papers, r_data

    @staticmethod
    def _mandatory_fields_absent(ss_paper_data: dict[str, Any]) -> bool:
        mandatory_fields = ["title", "url", "abstract"]
        return any(ss_paper_data[f] is None for f in mandatory_fields)


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
            except UnexpectedModelBehavior as e:
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


class AssignedPaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    citation_count: NonNegativeInt | None = Field(description="Number of citations", default=None)
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(description="Date of publication", default=None)
    assigned_topics: list[str]


class AssignedPaperInfoCollection(BaseModel):
    papers: list[AssignedPaperInfo]


class ListOfTopics(BaseModel):
    topics: list[str] = Field(description="List of topics")


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

    def _run(self, data: PaperInfoCollection) -> tuple[AssignedPaperInfoCollection, str]:
        text_for_monitoring = ""
        output = AssignedPaperInfoCollection(papers=[])

        for p in data.papers:
            with capture_run_messages() as messages:
                try:
                    result = self.agent.run_sync(
                        self.user_prompt_template.format(
                            topics=self.topics,
                            paper=json.dumps({"title": p.title, "abstract": p.abstract}, indent=2, ensure_ascii=False),
                        )
                    )
                except UnexpectedModelBehavior as e:
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


class PaperScoredByAgent(BaseModel):
    title: str = Field(description="Title of the paper")
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


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
                                p.model_dump_json(exclude_none=True, indent=2, ensure_ascii=False)
                                for p in papers
                            ]
                        ),
                    )
                )
            except UnexpectedModelBehavior as e:
                print(f"Failed to validate model answer. Skip scoring for {p_i_end - p_i_start} papers")
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


class QueryInfo(BaseModel):
    general_queries: list[str]
    specific_queries: list[str]


class QueryGeneratorViaLlm(
    Function[Any, QueryInfo]
):
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
        self.agent = Agent(
            model, output_type=QueryInfo, system_prompt=system_prompt
        )
        self.topic_name = topic_name
        self.topic_description = topic_description
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(
        self,
        data: Any
    ) -> tuple[QueryInfo, str]:
        result = self.agent.run_sync(
            self.user_prompt_template.format(
                topic_name=self.topic_name,
                topic_description=self.topic_description
            )
        )

        text_for_monitoring = self.text_for_monitoring_template.format(
            topic_name=self.topic_name,
            topic_description=self.topic_description,
            general_queries="\n".join(result.output.general_queries),
            specific_queries="\n".join(result.output.specific_queries),
        )

        return result.output, text_for_monitoring


class BusinessProductInfo(BaseModel):
    product_description: str = Field(description="Product description")
    challenges: list[str] = Field(description="A list of techinical challenges faced by the product")


class CandidateTopicInfo(BaseModel):
    candidate_topic_name: str = Field(description="Candidate topic name")
    candidate_topic_description: str = Field(description="Detailed description of the candidate topic")
    business_product: dict[str, BusinessProductInfo] = Field(description="A dictionary of business product where the key is the business product and the value is its description")


class CandidateTopicRelevanceInfo(BaseModel):
    product_name: str = Field(description="Prodict name")
    relevant_challenges: list[str] = Field(description="A list of relevant business product challenges")
    relevance_justification: str = Field(description="Justification of why the candidate topic may be relevant to the listed challenges")
    relevance_score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)


class CandidateTopicAssessment(BaseModel):
    assessment: list[CandidateTopicRelevanceInfo]


class CandidateTopicAssessor(
    Function[CandidateTopicInfo, CandidateTopicAssessment]
):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        system_prompt: str,
        user_prompt_template: str,
        text_for_monitoring: str,
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
                business_prodict="\n\n".join(["#### Product '{name}'\n" + bp_info.model_dump_json(indent=2, ensure_ascii=False) for name, bp_info in data.business_products.items()]),
            )
        )

        text_for_monitoring = self.text_for_monitoring_template.format(
            candidate_topic_name=data.candidate_topic_name,
            relevance_info="\n\n".join([a.model_dump_json(indent=2, ensure_ascii=False) for a in result.output.assessment])
        )

        return result.output, text_for_monitoring


def to_text_description(template: str, paper_info: PaperInfo | ScoredPaperInfo) -> str:
    return dedent(template).format(**(paper_info.dict())) + "\n"
