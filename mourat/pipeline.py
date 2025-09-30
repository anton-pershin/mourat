import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Any, Optional

import feedparser
import requests
from rally.interaction import request_based_on_prompts
from rally.llm import Llm
from rally.utils.common import to_boolean
from rich.progress import track

from mourat.monitoring import MonitoringHandler

FunctionData = dict[str, Any]


@dataclass
class FunctionOutput:
    data: FunctionData
    text_for_monitoring: str


class Function(ABC):
    def __init__(self, required_data_fields: list[str]) -> None:
        self.required_data_fields = required_data_fields

    def __call__(self, data: FunctionData) -> FunctionOutput:
        self._validate_data(data)
        return self._run(data)

    @abstractmethod
    def _run(self, data: FunctionData) -> FunctionOutput:
        raise NotImplementedError()

    def _validate_data(self, data: FunctionData) -> None:
        for field in self.required_data_fields:
            if field not in data:
                raise KeyError(f"Required field {field} is not in data")


class Pipeline:
    def __init__(
        self, functions: list[Function], monitoring_handler: MonitoringHandler
    ) -> None:
        self.functions = functions
        self.monitoring_handler = monitoring_handler

    def __call__(self, data: FunctionData) -> FunctionData:
        for i, function in enumerate(self.functions):
            function_output = function(data)
            data = function_output.data
            self.monitoring_handler(
                step=str(i), text_for_monitoring=function_output.text_for_monitoring
            )

        return data


class ArxivPaperCollector(Function):
    def __init__(
        self,
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
        super().__init__(required_data_fields=[])

    def _run(self, data: FunctionData) -> FunctionOutput:
        """Input: none
        Output: papers which stores title, link and abstract, e.g.
        data["papers"][i]["title"]
        """
        data["papers"] = []
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
            timeout=5
        )

        feed = feedparser.parse(r.text)
        for entry in feed.entries:
            link = entry.link
            title = entry.title.replace("\n", " ")
            if title not in already_processed_titles:
                abstract = entry.description.replace("\n", " ")

                data["papers"].append(
                    {
                        "title": title,
                        "link": link,
                        "abstract": abstract,
                    }
                )
                already_processed_titles.add(title)

        text_for_monitoring = (
            "# Keywords\n"
            f"{self.keywords}\n\n"
            "# Papers\n"
            f"Total {len(data['papers'])}"
        )

        return FunctionOutput(data=data, text_for_monitoring=text_for_monitoring)


class BinaryPaperClassifier(Function):
    def __init__(
        self,
        paper_topic: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        llm: Llm,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.paper_topic = paper_topic
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.llm = llm
        self.system_prompt = system_prompt
        self.progress_title = progress_title
        super().__init__(required_data_fields=["papers"])

    def _run(self, data: FunctionData) -> FunctionOutput:
        text_for_monitoring = ""

        for p in track(data["papers"][:], description="Classify papers"):

            # TODO: batch size = 1 for no reason
            responses = request_based_on_prompts(
                llm_server_url=self.llm.url,
                max_concurrent_requests=self.llm.max_concurrent_requests,
                system_prompt=self.system_prompt,
                user_prompts=[
                    self.user_prompt_template.format(
                        title=p["title"],
                        abstract=p["abstract"],
                        paper_topic=self.paper_topic,
                    )
                ],
                authorization=self.llm.authorization,
                model=self.llm.model,
                progress_title=self.progress_title,
            )

            relevant = to_boolean(responses[0])

            if relevant:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper=p,
                )
            else:
                data["papers"].remove(p)

        return FunctionOutput(data=data, text_for_monitoring=text_for_monitoring)


class PaperScorer(Function):
    def __init__(
        self,
        problem_being_addressed: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        llm: Llm,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.problem_being_addressed = problem_being_addressed
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.llm = llm
        self.system_prompt = system_prompt
        self.progress_title = progress_title
        super().__init__(required_data_fields=["papers"])

    def _run(self, data: FunctionData) -> FunctionOutput:
        text_for_monitoring = ""

        n_papers_in_group = 10
        n_groups = math.ceil(len(data["papers"]) // n_papers_in_group) + 1
        for g_i in track(range(n_groups), description="Score papers"):
            p_i_start = g_i * n_papers_in_group
            p_i_end = min((g_i + 1) * n_papers_in_group, len(data["papers"]))
            papers = data["papers"][p_i_start:p_i_end]

            # TODO: batch size = 1 for no reason
            responses = request_based_on_prompts(
                llm_server_url=self.llm.url,
                max_concurrent_requests=self.llm.max_concurrent_requests,
                system_prompt=self.system_prompt,
                user_prompts=[
                    self.user_prompt_template.format(
                        problem_being_addressed=self.problem_being_addressed,
                        papers_as_json=json.dumps(papers, indent=2),
                    )
                ],
                authorization=self.llm.authorization,
                model=self.llm.model,
                progress_title=self.progress_title,
            )

            # We expect the response to follow the json format
            response_as_json_dict = json.loads(responses[0])
            for p_i, p in enumerate(papers):
                p["score"] = int(response_as_json_dict[p_i]["score"])
                p["justification"] = response_as_json_dict[p_i]["justification"]

                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper=p,
                )

        return FunctionOutput(data=data, text_for_monitoring=text_for_monitoring)


class ScoreBasedPaperFilter(Function):
    def __init__(self, score_threshold: int, text_for_monitoring_template: str) -> None:
        self.score_threshold = score_threshold
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(required_data_fields=["papers"])

    def _run(self, data: FunctionData) -> FunctionOutput:
        text_for_monitoring = ""

        for p in data["papers"][:]:
            if p["score"] < self.score_threshold:
                data["papers"].remove(p)
            else:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper=p,
                )

        return FunctionOutput(data=data, text_for_monitoring=text_for_monitoring)


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


def to_text_description(template: str, paper: dict[str, Any]) -> str:
    return dedent(template).format(**paper) + "\n"
