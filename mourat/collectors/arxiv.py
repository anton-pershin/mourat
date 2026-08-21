import datetime
from typing import Any, Literal, TypeAlias

import feedparser
import httpx

from mourat.base import Function
from mourat.data_models import PaperInfo, PaperInfoCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import normalize_author_name

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
        else:
            self.start_date: datetime.date = datetime.date.fromisoformat(start_date)

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
            authors = [
                normalize_author_name(author_data["name"])
                for author_data in entry.authors
            ]
            publication_date = datetime.date(
                year=entry.published_parsed.tm_year,
                month=entry.published_parsed.tm_mon,
                day=entry.published_parsed.tm_mday,
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
            authors = [
                normalize_author_name(author_data["name"])
                for author_data in entry.authors
            ]
            publication_date = datetime.date(
                year=entry.published_parsed.tm_year,
                month=entry.published_parsed.tm_mon,
                day=entry.published_parsed.tm_mday,
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
