import datetime
import logging
import re
from typing import Any, Literal, TypeAlias

import feedparser
import httpx

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import normalize_author_name

ArxivSearchMode: TypeAlias = Literal["newest", "most_relevant"]

_ARXIV_ABS_ID_RE = re.compile(r"/abs/([\d.]+)(?:v\d+)?$")
_ANNOUNCE_TYPE_RE = re.compile(r"Announce Type: (\w+)")
logger = logging.getLogger(__name__)


def _extract_arxiv_id(link: str) -> str | None:
    """Base arXiv id from an abs link; version suffix stripped (R1)."""
    m = _ARXIV_ABS_ID_RE.search(link)
    return m.group(1) if m else None


def _candidate_from_entry(entry) -> PaperCandidate:
    """One RSS item -> PaperCandidate.

    The feed items carry no per-item date (feedparser's published_parsed is
    the feed-generation date), so publication_date is left None here; the
    affiliation fetcher sets it from the arXiv Atom API's `published`
    field (first submission date). `announce_type` is read from the structured RSS
    field and kept in-flight for monitoring ("new", "replace", "cross", or
    "replace-cross").
    """
    link = entry.link
    title = entry.title.replace("\n", " ")
    abstract = entry.description.split("\n")[1][10:]
    authors = [
        normalize_author_name(author_data["name"]) for author_data in entry.authors
    ]
    announce_type = getattr(entry, "arxiv_announce_type", None)
    if announce_type is None:
        m = _ANNOUNCE_TYPE_RE.search(entry.description)
        announce_type = m.group(1).lower() if m else None

    return PaperCandidate(
        title=title,
        authors=authors,
        description=abstract,
        urls_seen=[link],
        arxiv_id=_extract_arxiv_id(link),
        publication_date=None,
        provenance=["arxiv_rss"],
        announce_type=announce_type,
    )


class ArxivPaperCollector(Function[Any, PaperCandidateCollection]):
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
        only_new_submissions: bool = True,
    ) -> None:
        self.http_client = http_client
        self.api_url = api_url
        self.only_new_submissions = only_new_submissions
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

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        output = self.mode_to_handler[self.mode]()
        collected_count = len(output.papers)
        excluded_count = 0

        if self.mode == "newest" and self.only_new_submissions:
            output.papers = [
                paper for paper in output.papers if paper.announce_type == "new"
            ]
            excluded_count = collected_count - len(output.papers)
            logger.info(
                "arXiv newest feed: collected %d papers, kept %d, excluded %d by new-only filter",
                collected_count,
                len(output.papers),
                excluded_count,
            )

        text_for_monitoring = (
            "# URL;\n"
            f"{self.api_url}\n\n"
            "# Keywords\n"
            f"{self.keywords}\n\n"
            "# Papers\n"
            f"Total {len(output.papers)}"
        )
        if self.mode == "newest" and self.only_new_submissions:
            text_for_monitoring += (
                "\n\n# New-submission filter\n"
                f"Collected {collected_count}; kept {len(output.papers)}; "
                f"excluded {excluded_count}"
            )

        return output, text_for_monitoring

    def _handle_newest_mode(self) -> PaperCandidateCollection:
        r: httpx.Response = self.http_client.get(self.api_url)
        feed = feedparser.parse(r.text)
        output = PaperCandidateCollection(
            papers=[_candidate_from_entry(entry) for entry in feed.entries]
        )
        return output

    def _handle_most_relevant_mode(self) -> PaperCandidateCollection:
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
        output = PaperCandidateCollection(
            papers=[_candidate_from_entry(entry) for entry in feed.entries]
        )
        return output
