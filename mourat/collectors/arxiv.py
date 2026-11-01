"""arXiv paper collector: Function adapter over comourat."""

from __future__ import annotations

import logging
from typing import Any

from comourat.collectors.arxiv import ArxivPaperCollectorCore, ArxivSearchMode
import feedparser

from comourat.models import PaperCandidateCollection
from mourat.base import Function
from mourat.data_models import PaperCandidate
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)


class ArxivPaperCollector(Function[Any, PaperCandidateCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client,
        api_url: str,
        mode: ArxivSearchMode,
        start_date: str | None = None,  # YYYY-MM-DD
        end_date: str | None = None,  # YYYY-MM-DD
        max_results: int | None = None,
        keywords: str | None = None,
        only_new_submissions: bool = True,
    ) -> None:
        self.core = ArxivPaperCollectorCore(
            http_client=http_client,
            api_url=api_url,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            keywords=keywords,
            candidate_model=PaperCandidate,
            feed_parser=feedparser.parse,
        )
        self.only_new_submissions = only_new_submissions
        self.api_url = api_url
        self.keywords = keywords
        self.mode = mode
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        output = self.core.collect()
        collected_count = len(output.papers)
        excluded_count = 0

        if self.mode == "newest" and self.only_new_submissions:
            output, kept, excluded_count = self.core.filter_new_submissions(output)
            logger.info(
                "arXiv newest feed: collected %d papers, kept %d, excluded %d by new-only filter",
                collected_count,
                kept,
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
