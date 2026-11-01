"""arXiv HTML affiliation fetcher: Function adapter over comourat."""

from __future__ import annotations

import httpx

from comourat.collectors.arxiv_html_affiliations import (
    ArxivHtmlAffiliationFetcherCore,
    _extract_affiliations,
)
from comourat.models import PaperCandidateCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler

__all__ = ["ArxivHtmlAffiliationFetcher", "_extract_affiliations"]


class ArxivHtmlAffiliationFetcher(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Attaches author affiliations and true publication dates.

    Pass-through by count: every candidate is returned. Two in-flight
    fields change: `affiliations` (None when unknown) and
    `publication_date`, set from the arXiv Atom API's `published` field —
    the FIRST submission date, not the last revision. On Atom API failure
    the date stays None rather than a wrong value. Fetching is paced:
    `request_delay_seconds` between requests.
    """

    _ATOM_API_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.core = ArxivHtmlAffiliationFetcherCore(
            http_client=http_client,
            request_delay_seconds=request_delay_seconds,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        output, lines = self.core.fetch_affiliations(data)
        return output, "\n".join(lines)
