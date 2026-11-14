"""Author affiliation fetcher with arXiv and Semantic Scholar sources."""

from __future__ import annotations

import logging

import httpx
from comourat.clients.semantic_scholar import SemanticScholarClient
from comourat.collectors.arxiv_html_affiliations import (
    ArxivHtmlAffiliationFetcherCore,
    _extract_affiliations,
)
from comourat.models import (
    AuthorRecord,
    PaperCandidate,
    PaperCandidateCollection,
)

from mourat.base import Function
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

__all__ = ["ArxivHtmlAffiliationFetcher", "_extract_affiliations"]

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"


class ArxivHtmlAffiliationFetcher(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Attach author affiliations and arXiv first-publication dates.

    Candidates with an arXiv ID use the existing arXiv HTML and Atom API path.
    DOI-only Semantic Scholar candidates use their retained author IDs with the
    Semantic Scholar author-batch endpoint. Other candidates pass through with
    unknown affiliations. Every candidate is returned.
    """

    _ATOM_API_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        request_delay_seconds: float = 1.0,
        semantic_scholar_client: SemanticScholarClient | None = None,
        semantic_scholar_api_url: str = SEMANTIC_SCHOLAR_BASE_URL,
        semantic_scholar_api_key: str | None = None,
        semantic_scholar_proxy: str | None = None,
        semantic_scholar_timeout_seconds: float = 30.0,
        semantic_scholar_max_retries: int = 4,
        semantic_scholar_backoff_seconds: float = 2.0,
        semantic_scholar_regular_delay_seconds: float = 4.0,
        semantic_scholar_jitter_seconds: float = 1.0,
    ) -> None:
        self.core = ArxivHtmlAffiliationFetcherCore(
            http_client=http_client,
            request_delay_seconds=request_delay_seconds,
        )
        self.semantic_scholar_client = (
            semantic_scholar_client
            if semantic_scholar_client is not None
            else SemanticScholarClient(
                api_url=semantic_scholar_api_url.rstrip("/"),
                api_key=semantic_scholar_api_key,
                proxy=semantic_scholar_proxy,
                timeout_seconds=semantic_scholar_timeout_seconds,
                max_retries=semantic_scholar_max_retries,
                backoff_seconds=semantic_scholar_backoff_seconds,
                regular_delay_seconds=semantic_scholar_regular_delay_seconds,
                jitter_seconds=semantic_scholar_jitter_seconds,
            )
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        # The core mutates the candidates in place. Restrict its report to arXiv
        # candidates so DOI-only successes are not also reported as unknown.
        arxiv_candidates = PaperCandidateCollection(
            papers=[paper for paper in data.papers if paper.arxiv_id is not None]
        )
        _, arxiv_lines = self.core.fetch_affiliations(arxiv_candidates)
        semantic_scholar_lines = self._fetch_semantic_scholar_affiliations(data)
        return data, "\n".join([*arxiv_lines, *semantic_scholar_lines])

    def _fetch_semantic_scholar_affiliations(
        self, data: PaperCandidateCollection
    ) -> list[str]:
        candidates = [
            paper
            for paper in data.papers
            if paper.arxiv_id is None
            and paper.doi is not None
            and "semantic_scholar" in paper.provenance
        ]
        with_affiliations = 0
        without_ids: list[str] = []
        failures: list[str] = []

        for paper in candidates:
            author_ids = [
                author_id for author_id in paper.author_ids or [] if author_id
            ]
            if not author_ids:
                paper.affiliations = None
                without_ids.append(paper.title)
                continue

            try:
                records = self.semantic_scholar_client.get_authors(author_ids)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                paper.affiliations = None
                failures.append(paper.title)
                logger.warning(
                    "Semantic Scholar affiliation fetch failed for '%s' (%s)",
                    paper.title,
                    type(exc).__name__,
                )
                continue

            affiliations = self._affiliations_by_author(paper, records)
            if affiliations is None:
                paper.affiliations = None
            else:
                paper.affiliations = affiliations
                with_affiliations += 1

        unknown_count = len(candidates) - with_affiliations
        lines = [
            (
                f"Semantic Scholar DOI-only papers: {len(candidates)}, "
                f"with affiliations: {with_affiliations}, "
                f"affiliations unknown: {unknown_count}"
            ),
            f"Semantic Scholar affiliation failures: {len(failures)}",
        ]
        if without_ids:
            lines.append("Semantic Scholar author IDs unavailable for:")
            lines.extend(f"- {title}" for title in without_ids)
        if failures:
            lines.append("Semantic Scholar affiliation failures for:")
            lines.extend(f"- {title}" for title in failures)
        return lines

    @staticmethod
    def _affiliations_by_author(
        paper: PaperCandidate, records: list[AuthorRecord]
    ) -> dict[str, list[str]] | None:
        names_by_id = {
            author_id: name
            for name, author_id in zip(paper.authors, paper.author_ids or [])
            if name and author_id
        }
        known: dict[str, list[str]] = {}
        for record in records:
            institutions = [
                affiliation for affiliation in record.affiliations if affiliation
            ]
            if not institutions:
                continue
            name = names_by_id.get(record.author_id) or record.name
            if not name:
                continue
            retained = known.setdefault(name, [])
            for institution in institutions:
                if institution not in retained:
                    retained.append(institution)

        if not known:
            return None

        # Keep missing data explicit for each author without hiding known data.
        affiliations = {name: ["unknown"] for name in paper.authors if name}
        affiliations.update(known)
        return affiliations
