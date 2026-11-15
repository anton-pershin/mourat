"""Author affiliation fetcher with arXiv, Semantic Scholar, and OpenAlex."""

from __future__ import annotations

import logging

import httpx
from comourat.clients.openalex import OpenAlexClient
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
DEFAULT_OPENALEX_USER_AGENT = "mourat/0.1 (mailto:contact@example.com)"


class ArxivHtmlAffiliationFetcher(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Attach author affiliations and arXiv first-publication dates.

    Candidates with an arXiv ID use the existing arXiv HTML and Atom API path.
    DOI-only Semantic Scholar candidates use their retained author IDs with the
    Semantic Scholar author-batch endpoint. OpenAlex supplies paper-level data
    when Semantic Scholar has no usable affiliations. Other candidates pass
    through with unknown affiliations. Every candidate is returned.
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
        openalex_client: OpenAlexClient | None = None,
        openalex_enabled: bool = True,
        openalex_user_agent: str = DEFAULT_OPENALEX_USER_AGENT,
        openalex_api_key: str | None = None,
        openalex_timeout_seconds: float = 30.0,
        openalex_max_retries: int = 4,
        openalex_backoff_seconds: float = 2.0,
        openalex_regular_delay_seconds: float = 2.0,
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
        self.openalex_enabled = openalex_enabled
        self.openalex_client = openalex_client
        if self.openalex_client is None and self.openalex_enabled:
            self.openalex_client = OpenAlexClient(
                user_agent=openalex_user_agent,
                api_key=openalex_api_key,
                timeout_seconds=openalex_timeout_seconds,
                max_retries=openalex_max_retries,
                backoff_seconds=openalex_backoff_seconds,
                regular_delay_seconds=openalex_regular_delay_seconds,
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
        doi_only_lines = self._fetch_doi_only_affiliations(data)
        return data, "\n".join([*arxiv_lines, *doi_only_lines])

    def _fetch_doi_only_affiliations(self, data: PaperCandidateCollection) -> list[str]:
        candidates = [
            paper
            for paper in data.papers
            if paper.arxiv_id is None
            and paper.doi is not None
            and "semantic_scholar" in paper.provenance
        ]
        semantic_scholar_with_affiliations = 0
        openalex_attempts = 0
        openalex_with_affiliations = 0
        without_ids: list[str] = []
        semantic_scholar_failures: list[str] = []
        openalex_unavailable: list[str] = []
        openalex_failures: list[str] = []

        for paper in candidates:
            affiliations = None
            author_ids = [
                author_id for author_id in paper.author_ids or [] if author_id
            ]
            if not author_ids:
                without_ids.append(paper.title)
            else:
                try:
                    records = self.semantic_scholar_client.get_authors(author_ids)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    semantic_scholar_failures.append(paper.title)
                    logger.warning(
                        "Semantic Scholar affiliation fetch failed for '%s' (%s)",
                        paper.title,
                        type(exc).__name__,
                    )
                else:
                    affiliations = self._affiliations_by_author(paper, records)

            if affiliations is not None:
                paper.affiliations = affiliations
                semantic_scholar_with_affiliations += 1
                continue

            paper.affiliations = None
            if not self.openalex_enabled:
                continue

            openalex_attempts += 1
            affiliations, failed = self._fetch_openalex_affiliations(paper)
            if affiliations is not None:
                paper.affiliations = affiliations
                openalex_with_affiliations += 1
            elif failed:
                openalex_failures.append(paper.title)
            else:
                openalex_unavailable.append(paper.title)

        return self._doi_only_monitoring_lines(
            len(candidates),
            semantic_scholar_with_affiliations,
            openalex_attempts,
            openalex_with_affiliations,
            without_ids,
            semantic_scholar_failures,
            openalex_unavailable,
            openalex_failures,
        )

    def _doi_only_monitoring_lines(
        self,
        candidate_count: int,
        semantic_scholar_with_affiliations: int,
        openalex_attempts: int,
        openalex_with_affiliations: int,
        without_ids: list[str],
        semantic_scholar_failures: list[str],
        openalex_unavailable: list[str],
        openalex_failures: list[str],
    ) -> list[str]:
        unknown_count = (
            candidate_count
            - semantic_scholar_with_affiliations
            - openalex_with_affiliations
        )
        lines = [
            (
                f"Semantic Scholar DOI-only papers: {candidate_count}, "
                f"with affiliations: {semantic_scholar_with_affiliations}, "
                "without usable affiliations: "
                f"{candidate_count - semantic_scholar_with_affiliations}"
            ),
            (
                "Semantic Scholar affiliation failures: "
                f"{len(semantic_scholar_failures)}"
            ),
            f"OpenAlex fallback enabled: {self.openalex_enabled}",
            f"OpenAlex fallback attempts: {openalex_attempts}",
            f"OpenAlex papers with affiliations: {openalex_with_affiliations}",
            f"OpenAlex unresolved or empty results: {len(openalex_unavailable)}",
            f"OpenAlex affiliation failures: {len(openalex_failures)}",
            (
                "DOI-only affiliations unknown after all enabled sources: "
                f"{unknown_count}"
            ),
        ]
        if without_ids:
            lines.append("Semantic Scholar author IDs unavailable for:")
            lines.extend(f"- {title}" for title in without_ids)
        if semantic_scholar_failures:
            lines.append("Semantic Scholar affiliation failures for:")
            lines.extend(f"- {title}" for title in semantic_scholar_failures)
        if openalex_unavailable:
            lines.append("OpenAlex affiliations unavailable for:")
            lines.extend(f"- {title}" for title in openalex_unavailable)
        if openalex_failures:
            lines.append("OpenAlex affiliation failures for:")
            lines.extend(f"- {title}" for title in openalex_failures)
        return lines

    def _fetch_openalex_affiliations(
        self, paper: PaperCandidate
    ) -> tuple[dict[str, list[str]] | None, bool]:
        """Return OpenAlex affiliations and whether the lookup failed."""
        doi = paper.doi
        if doi is None or self.openalex_client is None:
            return None, False
        try:
            record = self.openalex_client.resolve_by_doi(doi)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "OpenAlex affiliation fetch failed for '%s' (%s)",
                paper.title,
                type(exc).__name__,
            )
            return None, True
        if record is None:
            return None, False
        return self._affiliations_by_author(paper, record.author_records), False

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
            name = record.name
            if record.author_id is not None:
                name = names_by_id.get(record.author_id, name)
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
