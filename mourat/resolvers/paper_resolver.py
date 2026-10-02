"""Resolve candidates through a provider-neutral paper graph client."""

from __future__ import annotations

import logging
import re
import time

from mourat.base import Function
from mourat.clients.arxiv import ArxivClient
from mourat.clients.paper_graph import PaperGraphClient, PaperRecord
from mourat.data_models import (
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.similarity import titles_match

logger = logging.getLogger(__name__)
_ARXIV_ID_IN_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.\d{4,5})(?:v[0-9]+)?/?"
)
STATUS_RESOLVED = "resolved"


def _abstract_from_inverted_index(inv: dict[str, list[int]] | None) -> str:
    """Reconstruct an OpenAlex inverted-index abstract for compatibility."""
    if not inv:
        return ""
    positions = [
        (position, word) for word, indexes in inv.items() for position in indexes
    ]
    positions.sort()
    return " ".join(word for _, word in positions)


def _extract_arxiv_id(urls: list[str]) -> str | None:
    for url in urls:
        match = _ARXIV_ID_IN_URL.search(url)
        if match:
            return match.group(1)
    return None


class PaperResolver(Function[PaperCandidateCollection, ResolvedPaperCollection]):
    """Resolve candidates by canonical identity, then title search fallback."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        paper_graph_client: PaperGraphClient | None = None,
        arxiv_client: ArxivClient | None = None,
        title_similarity_threshold: float = 0.9,
        openalex_client: PaperGraphClient | None = None,
    ) -> None:
        self.client = paper_graph_client or openalex_client
        if self.client is None:
            raise ValueError("paper_graph_client is required")
        self.arxiv = arxiv_client
        self.title_similarity_threshold = title_similarity_threshold
        super().__init__(monitoring_handler)

    def _resolve_record(self, candidate):
        arxiv_id = candidate.arxiv_id or _extract_arxiv_id(candidate.urls_seen)
        record = self.client.resolve_by_arxiv_id(arxiv_id) if arxiv_id else None
        if record is None and candidate.doi:
            record = self.client.resolve_by_doi(candidate.doi)
        if record is None:
            page = self.client.search_papers(candidate.title)
            record = page.papers[0] if page.papers else None
        if record is None:
            return None, "no_paper_found"
        if not titles_match(
            candidate.title, record.title, self.title_similarity_threshold
        ):
            return None, "provider_title_mismatch"
        return record, "resolved"

    def _resolve_one(self, candidate):
        try:
            record, reason = self._resolve_record(candidate)
        except Exception:
            logger.exception("resolution request failed for '%s'", candidate.title)
            return None, "api_error"
        if record is None:
            return None, reason
        identity = record.identity
        return (
            ResolvedPaper(
                title=record.title,
                abstract=record.abstract or candidate.description,
                authors=record.authors,
                publication_date=(
                    record.publication_date.isoformat()
                    if record.publication_date
                    else None
                ),
                url="",
                provenance=candidate.provenance,
                doi=identity.doi,
                arxiv_id=identity.arxiv_id,
                influence_fwci=record.raw_influence.get("fwci"),
                influence_cited_by_count=record.citation_count,
                resolution_status=STATUS_RESOLVED,
            ),
            "resolved",
        )

    def _run(self, data: PaperCandidateCollection):
        started = time.monotonic()
        resolved, dropped = [], []
        for candidate in data.papers:
            item, reason = self._resolve_one(candidate)
            (resolved if item is not None else dropped).append(
                item if item is not None else (candidate.title, reason)
            )
        lines = [
            f"Candidates in: {len(data.papers)}, resolved: {len(resolved)}, unresolved (dropped): {len(dropped)}"
        ]
        for title, reason in dropped:
            lines.append(f"### UNRESOLVED: {title}\nReason: {reason}\n")
        for item in resolved:
            lines.append(
                f"### {item.title}\n"
                f"Identifiers: doi={item.doi or '-'} arxiv={item.arxiv_id or '-'}\n"
                f"Provenance: {', '.join(item.provenance) or '-'}\n"
            )
        logger.debug(
            "resolved %d/%d candidates | %.2fs",
            len(resolved),
            len(data.papers),
            time.monotonic() - started,
        )
        return ResolvedPaperCollection(papers=resolved), "\n".join(lines)
