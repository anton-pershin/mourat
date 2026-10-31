"""Paper resolver: Function adapter over comourat's neutral resolution core."""

from __future__ import annotations

import logging
import time

from comourat.clients.arxiv import ArxivClient
from comourat.clients.paper_graph import PaperGraphClient
from comourat.resolution.paper_resolver import (
    PaperResolverCore,
    extract_arxiv_id as _extract_arxiv_id,
)
from mourat.base import Function
from mourat.data_models import (
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
)
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)
STATUS_RESOLVED = "resolved"


class PaperResolver(Function[PaperCandidateCollection, ResolvedPaperCollection]):
    """Resolve candidates to canonical metadata, dropping unresolved items."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        paper_graph_client: PaperGraphClient | None = None,
        arxiv_client: ArxivClient | None = None,
        title_similarity_threshold: float = 0.9,
        openalex_client: PaperGraphClient | None = None,
    ) -> None:
        client = paper_graph_client or openalex_client
        if client is None:
            raise ValueError("paper_graph_client is required")
        # `arxiv_client` remains accepted for existing Hydra configurations.
        # Identity lookup and canonical metadata retrieval use the graph client.
        self.core = PaperResolverCore(
            paper_graph_client=client,
            title_similarity_threshold=title_similarity_threshold,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        started = time.monotonic()
        candidates = data.papers
        result = self.core.resolve(candidates)
        resolved: list[ResolvedPaper] = []
        dropped: list[tuple[str, str]] = []

        for index, outcome in enumerate(result.resolutions, start=1):
            candidate = outcome.candidate
            logger.info(
                "resolving paper %d/%d: '%s'",
                index,
                len(candidates),
                candidate.title,
            )
            if outcome.record is None:
                reason = outcome.reason or "no_paper_found"
                if reason == "api_error":
                    logger.warning(
                        "resolution request failed for '%s'", candidate.title
                    )
                dropped.append((candidate.title, reason))
                continue

            record = outcome.record
            identity = record.identity
            resolved.append(
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
                    provenance=getattr(candidate, "provenance", ["web_search"]),
                    doi=identity.doi,
                    arxiv_id=identity.arxiv_id,
                    influence_fwci=record.raw_influence.get("fwci"),
                    influence_cited_by_count=record.citation_count,
                    resolution_status=STATUS_RESOLVED,
                )
            )

        lines = [
            f"Candidates in: {len(candidates)}, resolved: {len(resolved)}, unresolved (dropped): {len(dropped)}"
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
            len(candidates),
            time.monotonic() - started,
        )
        return ResolvedPaperCollection(papers=resolved), "\n".join(lines)
