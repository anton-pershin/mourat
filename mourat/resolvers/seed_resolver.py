"""Resolve stored content items into canonical paper-graph seeds."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from mourat.base import Function
from mourat.clients.paper_graph import PaperGraphClient, PaperIdentity, PaperRecord
from mourat.data_models import ContentItemCollection, Seed, SeedCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.similarity import titles_match

logger = logging.getLogger(__name__)


def _legacy_record(work: dict[str, Any]) -> PaperRecord:
    doi = work.get("doi")
    arxiv_id = None
    for location in work.get("locations") or []:
        landing = location.get("landing_page_url") or ""
        if "arxiv.org/abs/" in landing:
            arxiv_id = landing.rsplit("/", 1)[-1].split("v", 1)[0]
            break
    authors = [
        entry.get("author", {}).get("display_name", "")
        for entry in work.get("authorships", [])
    ]
    abstract = ""
    index = work.get("abstract_inverted_index") or {}
    words = [
        (position, word) for word, positions in index.items() for position in positions
    ]
    abstract = " ".join(word for _, word in sorted(words))
    return PaperRecord(
        identity=PaperIdentity.from_values(arxiv_id=arxiv_id, doi=doi),
        title=work.get("display_name") or work.get("title") or "",
        authors=[name for name in authors if name],
        abstract=abstract,
        publication_date=(
            date.fromisoformat(work["publication_date"])
            if work.get("publication_date")
            else None
        ),
        citation_count=work.get("cited_by_count"),
        raw_influence=(
            {"fwci": work["fwci"]} if isinstance(work.get("fwci"), (int, float)) else {}
        ),
    )


class SeedResolver(Function[ContentItemCollection, SeedCollection]):
    """Resolve seeds through the selected provider-neutral graph client."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        paper_graph_client: PaperGraphClient | None = None,
        title_similarity_threshold: float = 0.9,
        openalex_client: PaperGraphClient | None = None,
    ) -> None:
        self.client = paper_graph_client or openalex_client
        if self.client is None:
            raise ValueError("paper_graph_client is required")
        self.title_similarity_threshold = title_similarity_threshold
        super().__init__(monitoring_handler)

    def _resolve_one(self, item) -> tuple[Seed | None, str]:
        if hasattr(self.client, "search_papers"):
            records = self.client.search_papers(item.name).papers
        else:
            envelope = self.client.search_works_by_title(item.name)
            records = [_legacy_record(work) for work in envelope.get("results") or []]
        if not records:
            return None, "no_paper_found"
        best = records[0]
        if not titles_match(item.name, best.title, self.title_similarity_threshold):
            return None, "provider_title_mismatch"
        identity = best.identity
        if identity.arxiv_id is None and identity.doi is None:
            return None, "no_canonical_identity"
        return (
            Seed(
                content_item_id=item.id,
                arxiv_id=identity.arxiv_id,
                doi=identity.doi,
                title=item.name,
                influence_value=(
                    float(item.influence_score)
                    if item.influence_score is not None
                    else None
                ),
            ),
            "resolved",
        )

    def _run(self, data: ContentItemCollection) -> tuple[SeedCollection, str]:
        resolved, skipped = [], []
        for item in data.items:
            try:
                seed, reason = self._resolve_one(item)
            except Exception:
                logger.exception("seed resolution failed for '%s'", item.name)
                seed, reason = None, "api_error"
            (resolved if seed is not None else skipped).append(
                seed if seed is not None else (item.name, reason)
            )
        lines = [
            f"Seeds in: {len(data.items)}, resolved: {len(resolved)}, skipped: {len(skipped)}"
        ]
        lines.extend(
            f"### SEED SKIPPED: {name}\nReason: {reason}\n" for name, reason in skipped
        )
        for seed in resolved:
            lines.append(
                f"### {seed.title}\nIdentifiers: arxiv={seed.arxiv_id or '-'} doi={seed.doi or '-'} | influence: {seed.influence_value if seed.influence_value is not None else '-'}\n"
            )
        return SeedCollection(seeds=resolved), "\n".join(lines)
