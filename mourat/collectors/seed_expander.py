"""Expand seeds through the selected normalized paper graph client."""

from __future__ import annotations

import logging
from typing import Any

from mourat.base import Function
from mourat.clients.paper_graph import PaperGraphClient, PaperIdentity, PaperRecord
from mourat.data_models import PaperCandidate, PaperCandidateCollection, SeedCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_title_preview

logger = logging.getLogger(__name__)

GEN_FORWARD = "forward_citations"
GEN_SEARCH = "relevance_search"
GEN_BACKWARD = "backward_references"


def _candidate_from_record(record: PaperRecord) -> PaperCandidate | None:
    if not record.title or (
        record.identity.arxiv_id is None and record.identity.doi is None
    ):
        return None
    return PaperCandidate(
        title=record.title,
        authors=record.authors,
        description=record.abstract,
        urls_seen=[],
        arxiv_id=record.identity.arxiv_id,
        doi=record.identity.doi,
        provenance=[],
    )


class SeedExpander(Function[SeedCollection, PaperCandidateCollection]):
    """Run forward, relevance-search, and backward graph generators."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        paper_graph_client: PaperGraphClient | None = None,
        forward_budget: int = 100,
        search_budget: int = 25,
        backward_budget: int = 100,
        openalex_client: PaperGraphClient | None = None,
    ) -> None:
        self.client = paper_graph_client or openalex_client
        if self.client is None:
            raise ValueError("paper_graph_client is required")
        self.forward_budget = forward_budget
        self.search_budget = search_budget
        self.backward_budget = backward_budget
        super().__init__(monitoring_handler)

    @staticmethod
    def _identity(seed) -> PaperIdentity:
        return PaperIdentity.from_values(arxiv_id=seed.arxiv_id, doi=seed.doi)

    def _forward(self, seed) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        continuation: Any = None
        while len(out) < self.forward_budget:
            page = self.client.get_citations(self._identity(seed), continuation)
            out.extend(page.papers)
            if page.continuation is None:
                break
            continuation = page.continuation
        return out[: self.forward_budget]

    def _search(self, seed) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        continuation: Any = None
        while len(out) < self.search_budget:
            page = self.client.search_papers(seed.title, continuation)
            out.extend(page.papers)
            if page.continuation is None:
                break
            continuation = page.continuation
        return out[: self.search_budget]

    def _backward(self, seed) -> list[PaperRecord]:
        return self.client.get_references(
            self._identity(seed), limit=self.backward_budget
        )[: self.backward_budget]

    @staticmethod
    def _key(record: PaperRecord) -> tuple[str, str]:
        if record.identity.arxiv_id:
            return ("arxiv", record.identity.arxiv_id)
        return ("doi", (record.identity.doi or "").lower())

    def _run(self, data: SeedCollection) -> tuple[PaperCandidateCollection, str]:
        merged: dict[tuple[str, str], tuple[PaperCandidate, set[str]]] = {}
        per_gen = {GEN_FORWARD: 0, GEN_SEARCH: 0, GEN_BACKWARD: 0}
        for seed in data.seeds:
            for generator, records in (
                (GEN_FORWARD, self._forward(seed)),
                (GEN_SEARCH, self._search(seed)),
                (GEN_BACKWARD, self._backward(seed)),
            ):
                for record in records:
                    candidate = _candidate_from_record(record)
                    if candidate is None:
                        continue
                    key = self._key(record)
                    if key in merged:
                        merged[key][1].add(generator)
                    else:
                        merged[key] = (candidate, {generator})
                    per_gen[generator] += 1
            logger.debug("expanded seed '%s'", to_title_preview(seed.title))
        for candidate, generators in merged.values():
            candidate.provenance = sorted(generators)
        lines = [f"Seeds expanded: {len(data.seeds)}, candidates out: {len(merged)}"]
        lines.extend(f"{name}: {count} candidates" for name, count in per_gen.items())
        return PaperCandidateCollection(
            papers=[x[0] for x in merged.values()]
        ), "\n".join(lines)
