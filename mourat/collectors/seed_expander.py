"""SeedExpander: seeds -> candidate papers via the citation graph and search.

The three generators of spec 11 FR2, each bounded by its own budget (NFR2):

- **forward**: works citing each resolved seed (`filter=cites:W...`).
- **search**: relevance-ranked title search per seed, paged up to the budget;
  never citation-ranked (FR3 — sorting by citations returns highly-cited
  works unrelated to the query rather than on-topic ones).
- **backward**: works each seed cites (its `referenced_works` list). Empty
  for preprint-only records is normal, not an error (FR2).

Merged candidates are de-duplicated by the metadata API's own work id; a
work produced by several generators appears once and its provenance names
every generator that produced it.
"""

import logging
import time
from typing import Any

from mourat.base import Function
from mourat.clients.openalex import OpenAlexClient
from mourat.data_models import (
    PaperCandidate,
    PaperCandidateCollection,
    SeedCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_title_preview

logger = logging.getLogger(__name__)

GEN_FORWARD = "forward_citations"
GEN_SEARCH = "relevance_search"
GEN_BACKWARD = "backward_references"


def _candidate_from_work(work: dict[str, Any]) -> PaperCandidate | None:
    """Build a candidate from a works record, or None if it has no identity."""
    work_id = work.get("id")
    title = work.get("display_name") or work.get("title") or ""
    if not work_id or not title:
        return None
    authors = [
        a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])
    ]
    authors = [name for name in authors if name]
    return PaperCandidate(
        title=title,
        authors=authors,
        description="",  # canonical resolution fills the real abstract later
        urls_seen=[],
        provenance=[],  # set by the caller from generator names
    )


class SeedExpander(Function[SeedCollection, PaperCandidateCollection]):
    """Runs the three generators, merges, de-duplicates, records provenance."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        openalex_client: OpenAlexClient,
        forward_budget: int = 100,
        search_budget: int = 25,
        backward_budget: int = 100,
    ) -> None:
        self.openalex = openalex_client
        self.forward_budget = forward_budget
        self.search_budget = search_budget
        self.backward_budget = backward_budget
        super().__init__(monitoring_handler)

    def _forward(self, seed) -> list[dict[str, Any]]:
        """Works citing the seed, up to the budget."""
        out: list[dict[str, Any]] = []
        cursor: str | None = "*"
        while cursor is not None and len(out) < self.forward_budget:
            envelope = self.openalex.search_works_citing(seed.work_id, cursor=cursor)
            out.extend(envelope.get("results") or [])
            cursor = (envelope.get("meta") or {}).get("next_cursor")
            if not isinstance(cursor, str):
                break
        return out[: self.forward_budget]

    def _search(self, seed) -> list[dict[str, Any]]:
        """Relevance-ranked title search per seed, up to the budget."""
        out: list[dict[str, Any]] = []
        cursor: str | None = "*"
        while cursor is not None and len(out) < self.search_budget:
            envelope = self.openalex.search_works_by_title(seed.title, cursor=cursor)
            out.extend(envelope.get("results") or [])
            cursor = (envelope.get("meta") or {}).get("next_cursor")
            if not isinstance(cursor, str):
                break
        return out[: self.search_budget]

    def _backward(self, seed) -> list[dict[str, Any]]:
        """Works the seed cites, from its reference list, up to the budget.

        No reference list (preprint-only record) is an empty result, not an
        error (FR2). References arrive as ids; each is fetched as a record.
        """
        reference_ids = self.openalex.get_work_references(seed.work_id)
        records: list[dict[str, Any]] = []
        for work_id in reference_ids[: self.backward_budget]:
            records.append(self.openalex.get_work_by_id(work_id))
        return records

    def _run(self, data: SeedCollection) -> tuple[PaperCandidateCollection, str]:
        merged: dict[str, tuple[PaperCandidate, set[str]]] = {}
        for seed in data.seeds:
            for generator, records in (
                (GEN_FORWARD, self._forward(seed)),
                (GEN_SEARCH, self._search(seed)),
                (GEN_BACKWARD, self._backward(seed)),
            ):
                total_records = len(records)
                for i, work in enumerate(records):
                    key = work.get("id") if isinstance(work, dict) else str(work)
                    if not key:
                        continue
                    if key in merged:
                        merged[key][1].add(generator)
                        continue
                    candidate = _candidate_from_work(
                        work if isinstance(work, dict) else {"id": work}
                    )
                    if candidate is not None:
                        merged[key] = (candidate, {generator})

                logger.debug(
                    "processed %d records during %s expansion of '%s'",
                    len(records),
                    generator,
                    to_title_preview(seed.title),
                )

        # provenance naming every generator that produced the candidate (FR2)
        for work_id, (candidate, generators) in merged.items():
            candidate.provenance = sorted(generators)

        lines = [f"Seeds expanded: {len(data.seeds)}, candidates out: {len(merged)}"]
        per_gen: dict[str, int] = {GEN_FORWARD: 0, GEN_SEARCH: 0, GEN_BACKWARD: 0}
        for _, (_, generators) in merged.items():
            for g in generators:
                per_gen[g] += 1
        for name, count in per_gen.items():
            lines.append(f"{name}: {count} candidates")
        logger.debug("seed expansion produced %d candidates", len(merged))
        return (
            PaperCandidateCollection(papers=[c for c, _ in merged.values()]),
            "\n".join(lines),
        )
