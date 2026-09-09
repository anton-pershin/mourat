"""SeedResolver: stored content items -> seeds with metadata work ids.

Per spec 11 §4.1: seeds carry no stored identifier, so each is re-resolved at
the start of a run — title lookup via `OpenAlexClient` with the same title
cross-check `PaperResolver` applies. A seed that fails the cross-check (or the
lookup) is reported and skipped: it neither expands nor contributes to the
floor derivation (FR4), so the failure is visible rather than silent.

The resolved seeds' influence distribution (the stored `influence_score`
values, normalised 0-100 by the same convention the DB writer uses) is
exposed for `InfluenceFloorFilter`'s percentile derivation.
"""

import logging
import time

from mourat.base import Function
from mourat.clients.openalex import OpenAlexClient
from mourat.data_models import (
    ContentItemCollection,
    Seed,
    SeedCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.similarity import titles_match

logger = logging.getLogger(__name__)


class SeedResolver(Function[ContentItemCollection, SeedCollection]):
    """Resolves stored content items into seeds; skips unresolvable ones."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        openalex_client: OpenAlexClient,
        title_similarity_threshold: float = 0.9,
    ) -> None:
        self.openalex = openalex_client
        self.title_similarity_threshold = title_similarity_threshold
        super().__init__(monitoring_handler)

    def _resolve_one(self, item) -> tuple[Seed | None, str]:
        """Resolve one content item; returns (seed or None, reason)."""
        envelope = self.openalex.search_works_by_title(item.name)
        results = envelope.get("results") or []
        if not results:
            return None, "no_work_found"
        best = results[0]
        work_title = best.get("display_name") or best.get("title") or ""
        if not titles_match(item.name, work_title, self.title_similarity_threshold):
            return None, "openalex_title_mismatch"
        work_id = best.get("id") or None
        if work_id is None:
            return None, "no_work_id"
        return (
            Seed(
                content_item_id=item.id,
                work_id=work_id,
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
        resolved: list[Seed] = []
        skipped: list[tuple[str, str]] = []
        total_items = len(data.items)
        for i, item in enumerate(data.items):
            try:
                seed, reason = self._resolve_one(item)
            except Exception:
                logger.exception("resolution request failed for seed '%s'", item.name)
                seed, reason = None, "api_error"
            if seed is None:
                logger.error("skipped seed '%s': %s", item.name, reason)
                skipped.append((item.name, reason))
            else:
                logger.debug("resolved %d/%d: '%s'", i + 1, total_items, item.name)
                resolved.append(seed)

        lines = [
            f"Seeds in: {len(data.items)}, resolved: {len(resolved)}, "
            f"skipped: {len(skipped)}"
        ]
        for name, reason in skipped:
            lines.append(f"### SEED SKIPPED: {name}\nReason: {reason}\n")
        for seed in resolved:
            influence = (
                f"{seed.influence_value:.1f}"
                if seed.influence_value is not None
                else "-"
            )
            lines.append(
                f"### {seed.title}\nWork id: {seed.work_id} | influence: {influence}\n"
            )
        logger.info("resolved %d/%d seeds", len(resolved), len(data.items))
        return SeedCollection(seeds=resolved), "\n".join(lines)
