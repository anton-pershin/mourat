"""JSONL file writer pipeline stage for scored papers."""

from __future__ import annotations

import json
import logging

from mourat.base import Function
from mourat.data_models import ScoredPaperCollection
from mourat.database import content_item as ci
from mourat.monitoring import MonitoringHandler
from mourat.writers.db_writer import _content_item_id

logger = logging.getLogger(__name__)


class JsonlWriter(Function[ScoredPaperCollection, ScoredPaperCollection]):
    """Writes one JSON line per scored paper to a file.

    A pass-through stage that returns its input unchanged. Each line carries
    the content item attributes together with the relevance scores and
    justifications, so a run can be inspected without a database. When DB
    persistence is disabled this file is the run's output — never a discarded
    local variable.

    Appends to the file: a re-run adds a new record set rather than
    destroying the previous one. When ``skip_existing`` is enabled, papers
    already stored in the supplied database connection are excluded.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        output_path: str,
        skip_existing: bool = False,
        conn=None,
    ) -> None:
        if skip_existing and conn is None:
            raise ValueError("conn is required when skip_existing is enabled")
        self.output_path = output_path
        self.skip_existing = skip_existing
        self.conn = conn
        super().__init__(monitoring_handler)

    def _record(self, sp) -> dict:
        """One JSON-serialisable record for a scored paper."""
        return {
            "title": sp.paper.title,
            "authors": sp.paper.authors,
            "abstract": sp.paper.abstract,
            "url": sp.paper.url,
            "publication_date": sp.paper.publication_date,
            "influence_score": sp.paper.influence_score,
            "provenance": sp.paper.provenance,
            "relevance_scores": [se.model_dump() for se in sp.relevance_scores],
            "filtering_score": sp.filtering_score,
        }

    def _run(self, data: ScoredPaperCollection) -> tuple[ScoredPaperCollection, str]:
        papers = data.papers
        skipped = 0
        if self.skip_existing:
            kept = []
            for sp in papers:
                item_id = _content_item_id(
                    sp.paper.title, getattr(sp.paper, "arxiv_id", None)
                )
                if ci.get_content_item(self.conn, item_id) is None:
                    kept.append(sp)
                else:
                    skipped += 1
            papers = kept

        with open(self.output_path, "w", encoding="utf-8") as f:
            for sp in papers:
                f.write(json.dumps(self._record(sp), ensure_ascii=False) + "\n")

        logger.info("wrote %d papers to %s", len(papers), self.output_path)
        text_for_monitoring = (
            f"Wrote {len(papers)} papers to {self.output_path}; "
            f"written: {len(papers)}, skipped: {skipped}"
        )
        return ScoredPaperCollection(papers=papers), text_for_monitoring
