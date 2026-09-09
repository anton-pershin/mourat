"""JSONL file writer pipeline stage for scored papers."""

from __future__ import annotations

import json
import logging

from mourat.base import Function
from mourat.data_models import ScoredPaperCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)


class JsonlWriter(Function[ScoredPaperCollection, ScoredPaperCollection]):
    """Writes one JSON line per scored paper to a file.

    A pass-through stage that returns its input unchanged. Each line carries
    the content item attributes together with the relevance scores and
    justifications, so a run can be inspected without a database. When DB
    persistence is disabled this file is the run's output — never a discarded
    local variable.

    Appends to the file: a re-run adds a new record set rather than
    destroying the previous one.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        output_path: str,
    ) -> None:
        self.output_path = output_path
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
        with open(self.output_path, "a", encoding="utf-8") as f:
            for sp in data.papers:
                f.write(json.dumps(self._record(sp), ensure_ascii=False) + "\n")

        logger.info("wrote %d papers to %s", len(data.papers), self.output_path)
        text_for_monitoring = f"Wrote {len(data.papers)} papers to {self.output_path}"
        return data, text_for_monitoring
