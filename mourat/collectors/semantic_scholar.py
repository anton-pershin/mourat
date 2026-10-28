"""Semantic Scholar paper collector: Function adapter over comourat."""

from __future__ import annotations

from typing import Any

import httpx

from comourat.collectors.semantic_scholar import (
    SemanticScholarPaperCollectorCore,
    SemanticScholarSearchMode,
)
from comourat.models import PaperInfoCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler


class SemanticScholarPaperCollector(Function[Any, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        api_url: str,
        mode: SemanticScholarSearchMode,
        start_date: str | None,  # YYYY-MM-DD
        end_date: str | None,  # YYYY-MM-DD
        max_results: int,
        strict_keyword_query: str,
    ) -> None:
        self.core = SemanticScholarPaperCollectorCore(
            http_client=http_client,
            api_url=api_url,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            strict_keyword_query=strict_keyword_query,
        )
        self.strict_keyword_query = strict_keyword_query
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperInfoCollection, str]:
        output = self.core.collect()

        text_for_monitoring = (
            "# Query\n"
            f"{self.strict_keyword_query}\n\n"
            "# Papers\n"
            f"Total {len(output.papers)}"
        )

        return output, text_for_monitoring
