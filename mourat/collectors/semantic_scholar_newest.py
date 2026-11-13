"""Thin Semantic Scholar publication-window adapter over Comourat."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from comourat.collectors.semantic_scholar import SemanticScholarPaperCollectorCore
from comourat.models import SemanticScholarBulkSearchResult

from mourat.base import Function
from mourat.data_models import PaperCandidateCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"


def publication_window(
    reference_date: date,
    lag_days: int,
    stride_days: int,
    backfill_days: int,
) -> tuple[date, date]:
    """Return the inclusive publication window for one collection run."""
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    if stride_days < 1:
        raise ValueError("stride_days must be positive")
    if backfill_days < 0:
        raise ValueError("backfill_days must be non-negative")
    end_date = reference_date - timedelta(days=lag_days)
    start_date = end_date - timedelta(days=stride_days - 1 + backfill_days)
    return start_date, end_date


class SemanticScholarNewestPaperCollector(Function[Any, PaperCandidateCollection]):
    """Delegate publication-window collection to the Comourat core."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        queries: list[str],
        api_url: str = BASE_URL,
        lag_days: int = 30,
        stride_days: int = 1,
        backfill_days: int = 0,
        reference_date: str | None = None,
        max_results_per_query: int | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        regular_delay_seconds: float = 4.0,
        jitter_seconds: float = 1.0,
    ) -> None:
        run_date = (
            date.fromisoformat(reference_date)
            if reference_date is not None
            else datetime.now(timezone.utc).date()
        )
        self.start_date, self.end_date = publication_window(
            reference_date=run_date,
            lag_days=lag_days,
            stride_days=stride_days,
            backfill_days=backfill_days,
        )
        self.max_results_per_query = max_results_per_query
        self.core = SemanticScholarPaperCollectorCore(
            http_client=http_client,
            api_url=api_url.rstrip("/"),
            mode="publication_window",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            queries=queries,
            max_results_per_query=max_results_per_query,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            regular_delay_seconds=regular_delay_seconds,
            jitter_seconds=jitter_seconds,
        )
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        logger.info(
            "Semantic Scholar publication window: %s to %s (inclusive)",
            self.start_date,
            self.end_date,
        )
        result = self.core.collect()
        if not isinstance(result, SemanticScholarBulkSearchResult):
            raise TypeError("Comourat publication_window returned an unexpected result")

        output = PaperCandidateCollection(papers=list(result.papers))
        lines = [
            "# Publication window",
            f"{self.start_date.isoformat()} to {self.end_date.isoformat()} (inclusive)",
        ]
        for query_result in result.query_results:
            lines.append(
                f"Query `{query_result.query}`: "
                f"{query_result.candidate_count} candidates"
            )
            if query_result.cap_reached:
                lines.append(
                    f"Query `{query_result.query}`: result cap reached at "
                    f"{self.max_results_per_query} candidates"
                )
        lines.append(f"Total Semantic Scholar candidates: {len(output.papers)}")
        return output, "\n\n".join(lines)
