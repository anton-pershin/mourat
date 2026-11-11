"""Semantic Scholar publication-date collector for the newest-paper pipeline."""

from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from comourat.models import PaperIdentity

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,url,abstract,publicationDate,authors,externalIds"


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
    """Collect paper candidates from Semantic Scholar bulk search."""

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
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("queries must contain at least one non-empty query")
        if max_results_per_query is not None and max_results_per_query < 1:
            raise ValueError("max_results_per_query must be positive when set")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.http_client = http_client
        self.queries = [query.strip() for query in queries]
        self.api_url = api_url.rstrip("/")
        self.max_results_per_query = max_results_per_query
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.regular_delay_seconds = regular_delay_seconds
        self.jitter_seconds = jitter_seconds

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
        super().__init__(monitoring_handler)

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        try:
            delay = (
                max(float(retry_after), 0.0)
                if retry_after is not None
                else self.backoff_seconds * (2**attempt)
            )
        except ValueError:
            delay = self.backoff_seconds * (2**attempt)
        return delay + random.uniform(0.0, self.jitter_seconds)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        url = f"{self.api_url}/paper/search/bulk"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http_client.get(
                    url,
                    params=dict(params),
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Semantic Scholar transport failed with %s; retrying in %.1fs",
                    type(exc).__name__,
                    delay,
                )
                if delay:
                    time.sleep(delay)
                continue

            if response.status_code == 200:
                if self.regular_delay_seconds:
                    time.sleep(self.regular_delay_seconds)
                return response.json()
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
            if attempt >= self.max_retries:
                response.raise_for_status()

            delay = self._retry_delay(attempt, response.headers.get("Retry-After"))
            logger.warning(
                "Semantic Scholar bulk search failed with %s; retrying in %.1fs",
                response.status_code,
                delay,
            )
            if delay:
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def _candidate(self, payload: dict[str, Any]) -> PaperCandidate | None:
        title = payload.get("title")
        if not title:
            return None
        raw_date = payload.get("publicationDate")
        if not raw_date:
            return None
        try:
            published = date.fromisoformat(raw_date)
        except (TypeError, ValueError):
            return None
        if not self.start_date <= published <= self.end_date:
            return None

        external_ids = payload.get("externalIds") or {}
        identity = PaperIdentity.from_values(
            arxiv_id=external_ids.get("ArXiv"),
            doi=external_ids.get("DOI"),
        )
        url = payload.get("url")
        return PaperCandidate(
            title=title,
            authors=[
                author["name"]
                for author in payload.get("authors") or []
                if author.get("name")
            ],
            description=payload.get("abstract") or "",
            urls_seen=[url] if url else [],
            arxiv_id=identity.arxiv_id,
            doi=identity.doi,
            publication_date=published.isoformat(),
            provenance=["semantic_scholar"],
            semantic_scholar_id=payload.get("paperId"),
        )

    def _collect_query(
        self, query: str, output: PaperCandidateCollection
    ) -> tuple[int, bool]:
        params: dict[str, Any] = {
            "query": query,
            "sort": "publicationDate:desc",
            "fields": FIELDS,
            "publicationDateOrYear": (
                f"{self.start_date.isoformat()}:{self.end_date.isoformat()}"
            ),
        }
        token: str | None = None
        accepted = 0
        capped = False
        while not capped:
            if token is not None:
                params["token"] = token
            payload = self._get(params)
            for raw_paper in payload.get("data") or []:
                if (
                    self.max_results_per_query is not None
                    and accepted >= self.max_results_per_query
                ):
                    capped = True
                    break
                candidate = self._candidate(raw_paper)
                if candidate is not None:
                    output.papers.append(candidate)
                    accepted += 1
            token = payload.get("token")
            if (
                token is not None
                and self.max_results_per_query is not None
                and accepted >= self.max_results_per_query
            ):
                capped = True
            if token is None:
                break
        if capped:
            logger.warning(
                "Semantic Scholar query reached max_results_per_query=%d: %s",
                self.max_results_per_query,
                query,
            )
        return accepted, capped

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        output = PaperCandidateCollection(papers=[])
        logger.info(
            "Semantic Scholar publication window: %s to %s (inclusive)",
            self.start_date,
            self.end_date,
        )
        lines = [
            "# Publication window",
            f"{self.start_date.isoformat()} to {self.end_date.isoformat()} (inclusive)",
        ]
        for query in self.queries:
            accepted, capped = self._collect_query(query, output)
            lines.append(f"Query `{query}`: {accepted} candidates")
            if capped:
                lines.append(
                    f"Query `{query}`: result cap reached at "
                    f"{self.max_results_per_query} candidates"
                )
        lines.append(f"Total Semantic Scholar candidates: {len(output.papers)}")
        return output, "\n\n".join(lines)
