"""Semantic Scholar publication-date collector for the newest-paper pipeline.

This adapter is intentionally local to mourat. The pinned Comourat bulk-search
collector does not expose provider paper IDs or external identifiers, which are
required for durable idempotency and DOI-based output identity.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable

import httpx
from comourat.models import PaperCandidate as ComouratPaperCandidate
from comourat.models import normalize_arxiv_id, normalize_doi

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.database import semantic_scholar as s2db
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

_FIELDS = "paperId,title,url,abstract,publicationDate,authors,externalIds"


@dataclass(frozen=True)
class SemanticScholarQueryPlan:
    """Publication windows and the state to commit after a successful run."""

    query: str
    windows: list[tuple[date, date]]
    completed_end_date: date | None
    backfill_completed_on: date | None


@dataclass
class SemanticScholarCollectionReceipt:
    """New provider IDs and cursors produced by one collection call."""

    paper_ids: list[str] = field(default_factory=list)
    query_states: list[dict] = field(default_factory=list)


def _state_date(state: dict | None, key: str) -> date | None:
    value = state.get(key) if state else None
    return date.fromisoformat(value) if isinstance(value, str) else value


def _date_chunks(start: date, end: date, stride_days: int) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=stride_days - 1), end)
        windows.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return windows


def plan_query_windows(
    query: str,
    state: dict | None,
    today: date,
    publication_lag_days: int,
    stride_days: int,
    backfill_days: int,
    backfill_interval_days: int,
) -> SemanticScholarQueryPlan:
    """Build a catch-up plan, extended by a backfill when it is due."""
    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be non-negative")
    if stride_days < 1:
        raise ValueError("stride_days must be positive")
    if backfill_days < 0:
        raise ValueError("backfill_days must be non-negative")
    if backfill_interval_days < 1:
        raise ValueError("backfill_interval_days must be positive")

    target_end = today - timedelta(days=publication_lag_days)
    last_end = _state_date(state, "last_completed_end_date")
    last_backfill = _state_date(state, "last_backfill_completed_on")
    normal_start = (
        last_end + timedelta(days=1)
        if last_end is not None
        else target_end - timedelta(days=stride_days - 1)
    )
    backfill_due = backfill_days > 0 and (
        last_backfill is None or (today - last_backfill).days >= backfill_interval_days
    )

    range_start = normal_start
    if backfill_due:
        backfill_start = target_end - timedelta(days=backfill_days - 1)
        range_start = min(range_start, backfill_start)

    windows = (
        _date_chunks(range_start, target_end, stride_days)
        if range_start <= target_end
        else []
    )
    if not windows:
        return SemanticScholarQueryPlan(
            query=query,
            windows=[],
            completed_end_date=None,
            backfill_completed_on=last_backfill,
        )

    completed_end = max(last_end, target_end) if last_end is not None else target_end
    return SemanticScholarQueryPlan(
        query=query,
        windows=windows,
        completed_end_date=completed_end,
        backfill_completed_on=today if backfill_due else last_backfill,
    )


def _normalized_doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    return normalized.lower() if normalized else None


class SemanticScholarNewestPaperCollector(Function[Any, PaperCandidateCollection]):
    """Collect new Semantic Scholar records and expose a deferred-commit receipt."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        conn: sqlite3.Connection,
        queries: Iterable[str],
        api_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: str | None = None,
        publication_lag_days: int = 30,
        stride_days: int = 1,
        backfill_days: int = 90,
        backfill_interval_days: int = 7,
        max_results_per_query: int | None = None,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        regular_delay_seconds: float = 1.0,
        run_date: date | str | None = None,
    ) -> None:
        self.http_client = http_client
        self.conn = conn
        self.queries = [query.strip() for query in queries]
        if not self.queries or any(not query for query in self.queries):
            raise ValueError(
                "semantic_scholar_newest requires at least one non-empty query"
            )
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("semantic_scholar_newest queries must be unique")
        if max_results_per_query is not None and max_results_per_query < 1:
            raise ValueError("max_results_per_query must be positive when set")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.publication_lag_days = publication_lag_days
        self.stride_days = stride_days
        self.backfill_days = backfill_days
        self.backfill_interval_days = backfill_interval_days
        self.max_results_per_query = max_results_per_query
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.regular_delay_seconds = regular_delay_seconds
        if isinstance(run_date, str):
            run_date = date.fromisoformat(run_date)
        self.run_date = run_date
        self.receipt = SemanticScholarCollectionReceipt()
        super().__init__(monitoring_handler)

    def _request_page(self, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        url = f"{self.api_url}/paper/search/bulk"
        for attempt in range(self.max_retries + 1):
            response = self.http_client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                if self.regular_delay_seconds:
                    time.sleep(self.regular_delay_seconds)
                return response.json()
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
            if attempt >= self.max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(float(retry_after), 0.0)
                except ValueError:
                    delay = self.backoff_seconds * (2**attempt)
            else:
                delay = self.backoff_seconds * (2**attempt)
            logger.warning(
                "Semantic Scholar bulk search failed with %s; retrying in %.1fs",
                response.status_code,
                delay,
            )
            if delay:
                time.sleep(delay)
        raise RuntimeError("Semantic Scholar retry loop ended unexpectedly")

    def _collect_window(
        self,
        query: str,
        start: date,
        end: date,
        max_results: int | None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "sort": "publicationDate:desc",
            "publicationDateOrYear": f"{start.isoformat()}:{end.isoformat()}",
            "fields": _FIELDS,
        }
        records: list[dict[str, Any]] = []
        token: str | None = None
        while max_results is None or len(records) < max_results:
            if token is not None:
                params["token"] = token
            payload = self._request_page(params)
            page = payload.get("data") or []
            if max_results is None:
                records.extend(page)
            else:
                records.extend(page[: max_results - len(records)])
            token = payload.get("token")
            if token is None:
                break
        return records

    @staticmethod
    def _record_doi(record: dict[str, Any]) -> str | None:
        external_ids = record.get("externalIds") or {}
        return _normalized_doi(external_ids.get("DOI"))

    @classmethod
    def _combine_records(
        cls, records: Iterable[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        for record in records:
            paper_id = record.get("paperId")
            if not paper_id:
                continue
            paper_id = str(paper_id)
            current = combined.get(paper_id)
            # Prefer the representation with a DOI when query results differ.
            if current is None or (
                not cls._record_doi(current) and cls._record_doi(record)
            ):
                combined[paper_id] = record
        return combined

    @staticmethod
    def _candidate(record: dict[str, Any]) -> PaperCandidate | None:
        title = record.get("title")
        paper_id = record.get("paperId")
        if not title or not paper_id:
            return None
        external_ids = record.get("externalIds") or {}
        doi = _normalized_doi(external_ids.get("DOI"))
        arxiv_id = normalize_arxiv_id(external_ids.get("ArXiv"))
        url = record.get("url")
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if not url:
            url = f"https://www.semanticscholar.org/paper/{paper_id}"
        authors = [
            author["name"]
            for author in record.get("authors") or []
            if author.get("name")
        ]
        return PaperCandidate(
            title=title,
            authors=authors,
            description=record.get("abstract") or "",
            urls_seen=[url],
            arxiv_id=arxiv_id,
            doi=doi,
            publication_date=record.get("publicationDate"),
            provenance=["semantic_scholar"],
        )

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        self.receipt = SemanticScholarCollectionReceipt()
        today = self.run_date or datetime.now(UTC).date()
        all_records: list[dict[str, Any]] = []
        query_states: list[dict] = []
        monitoring_lines = []

        for query in self.queries:
            state = s2db.load_query_state(self.conn, query)
            plan = plan_query_windows(
                query=query,
                state=state,
                today=today,
                publication_lag_days=self.publication_lag_days,
                stride_days=self.stride_days,
                backfill_days=self.backfill_days,
                backfill_interval_days=self.backfill_interval_days,
            )
            query_records: list[dict[str, Any]] = []
            completed_end: date | None = None
            all_windows_completed = True
            for index, (start, end) in enumerate(plan.windows):
                remaining = (
                    None
                    if self.max_results_per_query is None
                    else self.max_results_per_query - len(query_records)
                )
                if remaining is not None and remaining <= 0:
                    all_windows_completed = False
                    break
                page_records = self._collect_window(query, start, end, remaining)
                query_records.extend(page_records)
                completed_end = end
                monitoring_lines.append(
                    f"Query '{query}' | {start.isoformat()}:{end.isoformat()} | "
                    f"records: {len(page_records)}"
                )
                if (
                    self.max_results_per_query is not None
                    and len(query_records) >= self.max_results_per_query
                    and index < len(plan.windows) - 1
                ):
                    all_windows_completed = False
                    break

            all_records.extend(query_records)
            if all_windows_completed and plan.completed_end_date is not None:
                query_states.append(
                    {
                        "query": query,
                        "completed_end_date": plan.completed_end_date,
                        "backfill_completed_on": plan.backfill_completed_on,
                    }
                )
            elif completed_end is not None:
                # A result cap completed only a prefix of the planned windows.
                # Leave backfill due so a later run revisits the full range.
                query_states.append(
                    {
                        "query": query,
                        "completed_end_date": completed_end,
                        "backfill_completed_on": None,
                    }
                )

        combined = self._combine_records(all_records)
        seen = s2db.get_seen_paper_ids(self.conn, combined.keys())
        new_records = {
            paper_id: record
            for paper_id, record in combined.items()
            if paper_id not in seen
        }
        candidates: list[ComouratPaperCandidate] = [
            candidate
            for record in new_records.values()
            if (candidate := self._candidate(record)) is not None
        ]
        self.receipt = SemanticScholarCollectionReceipt(
            paper_ids=sorted(new_records),
            query_states=query_states,
        )
        monitoring_lines.extend(
            [
                f"Unique provider records: {len(combined)}",
                f"Previously committed: {len(seen)}",
                f"New candidates: {len(candidates)}",
            ]
        )
        return PaperCandidateCollection(papers=candidates), "\n".join(monitoring_lines)

    def commit(self) -> None:
        """Persist the latest receipt after all downstream stages succeed."""
        s2db.record_successful_collection(
            self.conn,
            paper_ids=self.receipt.paper_ids,
            query_states=self.receipt.query_states,
        )
