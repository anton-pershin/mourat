"""Tests for Semantic Scholar newest-paper collection and merging."""

from __future__ import annotations

import sqlite3
from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.database import init_db
from mourat.monitoring import MonitoringHandler


class _Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _response(payload, status_code=200, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.headers = headers or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "request failed", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _paper(
    paper_id="ss-1",
    title="Journal paper",
    doi="10.1234/journal",
    arxiv_id=None,
    publication_date="2024-01-02",
):
    external_ids = {}
    if doi is not None:
        external_ids["DOI"] = doi
    if arxiv_id is not None:
        external_ids["ArXiv"] = arxiv_id
    return {
        "paperId": paper_id,
        "title": title,
        "url": f"https://example.com/{paper_id}",
        "abstract": "Abstract",
        "citationCount": 3,
        "publicationDate": publication_date,
        "authors": [{"name": "A. Author"}],
        "externalIds": external_ids,
    }


def _collector(http_client, handler=None, **overrides):
    from mourat.collectors.semantic_scholar_newest import (
        SemanticScholarNewestPaperCollector,
    )

    values = {
        "monitoring_handler": handler or _Handler(),
        "http_client": http_client,
        "api_url": "https://api.semanticscholar.org/graph/v1",
        "queries": ["keyword query"],
        "reference_date": "2024-02-01",
        "regular_delay_seconds": 0,
        "jitter_seconds": 0,
        "backoff_seconds": 0,
    }
    values.update(overrides)
    return SemanticScholarNewestPaperCollector(**values)


def test_publication_window_uses_delayed_stride_and_optional_backfill():
    from mourat.collectors.semantic_scholar_newest import publication_window

    assert publication_window(
        reference_date=date(2024, 2, 1),
        lag_days=30,
        stride_days=1,
        backfill_days=0,
    ) == (date(2024, 1, 2), date(2024, 1, 2))
    assert publication_window(
        reference_date=date(2024, 2, 1),
        lag_days=10,
        stride_days=3,
        backfill_days=7,
    ) == (date(2024, 1, 13), date(2024, 1, 22))


@pytest.mark.parametrize(
    "overrides",
    [
        {"lag_days": -1},
        {"stride_days": 0},
        {"backfill_days": -1},
        {"max_results_per_query": 0},
        {"queries": []},
        {"queries": ["valid", " "]},
    ],
)
def test_collector_validates_window_query_and_pagination_settings(overrides):
    with pytest.raises(ValueError):
        _collector(MagicMock(), **overrides)


def test_bulk_search_maps_identity_and_sends_query_window_and_api_key():
    http = MagicMock()
    http.get.return_value = _response({"data": [_paper()], "token": None})
    handler = _Handler()
    collector = _collector(
        http,
        handler,
        api_key="secret-test-key",
        queries=["first", "second"],
    )
    http.get.side_effect = [
        _response({"data": [_paper()], "token": None}),
        _response(
            {
                "data": [
                    _paper(
                        paper_id="ss-2",
                        title="Second",
                        doi="HTTPS://DOI.ORG/10.5678/Second",
                    )
                ],
                "token": None,
            }
        ),
    ]

    result = collector({}, "1")

    assert len(result.papers) == 2
    first = result.papers[0]
    assert first.semantic_scholar_id == "ss-1"
    assert first.doi == "10.1234/journal"
    assert first.provenance == ["semantic_scholar"]
    assert first.publication_date == "2024-01-02"
    assert result.papers[1].doi == "10.5678/Second"
    assert http.get.call_count == 2
    for call in http.get.call_args_list:
        assert call.args[0].endswith("/paper/search/bulk")
        assert call.kwargs["headers"]["x-api-key"] == "secret-test-key"
        assert call.kwargs["params"]["publicationDateOrYear"] == (
            "2024-01-02:2024-01-02"
        )
        assert "externalIds" in call.kwargs["params"]["fields"]
        assert "paperId" in call.kwargs["params"]["fields"]
    assert http.get.call_args_list[0].kwargs["params"]["query"] == "first"
    assert http.get.call_args_list[1].kwargs["params"]["query"] == "second"
    assert "2024-01-02" in handler.calls[-1][1]


def test_bulk_search_follows_tokens_and_applies_local_cap():
    http = MagicMock()
    http.get.side_effect = [
        _response({"data": [_paper("ss-1")], "token": "next-token"}),
        _response(
            {
                "data": [_paper("ss-2", title="Second")],
                "token": "unused-third-page",
            }
        ),
    ]
    handler = _Handler()
    result = _collector(http, handler, max_results_per_query=2)({}, "1")

    assert [paper.semantic_scholar_id for paper in result.papers] == ["ss-1", "ss-2"]
    assert http.get.call_args_list[1].kwargs["params"]["token"] == "next-token"
    assert "cap" in handler.calls[-1][1].lower()

    http.reset_mock()
    http.get.side_effect = [
        _response({"data": [_paper("ss-1")], "token": "next-token"}),
        _response({"data": [_paper("ss-2")], "token": None}),
    ]
    result = _collector(http, max_results_per_query=1)({}, "1")
    assert len(result.papers) == 1
    assert http.get.call_count == 1


def test_uncapped_bulk_search_follows_tokens_until_the_result_set_ends():
    http = MagicMock()
    http.get.side_effect = [
        _response({"data": [_paper("ss-1")], "token": "page-two"}),
        _response({"data": [_paper("ss-2")], "token": "page-three"}),
        _response({"data": [_paper("ss-3")], "token": None}),
    ]
    result = _collector(http, max_results_per_query=None)({}, "1")

    assert [paper.semantic_scholar_id for paper in result.papers] == [
        "ss-1",
        "ss-2",
        "ss-3",
    ]
    assert http.get.call_count == 3
    assert http.get.call_args_list[1].kwargs["params"]["token"] == "page-two"
    assert http.get.call_args_list[2].kwargs["params"]["token"] == "page-three"


def test_bulk_search_excludes_missing_and_out_of_window_publication_dates():
    http = MagicMock()
    http.get.return_value = _response(
        {
            "data": [
                _paper("missing-date", publication_date=None),
                _paper("outside-date", publication_date="2024-01-01"),
                _paper("inside-date", publication_date="2024-01-02"),
            ],
            "token": None,
        }
    )
    result = _collector(http)({}, "1")
    assert [paper.semantic_scholar_id for paper in result.papers] == ["inside-date"]


def test_arxiv_doi_is_canonicalized_to_arxiv_identity():
    http = MagicMock()
    http.get.return_value = _response(
        {
            "data": [
                _paper(
                    doi="10.48550/arXiv.2401.00001",
                    arxiv_id="2401.00001v2",
                )
            ],
            "token": None,
        }
    )
    result = _collector(http)({}, "1")
    assert result.papers[0].arxiv_id == "2401.00001"
    assert result.papers[0].doi is None


def test_bounded_retry_recovers_and_exhaustion_raises(monkeypatch):
    from mourat.collectors import semantic_scholar_newest

    sleeps = []
    monkeypatch.setattr(semantic_scholar_newest.time, "sleep", sleeps.append)
    http = MagicMock()
    http.get.side_effect = [
        _response({"message": "limited"}, 429, {"Retry-After": "3"}),
        _response({"data": [_paper()], "token": None}),
    ]
    result = _collector(http, max_retries=1)({}, "1")
    assert len(result.papers) == 1
    assert sleeps == [3.0]

    http.reset_mock()
    http.get.side_effect = [
        _response({"message": "down"}, 500),
        _response({"message": "down"}, 500),
    ]
    with pytest.raises(httpx.HTTPStatusError):
        _collector(http, max_retries=1)({}, "1")
    assert http.get.call_count == 2


def test_transport_errors_use_the_same_bounded_retry_policy(monkeypatch):
    from mourat.collectors import semantic_scholar_newest

    sleeps = []
    monkeypatch.setattr(semantic_scholar_newest.time, "sleep", sleeps.append)
    request = httpx.Request("GET", "https://example.com")
    http = MagicMock()
    http.get.side_effect = [
        httpx.ConnectError("connection failed", request=request),
        _response({"data": [_paper()], "token": None}),
    ]
    result = _collector(http, max_retries=1, backoff_seconds=1)({}, "1")

    assert len(result.papers) == 1
    assert sleeps == [1.0]

    http.reset_mock()
    http.get.side_effect = httpx.TimeoutException("timeout", request=request)
    with pytest.raises(httpx.TimeoutException):
        _collector(http, max_retries=1, backoff_seconds=0)({}, "1")
    assert http.get.call_count == 2


def _candidate(
    title,
    arxiv_id=None,
    doi=None,
    provider_id=None,
    provenance=None,
):
    values = {
        "title": title,
        "authors": ["Author"],
        "description": f"Description of {title}",
        "urls_seen": [f"https://example.com/{title}"],
        "arxiv_id": arxiv_id,
        "doi": doi,
        "provenance": provenance or [],
    }
    if provider_id is not None:
        values["semantic_scholar_id"] = provider_id
    return PaperCandidate(**values)


def _merge(arxiv=(), semantic_scholar=(), seen=()):
    from mourat.processors.newest_paper_merger import (
        merge_newest_paper_candidates,
    )

    return merge_newest_paper_candidates(
        PaperCandidateCollection(papers=list(arxiv)),
        PaperCandidateCollection(papers=list(semantic_scholar)),
        seen_provider_ids=seen,
    )


def test_merge_prefers_arxiv_metadata_and_enriches_doi_and_provenance():
    arxiv = _candidate(
        "arXiv primary",
        arxiv_id="2401.00001",
        provenance=["arxiv"],
    )
    semantic = _candidate(
        "Semantic Scholar secondary",
        arxiv_id="2401.00001v3",
        doi="10.1234/same",
        provider_id="ss-1",
        provenance=["semantic_scholar"],
    )
    result = _merge([arxiv], [semantic])

    assert len(result.collection.papers) == 1
    merged = result.collection.papers[0]
    assert merged.title == "arXiv primary"
    assert merged.description == "Description of arXiv primary"
    assert merged.doi == "10.1234/same"
    assert merged.provenance == ["arxiv", "semantic_scholar"]
    assert result.merged_cross_source == 1
    assert result.new_semantic_scholar_ids == ["ss-1"]


def test_merge_matches_by_doi_and_deduplicates_semantic_scholar_records():
    arxiv = _candidate(
        "arXiv primary",
        arxiv_id="2401.00001",
        doi="10.1234/SAME",
        provenance=["arxiv"],
    )
    first = _candidate(
        "S2 one",
        doi="https://doi.org/10.1234/same",
        provider_id="ss-1",
        provenance=["semantic_scholar"],
    )
    second = _candidate(
        "S2 two",
        doi="10.1234/same",
        provider_id="ss-2",
        provenance=["semantic_scholar"],
    )
    result = _merge([arxiv], [first, second])
    assert len(result.collection.papers) == 1
    assert result.collection.papers[0].title == "arXiv primary"
    assert result.new_semantic_scholar_ids == ["ss-1", "ss-2"]


def test_semantic_scholar_only_requires_provider_id_and_non_arxiv_doi():
    result = _merge(
        semantic_scholar=[
            _candidate("Accepted", doi="10.1234/accepted", provider_id="ss-1"),
            _candidate("No DOI", provider_id="ss-2"),
            _candidate(
                "Only arXiv DOI",
                arxiv_id="2401.00002",
                doi="10.48550/arXiv.2401.00002",
                provider_id="ss-3",
            ),
            _candidate("No provider ID", doi="10.1234/no-provider"),
        ]
    )
    assert [paper.title for paper in result.collection.papers] == ["Accepted"]
    assert result.missing_doi_excluded == 2
    assert result.missing_provider_id_excluded == 1
    assert result.new_semantic_scholar_ids == ["ss-1", "ss-2", "ss-3"]


def test_semantic_scholar_without_doi_can_match_arxiv_candidate():
    result = _merge(
        arxiv=[_candidate("arXiv", arxiv_id="2401.00001")],
        semantic_scholar=[
            _candidate(
                "S2 duplicate",
                arxiv_id="2401.00001",
                provider_id="ss-1",
            )
        ],
    )
    assert len(result.collection.papers) == 1
    assert result.missing_doi_excluded == 0


def test_ledger_ids_are_excluded_but_missing_doi_ids_are_recorded_after_success(
    tmp_path,
):
    from mourat.database.paper_identity import (
        list_semantic_scholar_ledger_ids,
        record_semantic_scholar_processed,
    )

    conn = init_db(tmp_path / "ledger.db")
    record_semantic_scholar_processed(conn, ["seen"])
    result = _merge(
        semantic_scholar=[
            _candidate("Seen", doi="10.1234/seen", provider_id="seen"),
            _candidate("New without DOI", provider_id="new"),
        ],
        seen=list_semantic_scholar_ledger_ids(conn),
    )
    assert result.ledger_excluded == 1
    assert result.collection.papers == []
    assert result.new_semantic_scholar_ids == ["new"]

    record_semantic_scholar_processed(conn, result.new_semantic_scholar_ids)
    assert list_semantic_scholar_ledger_ids(conn) == {"seen", "new"}
    conn.close()


def test_ledger_batch_insert_is_atomic(tmp_path):
    from mourat.database.paper_identity import (
        list_semantic_scholar_ledger_ids,
        record_semantic_scholar_processed,
    )

    conn = init_db(tmp_path / "ledger.db")
    original_execute = conn.execute

    class FailingConnection:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        def execute(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise sqlite3.DatabaseError("injected")
            return original_execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.wrapped.__exit__(exc_type, exc_value, traceback)

    with pytest.raises(sqlite3.DatabaseError):
        record_semantic_scholar_processed(FailingConnection(conn), ["one", "two"])
    assert list_semantic_scholar_ledger_ids(conn) == set()
    conn.close()
