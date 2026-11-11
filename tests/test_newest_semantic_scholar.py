"""Tests for Semantic Scholar support in newest-paper collection (spec 28)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from mourat.collectors.semantic_scholar_newest import (
    SemanticScholarNewestPaperCollector,
    plan_query_windows,
)
from mourat.data_models import (
    PaperCandidate,
    PaperCandidateCollection,
    ResolvedPaper,
    ScoredPaper,
    ScoredPaperCollection,
)
from mourat.database import content_item as ci
from mourat.database import init_db
from mourat.database import semantic_scholar as s2db
from mourat.monitoring import MonitoringHandler
from mourat.processors.candidate_converter import CandidateToResolvedConverter
from mourat.processors.paper_deduplicator import PaperCandidateDeduplicator
from mourat.writers.db_writer import ContentItemDbWriter, _content_item_id
from mourat.writers.jsonl_writer import JsonlWriter


class _Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


@pytest.fixture()
def db_conn():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = init_db(path)
    yield conn
    conn.close()
    os.unlink(path)


def _response(payload, status_code=200, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.text = json.dumps(payload)
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _paper_payload(paper_id="s2-1", **overrides):
    payload = {
        "paperId": paper_id,
        "title": "Semantic Scholar paper",
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "abstract": "An abstract",
        "publicationDate": "2026-01-30",
        "authors": [{"name": "A. Researcher"}],
        "externalIds": {"DOI": "10.1000/example"},
    }
    payload.update(overrides)
    return payload


def _collector(db_conn, http_client, **overrides):
    values = {
        "monitoring_handler": _Handler(),
        "http_client": http_client,
        "conn": db_conn,
        "queries": ["machine learning"],
        "api_url": "https://api.semanticscholar.org/graph/v1",
        "api_key": "secret-key",
        "publication_lag_days": 1,
        "stride_days": 1,
        "backfill_days": 0,
        "run_date": date(2026, 1, 31),
        "regular_delay_seconds": 0,
        "backoff_seconds": 0,
    }
    values.update(overrides)
    return SemanticScholarNewestPaperCollector(**values)


def _candidate(
    title,
    *,
    doi=None,
    arxiv_id=None,
    provenance,
    url="https://example.com/paper",
):
    return PaperCandidate(
        title=title,
        authors=["Author"],
        description=f"Abstract for {title}",
        urls_seen=[url] if url else [],
        doi=doi,
        arxiv_id=arxiv_id,
        publication_date="2026-01-30",
        provenance=provenance,
    )


def _scored(paper):
    return ScoredPaper(paper=paper, relevance_scores=[], filtering_score=0)


class TestSourceSelection:
    @pytest.mark.parametrize(
        ("sources", "expected"),
        [
            (["arxiv", "semantic_scholar"], ["arxiv", "semantic_scholar"]),
            (["arxiv"], ["arxiv"]),
            (["semantic_scholar"], ["semantic_scholar"]),
        ],
    )
    def test_valid_source_modes(self, sources, expected):
        from mourat.scripts.collect_newest_papers import _validate_sources

        cfg = OmegaConf.create({"sources": sources})
        assert _validate_sources(cfg) == expected

    @pytest.mark.parametrize("sources", [[], ["arxiv", "arxiv"], ["unknown"]])
    def test_invalid_source_modes(self, sources):
        from mourat.scripts.collect_newest_papers import _validate_sources

        cfg = OmegaConf.create({"sources": sources})
        with pytest.raises(ValueError):
            _validate_sources(cfg)

    @pytest.mark.parametrize("queries", [[], [""], ["duplicate", "duplicate"]])
    def test_semantic_scholar_requires_valid_unique_queries(self, db_conn, queries):
        with pytest.raises(ValueError):
            _collector(db_conn, MagicMock(), queries=queries)


class TestPublicationWindowPlanning:
    def test_new_query_uses_delayed_stride(self):
        plan = plan_query_windows(
            "q",
            state=None,
            today=date(2026, 1, 31),
            publication_lag_days=1,
            stride_days=1,
            backfill_days=0,
            backfill_interval_days=7,
        )
        assert plan.windows == [(date(2026, 1, 30), date(2026, 1, 30))]
        assert plan.completed_end_date == date(2026, 1, 30)

    def test_cursor_catches_up_each_missed_date(self):
        plan = plan_query_windows(
            "q",
            state={
                "last_completed_end_date": "2026-01-29",
                "last_backfill_completed_on": "2026-01-31",
            },
            today=date(2026, 2, 2),
            publication_lag_days=1,
            stride_days=1,
            backfill_days=30,
            backfill_interval_days=7,
        )
        assert plan.windows == [
            (date(2026, 1, 30), date(2026, 1, 30)),
            (date(2026, 1, 31), date(2026, 1, 31)),
            (date(2026, 2, 1), date(2026, 2, 1)),
        ]

    def test_first_run_backfill_extends_range(self):
        plan = plan_query_windows(
            "q",
            state=None,
            today=date(2026, 1, 31),
            publication_lag_days=1,
            stride_days=2,
            backfill_days=7,
            backfill_interval_days=7,
        )
        assert plan.windows[0][0] == date(2026, 1, 24)
        assert plan.windows[-1][1] == date(2026, 1, 30)
        assert all((end - start).days < 2 for start, end in plan.windows)
        assert plan.backfill_completed_on == date(2026, 1, 31)

    def test_periodic_backfill_runs_without_new_normal_date(self):
        plan = plan_query_windows(
            "q",
            state={
                "last_completed_end_date": "2026-01-30",
                "last_backfill_completed_on": "2026-01-24",
            },
            today=date(2026, 1, 31),
            publication_lag_days=1,
            stride_days=1,
            backfill_days=7,
            backfill_interval_days=7,
        )
        assert plan.windows[0] == (date(2026, 1, 24), date(2026, 1, 24))
        assert plan.windows[-1] == (date(2026, 1, 30), date(2026, 1, 30))

    def test_current_cursor_and_no_due_backfill_makes_no_request(self):
        plan = plan_query_windows(
            "q",
            state={
                "last_completed_end_date": "2026-01-30",
                "last_backfill_completed_on": "2026-01-31",
            },
            today=date(2026, 1, 31),
            publication_lag_days=1,
            stride_days=1,
            backfill_days=7,
            backfill_interval_days=7,
        )
        assert plan.windows == []
        assert plan.completed_end_date is None


class TestSemanticScholarCollection:
    def test_bulk_search_paginates_and_maps_identifiers(self, db_conn):
        http = MagicMock()
        http.get.side_effect = [
            _response({"data": [_paper_payload()], "token": "next-token"}),
            _response(
                {
                    "data": [
                        _paper_payload(
                            "s2-2",
                            title="Second paper",
                            externalIds={"ArXiv": "1706.03762v2"},
                        )
                    ],
                    "token": None,
                }
            ),
        ]
        collector = _collector(db_conn, http)

        result = collector({}, "1")

        assert len(result.papers) == 2
        first = result.papers[0]
        assert first.doi == "10.1000/example"
        assert first.provenance == ["semantic_scholar"]
        assert first.publication_date == "2026-01-30"
        assert result.papers[1].arxiv_id == "1706.03762"
        assert result.papers[1].doi is None
        assert set(collector.receipt.paper_ids) == {"s2-1", "s2-2"}
        first_call = http.get.call_args_list[0]
        assert first_call.args[0].endswith("/paper/search/bulk")
        assert first_call.kwargs["headers"] == {"x-api-key": "secret-key"}
        params = first_call.kwargs["params"]
        assert params["query"] == "machine learning"
        assert params["publicationDateOrYear"] == "2026-01-30:2026-01-30"
        assert params["sort"] == "publicationDate:desc"
        assert "paperId" in params["fields"]
        assert "externalIds" in params["fields"]
        assert http.get.call_args_list[1].kwargs["params"]["token"] == "next-token"

    def test_result_limit_stops_pagination(self, db_conn):
        http = MagicMock()
        http.get.return_value = _response({"data": [_paper_payload()], "token": "more"})
        collector = _collector(db_conn, http, max_results_per_query=1)

        result = collector({}, "1")

        assert len(result.papers) == 1
        assert http.get.call_count == 1

    @patch("mourat.collectors.semantic_scholar_newest.time.sleep")
    def test_rate_limit_uses_retry_after(self, mock_sleep, db_conn):
        http = MagicMock()
        http.get.side_effect = [
            _response({"message": "slow down"}, 429, {"Retry-After": "3"}),
            _response({"data": [], "token": None}),
        ]
        collector = _collector(db_conn, http)

        collector({}, "1")

        assert http.get.call_count == 2
        mock_sleep.assert_called_once_with(3.0)

    def test_missing_abstract_is_retained(self, db_conn):
        http = MagicMock()
        http.get.return_value = _response(
            {"data": [_paper_payload(abstract=None)], "token": None}
        )
        result = _collector(db_conn, http)({}, "1")
        assert result.papers[0].description == ""


class TestSemanticScholarState:
    def test_commit_stores_seen_ids_and_query_state(self, db_conn):
        s2db.record_successful_collection(
            db_conn,
            paper_ids=["s2-1", "s2-2"],
            query_states=[
                {
                    "query": "q",
                    "completed_end_date": date(2026, 1, 30),
                    "backfill_completed_on": date(2026, 1, 31),
                }
            ],
        )

        assert s2db.get_seen_paper_ids(db_conn, ["s2-1", "other"]) == {"s2-1"}
        state = s2db.load_query_state(db_conn, "q")
        assert state["last_completed_end_date"] == "2026-01-30"
        assert state["last_backfill_completed_on"] == "2026-01-31"

    def test_seen_id_is_not_sent_to_candidates_from_another_query(self, db_conn):
        s2db.record_successful_collection(
            db_conn,
            paper_ids=["s2-1"],
            query_states=[],
        )
        http = MagicMock()
        http.get.return_value = _response({"data": [_paper_payload()], "token": None})
        collector = _collector(db_conn, http, queries=["another query"])

        result = collector({}, "1")

        assert result.papers == []
        assert collector.receipt.paper_ids == []


class TestCandidateDeduplication:
    def test_arxiv_record_wins_and_receives_doi_and_provenance(self):
        arxiv = _candidate(
            "arXiv primary",
            arxiv_id="2401.00001",
            provenance=["arxiv_rss"],
            url="https://arxiv.org/abs/2401.00001",
        )
        s2 = _candidate(
            "Semantic Scholar secondary",
            doi="HTTPS://DOI.ORG/10.1000/Example",
            arxiv_id="2401.00001v2",
            provenance=["semantic_scholar"],
        )
        dedup = PaperCandidateDeduplicator(_Handler())

        result = dedup(PaperCandidateCollection(papers=[s2, arxiv]), "2")

        assert len(result.papers) == 1
        paper = result.papers[0]
        assert paper.title == "arXiv primary"
        assert paper.description == "Abstract for arXiv primary"
        assert paper.doi == "10.1000/example"
        assert paper.provenance == ["arxiv_rss", "semantic_scholar"]

    def test_doi_match_deduplicates_and_s2_only_without_doi_is_excluded(self):
        first = _candidate(
            "DOI primary",
            doi="10.1000/shared",
            provenance=["semantic_scholar"],
        )
        duplicate = _candidate(
            "DOI duplicate",
            doi="https://doi.org/10.1000/SHARED",
            provenance=["semantic_scholar"],
        )
        no_doi = _candidate("No DOI", provenance=["semantic_scholar"])

        result = PaperCandidateDeduplicator(_Handler())(
            PaperCandidateCollection(papers=[first, duplicate, no_doi]), "2"
        )

        assert [paper.title for paper in result.papers] == ["DOI primary"]

    def test_arxiv_only_candidate_does_not_require_doi(self):
        arxiv = _candidate("arXiv", arxiv_id="2401.00001", provenance=["arxiv_rss"])
        result = PaperCandidateDeduplicator(_Handler())(
            PaperCandidateCollection(papers=[arxiv]), "2"
        )
        assert result.papers == [arxiv]

    def test_converter_preserves_doi(self):
        candidate = _candidate(
            "DOI paper", doi="10.1000/example", provenance=["semantic_scholar"]
        )
        result = CandidateToResolvedConverter(_Handler())(
            PaperCandidateCollection(papers=[candidate]), "5"
        )
        assert result.papers[0].doi == "10.1000/example"


class TestPersistentPaperIdentity:
    def test_doi_id_is_normalized_and_arxiv_has_precedence(self):
        assert _content_item_id("Title", doi="HTTPS://DOI.ORG/10.1000/ABC") == (
            "paper_doi-10.1000/abc"
        )
        assert _content_item_id("Title", "2401.00001v2", "10.1000/abc") == (
            "paper_arxiv-2401.00001"
        )

    def test_doi_row_is_rekeyed_when_arxiv_identity_arrives(self, db_conn):
        writer = ContentItemDbWriter(_Handler(), db_conn)
        doi = "https://doi.org/10.1000/shared"
        writer(
            ScoredPaperCollection(
                papers=[_scored(ResolvedPaper(title="S2 title", doi=doi))]
            ),
            "8",
        )
        assert ci.get_content_item(db_conn, "paper_doi-10.1000/shared") is not None

        writer(
            ScoredPaperCollection(
                papers=[
                    _scored(
                        ResolvedPaper(
                            title="arXiv primary",
                            doi="10.1000/SHARED",
                            arxiv_id="2401.00001v3",
                        )
                    )
                ]
            ),
            "8",
        )

        assert ci.get_content_item(db_conn, "paper_doi-10.1000/shared") is None
        item = ci.get_content_item(db_conn, "paper_arxiv-2401.00001")
        assert item is not None
        assert item["name"] == "arXiv primary"
        assert db_conn.execute("SELECT COUNT(*) FROM content_items").fetchone()[0] == 1
        aliases = db_conn.execute(
            "SELECT identity_type, identity_value FROM paper_identities "
            "WHERE content_item_id = ? ORDER BY identity_type",
            ("paper_arxiv-2401.00001",),
        ).fetchall()
        assert [tuple(row) for row in aliases] == [
            ("arxiv", "2401.00001"),
            ("doi", "10.1000/shared"),
        ]

    def test_jsonl_skip_existing_uses_doi_alias(self, db_conn, tmp_path):
        db_writer = ContentItemDbWriter(_Handler(), db_conn)
        db_writer(
            ScoredPaperCollection(
                papers=[_scored(ResolvedPaper(title="Stored", doi="10.1000/shared"))]
            ),
            "8",
        )
        output = tmp_path / "papers.jsonl"
        writer = JsonlWriter(_Handler(), str(output), skip_existing=True, conn=db_conn)

        result = writer(
            ScoredPaperCollection(
                papers=[
                    _scored(
                        ResolvedPaper(
                            title="Different title",
                            doi="https://doi.org/10.1000/SHARED",
                        )
                    )
                ]
            ),
            "8",
        )

        assert result.papers == []
        assert output.read_text(encoding="utf-8") == ""
