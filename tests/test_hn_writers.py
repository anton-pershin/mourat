"""Tests for HN scorer binding and writers (spec 19)."""

import json

import pytest

from mourat.data_models import (
    HNPostCollection,
    HNPostInfo,
    ScoredHNPostCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.writers.hn_post_writers import (
    HNContentItemDbWriter,
    HNPostJsonlWriter,
)


@pytest.fixture()
def db_conn(tmp_path):
    from mourat.database import init_db

    conn = init_db(str(tmp_path / "hn.db"))
    yield conn
    conn.close()


class _Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _post(**overrides):
    base = {
        "hn_id": "100",
        "title": "Test post",
        "author": "bob",
        "date": "2026-10-15T00:00:00+00:00",
        "url": "https://example.com/x",
        "text": "",
        "points": 50,
        "num_comments": 3,
        "influence_score": 73,
        "description": "A compact summary.",
    }
    base.update(overrides)
    return HNPostInfo(**base)


def _scored(post, filtering_score=80.0):
    from mourat.data_models import ScoreEntry, ScoredHNPost

    return ScoredHNPost(
        post=post,
        relevance_scores=[
            ScoreEntry(id="1", type="rq", score=90, justification="j"),
        ],
        filtering_score=filtering_score,
    )


class TestHNContentItemDbWriter:
    def test_writes_post_with_hn_metadata(self, db_conn):
        # T17 (R9): reference rows, description = summary, influence_score as computed.
        handler = _Handler()
        writer = HNContentItemDbWriter(handler, conn=db_conn)
        writer(ScoredHNPostCollection(posts=[_scored(_post())]), step_id="6")
        row = db_conn.execute(
            "SELECT id, platform_id, influence_metric_id, description, influence_score "
            "FROM content_items WHERE id = 'hn_100'"
        ).fetchone()
        assert row is not None
        assert row["platform_id"] == "hackernews"
        assert row["influence_metric_id"] == "hn_points"
        assert row["description"] == "A compact summary."
        assert row["influence_score"] == 73

    def test_relevance_links_use_database_api_arguments(self, db_conn, monkeypatch):
        from mourat.data_models import ScoreEntry
        from mourat.database import content_item as ci

        captured = []

        def link(conn, content_id, target_id, justification="", relevance_score=None):
            captured.append((content_id, target_id, justification, relevance_score))

        monkeypatch.setattr(ci, "add_item_research_question", link)
        writer = HNContentItemDbWriter(_Handler(), conn=db_conn)
        scored = _scored(_post())
        scored.relevance_scores = [
            ScoreEntry(
                id="rq-1", type="rq", score=90, justification="Relevant research"
            )
        ]

        writer(ScoredHNPostCollection(posts=[scored]), step_id="6")

        assert captured == [("hn_100", "rq-1", "Relevant research", 90)]

    def test_dedup_skips_existing(self, db_conn):
        # T17 (R9): second write of the same hn_id is skipped, not duplicated.
        handler = _Handler()
        writer = HNContentItemDbWriter(handler, conn=db_conn)
        writer(ScoredHNPostCollection(posts=[_scored(_post())]), step_id="6")
        result = writer(ScoredHNPostCollection(posts=[_scored(_post())]), step_id="6")
        count = db_conn.execute(
            "SELECT COUNT(*) AS c FROM content_items WHERE id = 'hn_100'"
        ).fetchone()["c"]
        assert count == 1

    def test_fallback_clip_when_influence_none(self, db_conn):
        # T17 (R9, B1): influence_score None -> min(100, points).
        handler = _Handler()
        writer = HNContentItemDbWriter(handler, conn=db_conn)
        post = _post(hn_id="200", points=150, influence_score=None)
        writer(ScoredHNPostCollection(posts=[_scored(post)]), step_id="6")
        row = db_conn.execute(
            "SELECT influence_score FROM content_items WHERE id = 'hn_200'"
        ).fetchone()
        assert row["influence_score"] == 100


class TestHNPostJsonlWriter:
    def test_record_fields(self, tmp_path):
        # T18 (R9): record contains hn_id, points, influence_score, description.
        handler = _Handler()
        out = tmp_path / "hn_posts.jsonl"
        writer = HNPostJsonlWriter(handler, output_path=str(out))
        writer(ScoredHNPostCollection(posts=[_scored(_post())]), step_id="6")
        record = json.loads(out.read_text().splitlines()[0])
        assert record["hn_id"] == "100"
        assert record["points"] == 50
        assert record["influence_score"] == 73
        assert record["description"] == "A compact summary."
        assert record["filtering_score"] == 80.0


class TestHNContentItemScorer:
    def test_summary_only_and_influence_travel(self):
        # T16 (R8): item text is title + description; influence_score carried.
        from mourat.processors.content_item_scorer import HNContentItemScorer
        from unittest.mock import MagicMock

        handler = _Handler()
        scorer = HNContentItemScorer.__new__(HNContentItemScorer)
        core = MagicMock()
        core.scored_item_indices = [0]
        core.scored_items = [
            type(
                "S",
                (),
                {
                    "relevance_scores": [],
                    "filtering_score": 66.0,
                },
            )()
        ]
        captured = {}

        def fake_run(neutral):
            captured["items"] = neutral.items
            return neutral, "scored"

        core._run = fake_run
        scorer.core = core
        scorer.monitoring_handler = handler

        post = _post(title="My title", description="My summary.", influence_score=None)
        post.influence_score = 73
        out = scorer(HNPostCollection(posts=[post]), step_id="4")
        assert captured["items"][0].title == "My title"
        assert captured["items"][0].body_text == "My title\n\nMy summary."
        assert out.posts[0].post is post
        assert post.influence_score == 73
        assert out.posts[0].filtering_score == 66.0
