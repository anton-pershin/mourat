"""Tests for scored Reddit post writers (spec 15)."""

import json

import pytest

from mourat.data_models import (
    RedditPostInfo,
    ScoredRedditPost,
    ScoredRedditPostCollection,
    ScoreEntry,
)
from mourat.monitoring import MonitoringHandler
from mourat.writers.post_writers import PostContentItemDbWriter, PostJsonlWriter


class _Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _scored_post(**overrides):
    defaults = {
        "subreddit": "ml",
        "submission_id": "abc123",
        "title": "A useful research post",
        "author": "researcher",
        "date": "2026-10-01T12:00:00",
        "url": "https://reddit.com/r/ml/abc123",
        "text": "Post body",
        "score": 125,
    }
    post_fields = {**defaults, **overrides}
    post = RedditPostInfo(**post_fields)
    return ScoredRedditPost(
        post=post,
        additional_context=["A context point"],
        relevance_scores=[
            ScoreEntry(id="rq1", type="rq", score=87, justification="Directly relevant")
        ],
        filtering_score=87.0,
    )


@pytest.fixture
def db_conn(tmp_path):
    from mourat.database import init_db

    conn = init_db(str(tmp_path / "posts.db"))
    yield conn
    conn.close()


def test_db_writer_persists_reddit_metadata_capped_influence_and_links(db_conn):
    from mourat.database import content_item as ci
    from mourat.database import research_domain as rd

    rd.create_research_domain(db_conn, "rd1", "Domain")
    rd.create_research_direction(db_conn, "dir1", "Direction", "rd1")
    rd.create_research_object(db_conn, "obj1", "Object", "dir1")
    rd.create_research_question(db_conn, "rq1", "Question", "obj1", "Description")
    handler = _Handler()
    writer = PostContentItemDbWriter(handler, conn=db_conn)
    scored = ScoredRedditPostCollection(posts=[_scored_post()])

    output = writer(scored, step_id="8")

    assert output is scored
    item = ci.get_content_item(db_conn, "reddit_abc123")
    assert item["source_type_id"] == "post"
    assert item["platform_id"] == "reddit"
    assert item["influence_metric_id"] == "upvotes"
    assert item["name"] == "A useful research post"
    assert item["description"] == "Post body"
    assert item["authors"] == "researcher"
    assert item["influence_score"] == 100
    links = ci.list_item_research_questions(db_conn, "reddit_abc123")
    assert [
        (link["id"], link["relevance_score"], link["justification"]) for link in links
    ] == [("rq1", 87, "Directly relevant")]


def test_db_writer_skips_existing_post_without_updating_it(db_conn):
    from mourat.database import content_item as ci

    handler = _Handler()
    writer = PostContentItemDbWriter(handler, conn=db_conn)
    writer(ScoredRedditPostCollection(posts=[_scored_post()]), step_id="8")
    changed = _scored_post(title="Changed title", score=9)

    writer(ScoredRedditPostCollection(posts=[changed]), step_id="8")

    assert (
        ci.get_content_item(db_conn, "reddit_abc123")["name"]
        == "A useful research post"
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM content_items WHERE id = ?", ("reddit_abc123",)
        ).fetchone()[0]
        == 1
    )


def test_jsonl_writer_appends_post_score_context_and_relevance(tmp_path):
    path = tmp_path / "posts.jsonl"
    handler = _Handler()
    writer = PostJsonlWriter(handler, output_path=str(path))
    scored = _scored_post()

    output = writer(ScoredRedditPostCollection(posts=[scored]), step_id="8")

    assert output.posts == [scored]
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["submission_id"] == "abc123"
    assert record["score"] == 125
    assert "upvotes" not in record
    assert record["additional_context"] == ["A context point"]
    assert record["relevance_scores"][0]["justification"] == "Directly relevant"
    assert record["filtering_score"] == 87.0
