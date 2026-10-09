"""Tests for the normalized post influence score (spec 17)."""

import json

import pytest
from omegaconf import OmegaConf

from mourat.base import Function
from mourat.data_models import (
    RedditPostCollection,
    RedditPostInfo,
)
from mourat.monitoring import MonitoringHandler
from mourat.processors.post_influence import (
    PostInfluenceAssessor,
    PostInfluenceFilter,
    compute_influence_score,
)


class _Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _post(score, subreddit="ml", submission_id=None):
    return RedditPostInfo(
        subreddit=subreddit,
        submission_id=submission_id or f"id{score}",
        title=f"Post {score}",
        author="researcher",
        date="2026-10-01T12:00:00",
        url=f"https://reddit.com/r/{subreddit}/{submission_id or score}",
        text="Body",
        score=score,
    )


# --- Part A: calibration percentiles (T1-T4 live in
# tests/test_calibrate_subreddit_stats.py; T4 checks the config here too). ---


def test_calibration_config_has_percentiles_default():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config_calibrate_subreddit_stats")
    assert cfg.percentiles == []  # T4 (R1)


# --- Part B: assessor ---


def test_compute_influence_score_anchor_values():
    # T5 (R4): at reference -> 50, 0 -> 0, 9x -> 90, ratio 1/3 -> 25.
    assert compute_influence_score(10, 10) == 50
    assert compute_influence_score(0, 10) == 0
    assert compute_influence_score(90, 10) == 90
    assert compute_influence_score(1, 3) == 25  # 100 * (1/3)/(4/3) = 25.0


def test_compute_influence_score_reference_below_one():
    # T6 (R4): max(reference, 1) guard.
    assert compute_influence_score(1, 0.5) == 50


def test_assessor_scores_each_post():
    # T5 (R4): collection-level behavior; other fields untouched.
    handler = _Handler()
    assessor = PostInfluenceAssessor(handler, references={"ml": 10})
    posts = RedditPostCollection(posts=[_post(10), _post(0), _post(90)])

    output = assessor(posts, step_id="1.5")

    assert isinstance(output, RedditPostCollection)
    assert [p.influence_score for p in output.posts] == [50, 0, 90]
    assert [p.score for p in output.posts] == [10, 0, 90]
    assert [p.title for p in output.posts] == ["Post 10", "Post 0", "Post 90"]


def test_assessor_is_function():
    handler = _Handler()
    assessor = PostInfluenceAssessor(handler, references={"ml": 10})
    assert isinstance(assessor, Function)


def test_assessor_unknown_subreddit_raises_with_name():
    # T7 (R5, B3): loud failure naming the subreddit.
    handler = _Handler()
    assessor = PostInfluenceAssessor(handler, references={"ml": 10})
    posts = RedditPostCollection(posts=[_post(5, subreddit="unknownsub")])

    with pytest.raises(Exception, match="unknownsub"):
        assessor(posts, step_id="1.5")


# --- Part B: filter ---


def test_filter_drops_below_threshold():
    # T8 (R4): min_influence=50; 49 dropped, 50 and 51 survive.
    handler = _Handler()
    filt = PostInfluenceFilter(handler, min_influence=50)
    posts = [_post(49), _post(50), _post(51)]
    for p, infl in zip(posts, [49, 50, 51]):
        p.influence_score = infl
    collection = RedditPostCollection(posts=posts)

    output = filt(collection, step_id="1.6")

    assert [p.influence_score for p in output.posts] == [50, 51]
    assert "filtered: 1" in handler.calls[-1][1]


def test_filter_is_function():
    handler = _Handler()
    filt = PostInfluenceFilter(handler, min_influence=50)
    assert isinstance(filt, Function)


# --- Part B: score travel (T9) ---


def test_scorer_binding_carries_influence_score():
    # T9 (R6): influence_score travels from RedditPostInfo to ScoredRedditPost.
    from mourat.data_models import ScoredRedditPost

    post = _post(73)
    post.influence_score = 55
    scored = ScoredRedditPost(
        post=post,
        additional_context=[],
        relevance_scores=[],
        filtering_score=0.0,
    )
    assert scored.post.influence_score == 55


# --- Part B: config ---


def test_collect_posts_config_composes_with_post_influence():
    # T12 (R3, R8, B4): post_influence group with defaults.
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config_collect_posts")
    assert cfg.post_influence.references == {}
    assert cfg.post_influence.min_influence == 50
    assert cfg.post_influence.enabled is True


# --- Part B: writers (T10, T11) ---


@pytest.fixture
def db_conn(tmp_path):
    from mourat.database import init_db

    conn = init_db(str(tmp_path / "posts.db"))
    yield conn
    conn.close()


def _make_scored(score, influence_score):
    post = _post(score)
    post.influence_score = influence_score
    from mourat.data_models import ScoredRedditPost

    return ScoredRedditPost(
        post=post,
        additional_context=[],
        relevance_scores=[],
        filtering_score=0.0,
    )


def test_db_writer_stores_normalized_influence_when_present(db_conn):
    # T10 (R7, B2): normalized value stored, not the clip.
    from mourat.database import content_item as ci
    from mourat.data_models import ScoredRedditPostCollection
    from mourat.writers.post_writers import PostContentItemDbWriter

    writer = PostContentItemDbWriter(_Handler(), conn=db_conn)
    writer(
        ScoredRedditPostCollection(posts=[_make_scored(score=125, influence_score=73)]),
        step_id="8",
    )
    assert ci.get_content_item(db_conn, "reddit_id125")["influence_score"] == 73


def test_db_writer_falls_back_to_clip_without_influence(db_conn):
    # T11 (R7, B1): influence_score None -> legacy min(100, score).
    from mourat.database import content_item as ci
    from mourat.data_models import ScoredRedditPostCollection
    from mourat.writers.post_writers import PostContentItemDbWriter

    writer = PostContentItemDbWriter(_Handler(), conn=db_conn)
    writer(
        ScoredRedditPostCollection(
            posts=[_make_scored(score=125, influence_score=None)]
        ),
        step_id="8",
    )
    assert ci.get_content_item(db_conn, "reddit_id125")["influence_score"] == 100


def test_jsonl_writer_includes_influence_score(tmp_path):
    # T10 (R7, B2).
    from mourat.data_models import ScoredRedditPostCollection
    from mourat.writers.post_writers import PostJsonlWriter

    path = tmp_path / "posts.jsonl"
    writer = PostJsonlWriter(_Handler(), output_path=str(path))
    writer(
        ScoredRedditPostCollection(posts=[_make_scored(score=60, influence_score=42)]),
        step_id="8",
    )
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["influence_score"] == 42
    assert record["score"] == 60


# --- Part B: assessor -> filter integration (T13) ---


def test_assessor_then_filter_admits_posts_at_or_above_reference():
    # T13 (R4, B2): enabled flow keeps exactly the at-or-above-reference posts.
    handler = _Handler()
    assessor = PostInfluenceAssessor(handler, references={"ml": 10, "ai": 20})
    filt = PostInfluenceFilter(handler, min_influence=50)
    posts = RedditPostCollection(
        posts=[
            _post(10),
            _post(9),
            _post(20, subreddit="ai"),
            _post(19, subreddit="ai"),
        ]
    )
    kept = filt(assessor(posts, step_id="1.5"), step_id="1.6")
    assert [p.submission_id for p in kept.posts] == ["id10", "id20"]
