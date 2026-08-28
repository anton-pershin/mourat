"""Unit tests for filter modules."""

import datetime

from mourat.data_models import ScoredPaperInfo, ScoredPaperInfoCollection
from mourat.filters import ScoreBasedPaperFilter
from mourat.monitoring import MonitoringHandler


def _make_monitoring_handler():
    class DummyHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []
        def __call__(self, step: str, text_for_monitoring: str) -> None:
            self.calls.append((step, text_for_monitoring))
    return DummyHandler()


def _make_scored_papers():
    return ScoredPaperInfoCollection(papers=[
        ScoredPaperInfo(
            title="Paper A", link="http://a", abstract="abs A",
            citation_count=10, authors=["A"], publication_date=datetime.date(2024, 1, 1),
            score=5, justification="Great",
        ),
        ScoredPaperInfo(
            title="Paper B", link="http://b", abstract="abs B",
            citation_count=5, authors=["B"], publication_date=datetime.date(2024, 1, 2),
            score=3, justification="Okay",
        ),
        ScoredPaperInfo(
            title="Paper C", link="http://c", abstract="abs C",
            citation_count=1, authors=["C"], publication_date=datetime.date(2024, 1, 3),
            score=1, justification="Poor",
        ),
    ])


class TestScoreBasedPaperFilter:
    def test_filters_below_threshold(self):
        handler = _make_monitoring_handler()
        f = ScoreBasedPaperFilter(
            monitoring_handler=handler,
            score_threshold=3,
            text_for_monitoring_template="{title} {score}",
        )

        papers = _make_scored_papers()
        result = f(papers, "filter")

        assert len(result.papers) == 2
        titles = {p.title for p in result.papers}
        assert titles == {"Paper A", "Paper B"}

    def test_all_pass_when_threshold_zero(self):
        handler = _make_monitoring_handler()
        f = ScoreBasedPaperFilter(
            monitoring_handler=handler,
            score_threshold=0,
            text_for_monitoring_template="{title} {score}",
        )

        papers = _make_scored_papers()
        result = f(papers, "filter")

        assert len(result.papers) == 3

    def test_all_filtered_when_threshold_six(self):
        handler = _make_monitoring_handler()
        f = ScoreBasedPaperFilter(
            monitoring_handler=handler,
            score_threshold=6,
            text_for_monitoring_template="{title} {score}",
        )

        papers = _make_scored_papers()
        result = f(papers, "filter")

        assert len(result.papers) == 0


# --- Slop filtering (spec 04) ---

from mourat.data_models import (
    ClassifiedRedditPost,
    ClassifiedRedditPostCollection,
    RedditPostCollection,
    RedditPostInfo,
)
from mourat.filters import HeuristicSlopFilter, SlopFilter


def _make_post(
    submission_id: str,
    title: str = "A substantive title",
    text: str = "A long enough text body that passes the minimum length rule with ease.",
    author: str = "author",
    score: int = 10,
) -> RedditPostInfo:
    return RedditPostInfo(
        subreddit="testsub",
        submission_id=submission_id,
        title=title,
        author=author,
        date="2026-08-28T00:00:00+00:00",
        url=f"https://reddit.com/r/testsub/comments/{submission_id}",
        text=text,
        score=score,
    )


def _make_posts(*posts: RedditPostInfo) -> RedditPostCollection:
    return RedditPostCollection(posts=list(posts))


class TestHeuristicSlopFilter:
    def test_drops_short_text(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_text_chars=50)
        posts = _make_posts(_make_post("short", text="too short"), _make_post("long"))
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"long"}

    def test_drops_deleted_and_removed(self):
        f = HeuristicSlopFilter(
            _make_monitoring_handler(), deleted_markers=["[deleted]", "[removed]"]
        )
        posts = _make_posts(
            _make_post("del", text="[deleted]"),
            _make_post("rem", text="[removed]"),
            _make_post("ok"),
        )
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"ok"}

    def test_drops_link_only_text(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), drop_link_only=True)
        posts = _make_posts(
            _make_post("link", text="https://example.com/some-page"),
            _make_post("linkplus", text="https://example.com/x and here is my take on it"),
            _make_post("prose"),
        )
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"linkplus", "prose"}

    def test_drops_below_min_score(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_score=5)
        posts = _make_posts(_make_post("low", score=2), _make_post("high", score=10))
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"high"}

    def test_drops_denylisted_author_exact(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), author_denylist=["spammer"])
        posts = _make_posts(
            _make_post("a1", author="spammer"), _make_post("a2", author="gooduser")
        )
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"a2"}

    def test_drops_denylisted_author_regex(self):
        f = HeuristicSlopFilter(
            _make_monitoring_handler(), author_regex_denylist=[".*[Bb]ot$"]
        )
        posts = _make_posts(
            _make_post("b1", author="SpamBot"), _make_post("b2", author="human")
        )
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"b2"}

    def test_drops_denylisted_title_regex(self):
        f = HeuristicSlopFilter(
            _make_monitoring_handler(), title_regex_denylist=["^(ELI5|Weekly)"]
        )
        posts = _make_posts(
            _make_post("t1", title="Weekly discussion thread"),
            _make_post("t2", title="A normal question about retrieval"),
        )
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"t2"}

    def test_disabled_rule_never_drops(self):
        # min_text_chars omitted entirely (None) -> short post kept by this rule.
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_score=5)
        posts = _make_posts(_make_post("short", text="tiny"), _make_post("low", score=1))
        result = f(posts, "2")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"short"}

    def test_all_rules_disabled_is_passthrough(self):
        kwargs = {
            "min_text_chars": None,
            "deleted_markers": [],
            "drop_link_only": False,
            "min_score": None,
            "author_denylist": [],
            "author_regex_denylist": [],
            "title_regex_denylist": [],
        }
        f = HeuristicSlopFilter(_make_monitoring_handler(), **kwargs)
        posts = _make_posts(
            _make_post("a", text="[deleted]", author="bot", score=0, title="ELI5 x")
        )
        result = f(posts, "2")
        assert {p.submission_id for p in result.posts} == {"a"}

    def test_empty_collection(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_text_chars=10)
        result = f(RedditPostCollection(posts=[]), "2")
        assert len(result.posts) == 0

    def test_monitoring_reports_counts_and_dropped_posts(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_score=5)
        posts = _make_posts(_make_post("low", score=1), _make_post("high", score=10))
        f(posts, "2")

        step, text = f.monitoring_handler.calls[0]
        assert step == "2"
        assert "2" in text  # in count
        assert "min_score" in text  # rule name as drop reason
        assert posts.posts[0].url in text
        assert posts.posts[1].url not in text  # kept posts are not listed

    def test_returns_reddit_post_collection(self):
        f = HeuristicSlopFilter(_make_monitoring_handler(), min_text_chars=10)
        result = f(_make_posts(_make_post("a")), "2")
        assert isinstance(result, RedditPostCollection)


class TestSlopFilter:
    def _marked(self, *pairs) -> ClassifiedRedditPostCollection:
        return ClassifiedRedditPostCollection(
            posts=[
                ClassifiedRedditPost(post=post, is_slop=is_slop, justification=just)
                for post, is_slop, just in pairs
            ]
        )

    def test_keeps_only_non_slop(self):
        f = SlopFilter(_make_monitoring_handler())
        marked = self._marked(
            (_make_post("s1"), True, "meme"),
            (_make_post("s2"), False, "substantive"),
            (_make_post("s3"), True, "bot"),
        )
        result = f(marked, "4")
        ids = {p.submission_id for p in result.posts}
        assert ids == {"s2"}

    def test_output_is_reddit_post_collection(self):
        f = SlopFilter(_make_monitoring_handler())
        marked = self._marked((_make_post("s1"), False, "ok"))
        result = f(marked, "4")
        assert isinstance(result, RedditPostCollection)
        assert result.posts[0].title == "A substantive title"

    def test_all_slop_yields_empty(self):
        f = SlopFilter(_make_monitoring_handler())
        marked = self._marked((_make_post("s1"), True, "meme"))
        result = f(marked, "4")
        assert len(result.posts) == 0

    def test_none_slop_yields_all(self):
        f = SlopFilter(_make_monitoring_handler())
        marked = self._marked(
            (_make_post("s1"), False, "ok"), (_make_post("s2"), False, "ok")
        )
        result = f(marked, "4")
        assert {p.submission_id for p in result.posts} == {"s1", "s2"}

    def test_monitoring_reports_dropped_with_justification(self):
        f = SlopFilter(_make_monitoring_handler())
        marked = self._marked(
            (_make_post("s1"), True, "meme post"), (_make_post("s2"), False, "fine")
        )
        f(marked, "4")

        step, text = f.monitoring_handler.calls[0]
        assert step == "4"
        assert "meme post" in text
        assert marked.posts[0].post.url in text
        assert marked.posts[1].post.url not in text
