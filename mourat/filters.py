import re
import logging

from mourat.base import Function
from mourat.data_models import (
    ClassifiedRedditPostCollection,
    RedditPostCollection,
    RedditPostInfo,
    ScoredPaperInfoCollection,
    ScoredRedditPostCollection,
    ScoreEntry,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_text_description

logger = logging.getLogger(__name__)


class ScoreBasedPaperFilter(
    Function[ScoredPaperInfoCollection, ScoredPaperInfoCollection]
):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        score_threshold: int,
        text_for_monitoring_template: str,
    ) -> None:
        self.score_threshold = score_threshold
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(
        self, data: ScoredPaperInfoCollection
    ) -> tuple[ScoredPaperInfoCollection, str]:
        output = data
        text_for_monitoring = ""

        for p in output.papers[:]:
            if p.score < self.score_threshold:
                output.papers.remove(p)
            else:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p,
                )

        return output, text_for_monitoring


class PostScoreFilter(Function[ScoredRedditPostCollection, ScoredRedditPostCollection]):
    """Filters scored Reddit posts by max_score threshold."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        score_threshold: float,
        text_for_monitoring_template: str = (
            "{count} posts passed score threshold ({threshold}+)"
        ),
    ) -> None:
        self.score_threshold = score_threshold
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(
        self, data: ScoredRedditPostCollection
    ) -> tuple[ScoredRedditPostCollection, str]:
        output = ScoredRedditPostCollection(posts=[])
        text_for_monitoring = ""

        for p in data.posts:
            if p.max_score >= self.score_threshold:
                p.relevance_scores: list[ScoreEntry] = [
                    se for se in p.relevance_scores if se.score >= self.score_threshold
                ]
                output.posts.append(p)
                text_for_monitoring += (
                    f"### {p.post.title}\n"
                    f"URL: {p.post.url}\n"
                    f"Max score: {p.max_score}\n\n"
                )

        text_for_monitoring = (
            self.text_for_monitoring_template.format(
                count=len(output.posts),
                threshold=self.score_threshold,
            )
            + "\n\n"
            + text_for_monitoring
        )

        return output, text_for_monitoring


class HeuristicSlopFilter(Function[RedditPostCollection, RedditPostCollection]):
    """Drops posts that fail deterministic slop heuristics (no LLM, no network).

    Each rule is individually configurable; a threshold of None or an empty
    denylist disables the corresponding rule.
    """

    # Rule evaluation order; first match drops the post.
    RULE_NAMES = (
        "min_text_chars",
        "deleted_markers",
        "drop_link_only",
        "min_score",
        "author_denylist",
        "author_regex_denylist",
        "title_regex_denylist",
    )

    _BARE_URL_RE = re.compile(r"^\s*(https?://\S+\s*)+$", re.IGNORECASE)

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        min_text_chars: int | None = None,
        deleted_markers: list[str] | None = None,
        drop_link_only: bool = False,
        min_score: int | None = None,
        author_denylist: list[str] | None = None,
        author_regex_denylist: list[str] | None = None,
        title_regex_denylist: list[str] | None = None,
    ) -> None:
        self.min_text_chars = min_text_chars
        self.deleted_markers = [m.lower() for m in (deleted_markers or [])]
        self.drop_link_only = drop_link_only
        self.min_score = min_score
        self.author_denylist = set(author_denylist or [])
        self.author_regex_denylist = [
            re.compile(r) for r in (author_regex_denylist or [])
        ]
        self.title_regex_denylist = [
            re.compile(r) for r in (title_regex_denylist or [])
        ]
        super().__init__(monitoring_handler)

    def _check(self, post: RedditPostInfo) -> str | None:
        """Return the name of the first matching rule, or None if the post is kept."""
        text = post.text.strip().lower()
        if self.min_text_chars is not None and len(text) < self.min_text_chars:
            return "min_text_chars"
        if self.deleted_markers and text in self.deleted_markers:
            return "deleted_markers"
        if self.drop_link_only and self._BARE_URL_RE.match(post.text.strip()):
            return "drop_link_only"
        if self.min_score is not None and post.score < self.min_score:
            return "min_score"
        if post.author in self.author_denylist:
            return "author_denylist"
        for regex in self.author_regex_denylist:
            if regex.match(post.author):
                return "author_regex_denylist"
        for regex in self.title_regex_denylist:
            if regex.match(post.title):
                return "title_regex_denylist"
        return None

    def _run(self, data: RedditPostCollection) -> tuple[RedditPostCollection, str]:
        kept: list[RedditPostInfo] = []
        dropped: list[tuple[RedditPostInfo, str]] = []

        for post in data.posts:
            rule = self._check(post)
            if rule is None:
                kept.append(post)
            else:
                dropped.append((post, rule))

        total = len(data.posts)
        dropped_count = len(dropped)
        kept_count = len(kept)
        pct = (100.0 * dropped_count / total) if total else 0.0
        lines = [
            f"Posts in: {total}, kept: {kept_count}, dropped: {dropped_count} "
            f"({pct:.1f}% dropped)"
        ]

        for post, rule in dropped:
            lines.append(
                f"### {post.title}\nURL: {post.url}\nDropped by rule: {rule}\n"
            )

        return RedditPostCollection(posts=kept), "\n".join(lines)


class SlopFilter(Function[ClassifiedRedditPostCollection, RedditPostCollection]):
    """Keeps only posts classified as not slop and unwraps them to plain posts."""

    def _run(
        self, data: ClassifiedRedditPostCollection
    ) -> tuple[RedditPostCollection, str]:
        kept: list[RedditPostInfo] = []
        dropped = []

        for cp in data.posts:
            if cp.is_slop:
                dropped.append(cp)
            else:
                kept.append(cp.post)

        total = len(data.posts)
        dropped_count = len(dropped)
        pct = (100.0 * dropped_count / total) if total else 0.0
        lines = [
            f"Posts in: {total}, kept: {len(kept)}, dropped: {dropped_count} "
            f"({pct:.1f}% dropped)"
        ]

        for cp in dropped:
            lines.append(
                f"### {cp.post.title}\nURL: {cp.post.url}\n"
                f"Dropped by slop filter: {cp.justification}\n"
            )

        return RedditPostCollection(posts=kept), "\n".join(lines)
