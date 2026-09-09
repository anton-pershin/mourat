import logging
import math
import re

from mourat.base import Function
from mourat.data_models import (
    ClassifiedRedditPostCollection,
    RedditPostCollection,
    RedditPostInfo,
    ResolvedPaper,
    ResolvedPaperCollection,
    ScoredPaper,
    ScoredPaperCollection,
    ScoredPaperInfoCollection,
    ScoredRedditPost,
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


class ScoreFilter(Function):
    """Drops scored items whose filtering_score is below the threshold.

    Item-agnostic core: domain bindings set `items_attr` and implement
    `_render_item` / `_make_output`. Sub-criterion entries whose score is
    below the threshold are removed from the retained items' relevance_scores
    (as PostScoreFilter did). Monitoring leads with the dropped items.
    """

    items_attr: str = "posts"

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        score_threshold: float,
    ) -> None:
        self.score_threshold = score_threshold
        super().__init__(monitoring_handler)

    def _render_item(self, item) -> str:
        raise NotImplementedError

    def _make_output(self, items):
        raise NotImplementedError

    def _run(self, data) -> tuple:
        kept = []
        dropped = []

        for item in getattr(data, self.items_attr):
            if item.filtering_score >= self.score_threshold:
                item.relevance_scores = [
                    se
                    for se in item.relevance_scores
                    if se.score >= self.score_threshold
                ]
                kept.append(item)
            else:
                dropped.append(item)

        total = len(kept) + len(dropped)
        pct = (100.0 * len(dropped) / total) if total else 0.0
        lines = [
            f"Items in: {total}, kept: {len(kept)}, dropped: {len(dropped)} "
            f"({pct:.1f}% dropped, threshold {self.score_threshold})"
        ]
        for item in dropped:
            lines.append(self._render_item(item))

        return self._make_output(kept), "\n".join(lines)


class PostScoreFilter(ScoreFilter):
    """Filters scored Reddit posts by filtering_score threshold."""

    items_attr = "posts"

    def _render_item(self, p: ScoredRedditPost) -> str:
        return (
            f"### {p.post.title}\n"
            f"URL: {p.post.url}\n"
            f"Filtering score: {p.filtering_score}\n\n"
        )

    def _make_output(self, items) -> ScoredRedditPostCollection:
        return ScoredRedditPostCollection(posts=items)


class PaperScoreFilter(ScoreFilter):
    """Filters scored papers by filtering_score threshold."""

    items_attr = "papers"

    def _render_item(self, p: ScoredPaper) -> str:
        url = p.paper.url or "(none)"
        return (
            f"### {p.paper.title}\n"
            f"URL: {url}\n"
            f"Filtering score: {p.filtering_score}\n\n"
        )

    def _make_output(self, items) -> ScoredPaperCollection:
        return ScoredPaperCollection(papers=items)


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


class InfluenceFloorFilter(Function[ResolvedPaperCollection, ResolvedPaperCollection]):
    """The one-sided, seed-derived influence floor (spec 11 FR4).

    The floor is a percentile of the resolved seed set's normalised
    influence values (4.2, Idea A): a candidate is dropped unless its own
    normalised influence reaches that floor. One-sided by design — a
    candidate far above the floor is a better result, never a rejection.
    Candidates the assessor could not measure carry no influence and cannot
    be shown to reach the bar, so they are dropped and reported too.

    A very small seed set makes a percentile unstable; that is a config-
    documentation concern (4.2), not a code special case.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        seed_influences: list[float],
        percentile: float = 10.0,
    ) -> None:
        self.seed_influences = [float(v) for v in seed_influences]
        self.percentile = percentile
        super().__init__(monitoring_handler)

    def derive_floor(self) -> float | None:
        """The percentile of the seed influence values, or None with no seeds."""
        if not self.seed_influences:
            return None
        values = sorted(self.seed_influences)
        # linear interpolation between closest ranks (numpy's default method)
        pos = (len(values) - 1) * (self.percentile / 100.0)
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return values[int(pos)]
        frac = pos - lower
        return values[lower] * (1.0 - frac) + values[upper] * frac

    def _run(
        self, data: ResolvedPaperCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        floor = self.derive_floor()
        kept: list[ResolvedPaper] = []
        dropped: list[tuple[ResolvedPaper, str]] = []

        for item in data.papers:
            score = item.influence_score
            if score is None:
                dropped.append((item, "influence_unmeasurable"))
            elif floor is not None and score < floor:
                dropped.append((item, f"below_floor_{floor:g}"))
            else:
                kept.append(item)

        total = len(data.papers)
        lines = [
            f"Papers in: {total}, kept: {len(kept)}, dropped: {len(dropped)} "
            f"(floor {floor if floor is not None else 'n/a'} "
            f"= p{self.percentile:g} of {len(self.seed_influences)} seeds)"
        ]
        for item, reason in dropped:
            score_text = (
                f"{item.influence_score:g}"
                if item.influence_score is not None
                else "unmeasurable"
            )
            lines.append(
                f"### {item.title}\nInfluence: {score_text} | Reason: {reason}\n"
            )
        return ResolvedPaperCollection(papers=kept), "\n".join(lines)


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
