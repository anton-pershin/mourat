"""Normalized Reddit post influence: assessor and filter (spec 17).

Compares each post's raw upvote score against a per-subreddit reference
level (measured by the calibration script, spec 16, and supplied via the
`post_influence.references` config mapping). The influence-score formula
lives in `compute_influence_score` alone; the pipeline structure (assessor
and filter placement right after collection) is independent of the formula.
"""

from __future__ import annotations

from mourat.base import Function
from mourat.data_models import RedditPostCollection
from mourat.monitoring import MonitoringHandler


def compute_influence_score(score: int, reference: float) -> int:
    """Saturating normalized influence: 0 -> 0, at reference -> 50, large -> 100."""
    ratio = score / max(reference, 1)
    return round(100 * ratio / (ratio + 1))


class PostInfluenceAssessor(Function[RedditPostCollection, RedditPostCollection]):
    """Writes the normalized influence score onto each collected post.

    Raises when a post's subreddit has no configured reference: loud failure
    by design (spec 17 R5).
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        references: dict[str, float],
    ) -> None:
        self.references = references
        super().__init__(monitoring_handler)

    def _run(self, data: RedditPostCollection) -> tuple[RedditPostCollection, str]:
        posts = list(data.posts)
        for post in posts:
            if post.subreddit not in self.references:
                raise ValueError(
                    f"subreddit '{post.subreddit}' has no reference level in "
                    "post_influence.references; add it to the config or set "
                    "post_influence.enabled: false"
                )
            post.influence_score = compute_influence_score(
                post.score, self.references[post.subreddit]
            )
        output = RedditPostCollection(posts=posts)
        text_for_monitoring = (
            f"posts in: {len(posts)}, influence assigned: {len(posts)}"
        )
        return output, text_for_monitoring


class PostInfluenceFilter(Function[RedditPostCollection, RedditPostCollection]):
    """Drops posts whose influence_score is below `min_influence`."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        min_influence: int,
    ) -> None:
        self.min_influence = min_influence
        super().__init__(monitoring_handler)

    def _run(self, data: RedditPostCollection) -> tuple[RedditPostCollection, str]:
        kept = [
            p
            for p in data.posts
            if p.influence_score is not None and p.influence_score >= self.min_influence
        ]
        output = RedditPostCollection(posts=kept)
        text_for_monitoring = (
            f"posts in: {len(data.posts)}, posts out: {len(kept)}, "
            f"filtered: {len(data.posts) - len(kept)} "
            f"(min_influence={self.min_influence})"
        )
        return output, text_for_monitoring
