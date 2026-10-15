"""HN influence assessment and filtering."""

from __future__ import annotations

from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler
from mourat.processors.post_influence import compute_influence_score


class HNInfluenceAssessor(Function[HNPostCollection, HNPostCollection]):
    """Computes the normalized influence score for HN posts.

    Reuses the neutral `compute_influence_score` from spec 17: the post at
    the reference level scores 50. A non-positive reference level is a config
    error and fails loudly.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        reference_level: float,
        text_for_monitoring_template: str = "Assessed influence of {count} HN posts",
    ) -> None:
        if reference_level <= 0:
            raise ValueError(f"reference_level must be positive, got {reference_level}")
        self.reference_level = reference_level
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        for post in data.posts:
            post.influence_score = compute_influence_score(
                post.points, self.reference_level
            )
        text_for_monitoring = self.text_for_monitoring_template.format(
            count=len(data.posts)
        )
        return data, text_for_monitoring


class HNInfluenceFilter(Function[HNPostCollection, HNPostCollection]):
    """Drops HN posts whose influence score is below `min_influence`."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        min_influence: int = 50,
        text_for_monitoring_template: str = (
            "HN influence filter: {before} in, {after} out, {filtered} filtered"
        ),
    ) -> None:
        self.min_influence = min_influence
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        kept = [p for p in data.posts if (p.influence_score or 0) >= self.min_influence]
        filtered = len(data.posts) - len(kept)
        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), filtered=filtered
        )
        return HNPostCollection(posts=kept), text_for_monitoring
