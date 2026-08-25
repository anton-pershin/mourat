from mourat.base import Function
from mourat.data_models import ScoredPaperInfoCollection, ScoredRedditPostCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_text_description


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
