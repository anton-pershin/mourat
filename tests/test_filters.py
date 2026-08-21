"""Unit tests for filter modules."""

import datetime

from mourat.data_models import ScoredPaperInfo, ScoredPaperInfoCollection
from mourat.filters import ScoreBasedPaperFilter
from mourat.monitoring import MonitoringHandler


def _make_monitoring_handler():
    class DummyHandler(MonitoringHandler):
        def __init__(self):
            pass
        def __call__(self, step: str, text_for_monitoring: str) -> None:
            pass
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
