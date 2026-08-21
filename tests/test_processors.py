"""Unit tests for processor modules (classifiers, scorers, assigners, generators, assessors)."""

import datetime
from unittest.mock import MagicMock, patch

from mourat.classifiers import BinaryPaperClassifier
from mourat.scorers import PaperScorer
from mourat.assigners import PaperAssigner
from mourat.generators import QueryGeneratorViaLlm
from mourat.data_models import (
    PaperInfo, PaperInfoCollection,
    ScoredPaperInfo, ScoredPaperInfoCollection,
    PaperScoredByAgent,
    AssignedPaperInfo, AssignedPaperInfoCollection,
    QueryInfo,
)
from mourat.monitoring import MonitoringHandler


# --- Helpers ---

def _make_monitoring_handler():
    class DummyHandler(MonitoringHandler):
        def __init__(self):
            pass
        def __call__(self, step: str, text_for_monitoring: str) -> None:
            pass
    return DummyHandler()


def _make_sample_papers():
    return PaperInfoCollection(papers=[
        PaperInfo(
            title="Paper A",
            link="http://example.com/a",
            abstract="Abstract A",
            citation_count=10,
            authors=["Author A"],
            publication_date=datetime.date(2024, 1, 1),
        ),
        PaperInfo(
            title="Paper B",
            link="http://example.com/b",
            abstract="Abstract B",
            citation_count=5,
            authors=["Author B"],
            publication_date=datetime.date(2024, 1, 2),
        ),
    ])


# --- BinaryPaperClassifier ---

class TestBinaryPaperClassifier:
    def test_keeps_relevant_papers(self):
        mock_handler = _make_monitoring_handler()
        mock_model = MagicMock()

        # First paper relevant, second not
        mock_result_1 = MagicMock()
        mock_result_1.output = True
        mock_result_2 = MagicMock()
        mock_result_2.output = False

        agent_mock = MagicMock()
        agent_mock.run_sync.side_effect = [mock_result_1, mock_result_2]

        with patch("mourat.classifiers.Agent", return_value=agent_mock):
            classifier = BinaryPaperClassifier(
                monitoring_handler=mock_handler,
                model=mock_model,
                topic_name="test topic",
                user_prompt_template="{title} {abstract} {topic_name}",
                text_for_monitoring_template="{title} {link} {abstract}",
                system_prompt="test",
            )

        papers = _make_sample_papers()
        result = classifier(papers, "classify")

        assert isinstance(result, PaperInfoCollection)
        assert len(result.papers) == 1
        assert result.papers[0].title == "Paper A"

    def test_removes_all_irrelevant(self):
        mock_handler = _make_monitoring_handler()
        mock_model = MagicMock()

        mock_result = MagicMock()
        mock_result.output = False

        agent_mock = MagicMock()
        agent_mock.run_sync.return_value = mock_result

        with patch("mourat.classifiers.Agent", return_value=agent_mock):
            classifier = BinaryPaperClassifier(
                monitoring_handler=mock_handler,
                model=mock_model,
                topic_name="test topic",
                user_prompt_template="{title} {abstract} {topic_name}",
                text_for_monitoring_template="{title} {link} {abstract}",
                system_prompt="test",
            )

        papers = _make_sample_papers()
        result = classifier(papers, "classify")

        assert len(result.papers) == 0


# --- PaperScorer ---

class TestPaperScorer:
    def test_scores_papers(self):
        mock_handler = _make_monitoring_handler()
        mock_model = MagicMock()

        scored = [
            PaperScoredByAgent(title="Paper A", score=4, justification="Good match"),
            PaperScoredByAgent(title="Paper B", score=2, justification="Weak match"),
        ]
        mock_result = MagicMock()
        mock_result.output = scored

        agent_mock = MagicMock()
        agent_mock.run_sync.return_value = mock_result

        with patch("mourat.scorers.Agent", return_value=agent_mock):
            scorer = PaperScorer(
                monitoring_handler=mock_handler,
                model=mock_model,
                topic_name="test topic",
                topic_description="test desc",
                user_prompt_template="{topic_name} {topic_description} {papers_as_json}",
                text_for_monitoring_template="{title} {link} {abstract} {score} {justification}",
                system_prompt="test",
            )

        papers = _make_sample_papers()
        result = scorer(papers, "score")

        assert isinstance(result, ScoredPaperInfoCollection)
        assert len(result.papers) == 2
        assert result.papers[0].score == 4
        assert result.papers[1].score == 2


# --- PaperAssigner ---

class TestPaperAssigner:
    def test_assigns_topics(self):
        mock_handler = _make_monitoring_handler()
        mock_model = MagicMock()

        agent_mock = MagicMock()
        agent_mock.run_sync.side_effect = [
            MagicMock(output="topic_a"),
            MagicMock(output="none"),
        ]

        with patch("mourat.assigners.Agent", return_value=agent_mock):
            assigner = PaperAssigner(
                monitoring_handler=mock_handler,
                model=mock_model,
                topics="topic_a, topic_b",
                user_prompt_template="{topics} {paper}",
                text_for_monitoring_template="{title} {link} {abstract} {assigned_topics}",
                system_prompt="test",
            )

        papers = _make_sample_papers()
        result = assigner(papers, "assign")

        assert isinstance(result, AssignedPaperInfoCollection)
        assert len(result.papers) == 1
        assert result.papers[0].assigned_topics == ["topic_a"]


# --- QueryGeneratorViaLlm ---

class TestQueryGeneratorViaLlm:
    def test_generates_queries(self):
        mock_handler = _make_monitoring_handler()
        mock_model = MagicMock()

        mock_result = MagicMock()
        mock_result.output = QueryInfo(
            general_queries=["query 1", "query 2"],
            specific_queries=["specific 1"],
        )

        agent_mock = MagicMock()
        agent_mock.run_sync.return_value = mock_result

        with patch("mourat.generators.Agent", return_value=agent_mock):
            generator = QueryGeneratorViaLlm(
                monitoring_handler=mock_handler,
                model=mock_model,
                topic_name="test topic",
                topic_description="test desc",
                system_prompt="test",
                user_prompt_template="{topic_name} {topic_description}",
                text_for_monitoring_template="{topic_name} {topic_description} {general_queries} {specific_queries}",
            )

        result = generator(None, "generate")

        assert isinstance(result, QueryInfo)
        assert len(result.general_queries) == 2
        assert len(result.specific_queries) == 1
