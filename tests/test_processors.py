"""Unit tests for processor modules (classifiers, scorers, assigners, generators, assessors)."""

import datetime
import json
from unittest.mock import MagicMock, patch

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from mourat.assigners import PaperAssigner
from mourat.classifiers import BinaryPaperClassifier
from mourat.data_models import (
    AssignedPaperInfo,
    AssignedPaperInfoCollection,
    PaperInfo,
    PaperInfoCollection,
    PaperScoredByAgent,
    QueryInfo,
    ScoredPaperInfo,
    ScoredPaperInfoCollection,
)
from mourat.generators import QueryGeneratorViaLlm
from mourat.monitoring import MonitoringHandler
from mourat.scorers import PaperScorer

# --- Helpers ---


def _make_monitoring_handler():
    class DummyHandler(MonitoringHandler):
        def __init__(self):
            pass

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            pass

    return DummyHandler()


def _make_sample_papers():
    return PaperInfoCollection(
        papers=[
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
        ]
    )


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


# --- PostScorer ---


def _make_sample_post(**overrides):
    from mourat.data_models import RedditPostInfo

    defaults = {
        "subreddit": "test_sub",
        "submission_id": "abc123",
        "title": "Test Post",
        "author": "test_author",
        "date": "2026-08-30T12:00:00",
        "url": "https://example.com/post",
        "text": "RAW TEXT",
        "score": 5,
    }
    defaults.update(overrides)
    return RedditPostInfo(**defaults)


def _make_scorer(
    captured_prompts: list[str], captured_monitoring: list[str], scripted_scores: list,
    constraint_list: list | None = None,
):
    """Create a PostScorer with a FunctionModel returning scripted scoring results.

    scripted_scores: list of lists of raw score-entry dicts returned per call.
    """
    from mourat.data_models import ScoringResult
    from mourat.processors.post_scorer import PostScorer

    def model_fn(messages, agent):
        captured_prompts.append(messages[-1].parts[-1].content)
        result = ScoringResult.model_validate({"scores": scripted_scores.pop(0)})
        return ModelResponse(parts=[TextPart(result.model_dump_json())])

    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            pass

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            captured_monitoring.append(text_for_monitoring)

    return PostScorer(
        monitoring_handler=CapturingHandler(),
        model=FunctionModel(model_fn),
        rq_list=[{"id": "rq1", "name": "RQ one", "type": "rq", "description": "d"}],
        constraint_list=constraint_list or [],
    )


class TestPostScorerAdditionalContextAndMaxScore:
    def test_scorer_prompt_contains_text_then_points(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[{"id": "rq1", "type": "rq", "score": 50, "justification": "j"}]],
        )

        ep = EnrichedRedditPost(
            post=_make_sample_post(),
            additional_context=["ctx alpha", "ctx beta"],
        )
        scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        prompt = captured_prompts[0]
        assert "RAW TEXT" in prompt
        assert "ctx alpha" in prompt
        assert "ctx beta" in prompt
        assert prompt.index("RAW TEXT") < prompt.index("ctx alpha")

    def test_scorer_prompt_without_context(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[{"id": "rq1", "type": "rq", "score": 50, "justification": "j"}]],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        prompt = captured_prompts[0]
        assert "RAW TEXT" in prompt
        assert "Additional context" not in prompt

    def test_max_score_ignores_unknown_ids(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [
                [
                    {"id": "rq1", "type": "rq", "score": 40, "justification": "valid"},
                    {
                        "id": "hallucinated",
                        "type": "rq",
                        "score": 95,
                        "justification": "phantom",
                    },
                ]
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert len(sp.relevance_scores) == 1
        assert sp.relevance_scores[0].id == "rq1"
        assert sp.relevance_scores[0].score == 40
        assert sp.max_score == 40.0
        assert "Max score: 40" in captured_monitoring[-1]

    def test_max_score_zero_when_no_valid_entries(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [
                [
                    {
                        "id": "hallucinated",
                        "type": "rq",
                        "score": 95,
                        "justification": "phantom",
                    }
                ]
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert sp.relevance_scores == []
        assert sp.max_score == 0.0
        assert len(output.posts) == 1  # post retained, not dropped

    def test_constraint_scores_returned(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[
                {"id": "c1", "type": "constraint", "score": 70, "justification": "aligned"},
            ]],
            constraint_list=[
                {"id": "c1", "name": "Memory limit", "type": "constraint", "description": "d"}
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert len(sp.relevance_scores) == 1
        assert sp.relevance_scores[0].type == "constraint"
        assert sp.relevance_scores[0].score == 70
        # Constraint scores do not count toward max_score
        assert sp.max_score == 0.0
        assert "Memory limit" in captured_prompts[0]

    def test_max_score_excludes_constraints(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[
                {"id": "rq1", "type": "rq", "score": 90, "justification": "relevant"},
                {"id": "c1", "type": "constraint", "score": 20, "justification": "fails"},
            ]],
            constraint_list=[
                {"id": "c1", "name": "Memory limit", "type": "constraint", "description": "d"}
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert sp.max_score == 90.0
        assert {e.type for e in sp.relevance_scores} == {"rq", "constraint"}

    def test_max_score_constraint_only(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[
                {"id": "c1", "type": "constraint", "score": 85, "justification": "high"},
            ]],
            constraint_list=[
                {"id": "c1", "name": "Memory limit", "type": "constraint", "description": "d"}
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert sp.max_score == 0.0
        assert len(sp.relevance_scores) == 1
        assert sp.relevance_scores[0].type == "constraint"

    def test_invalid_constraint_ids_filtered(self):
        from mourat.data_models import (
            EnrichedRedditPost,
            EnrichedRedditPostCollection,
        )

        captured_prompts: list[str] = []
        captured_monitoring: list[str] = []
        scorer = _make_scorer(
            captured_prompts,
            captured_monitoring,
            [[
                {
                    "id": "hallucinated",
                    "type": "constraint",
                    "score": 95,
                    "justification": "phantom",
                },
            ]],
            constraint_list=[
                {"id": "c1", "name": "Memory limit", "type": "constraint", "description": "d"}
            ],
        )

        ep = EnrichedRedditPost(post=_make_sample_post(), additional_context=[])
        output = scorer(EnrichedRedditPostCollection(posts=[ep]), step_id="1")

        sp = output.posts[0]
        assert sp.relevance_scores == []
        assert sp.max_score == 0.0
