"""Unit tests for the web enricher (additional context points flow)."""

import json

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from mourat.data_models import EnrichmentResult, RedditPostCollection, RedditPostInfo
from mourat.enrichers.web_enricher import WebEnricher
from mourat.monitoring import MonitoringHandler

# --- Helpers ---


def _make_monitoring_handler(captured: list[str]):
    class DummyHandler(MonitoringHandler):
        def __init__(self):
            pass

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            captured.append(text_for_monitoring)

    return DummyHandler()


def _make_post(**overrides):
    defaults = {
        "subreddit": "test_sub",
        "submission_id": "abc123",
        "title": "Test Post",
        "author": "test_author",
        "date": "2026-08-30T12:00:00",
        "url": "https://example.com/post",
        "text": "RAW POST TEXT",
        "score": 5,
    }
    defaults.update(overrides)
    return RedditPostInfo(**defaults)


def _make_enricher(points_by_call: list, captured: list[str]) -> WebEnricher:
    """Create a WebEnricher whose mock model returns scripted points per call."""

    def model_fn(messages, agent):
        # Scripted single response regardless of prompt content.
        result = EnrichmentResult(additional_context=points_by_call.pop(0))
        return ModelResponse(parts=[TextPart(result.model_dump_json())])

    handler = _make_monitoring_handler(captured)
    return WebEnricher(monitoring_handler=handler, model=FunctionModel(model_fn))


def _single_post_collection(**overrides) -> RedditPostCollection:
    return RedditPostCollection(posts=[_make_post(**overrides)])


# --- Tests ---


class TestWebEnricherAdditionalContext:
    def test_enriched_post_carries_additional_context(self):
        captured: list[str] = []
        enricher = _make_enricher([["Point one.", "Point two."]], captured)
        output = enricher(_single_post_collection(), step_id="1")

        assert len(output.posts) == 1
        assert output.posts[0].additional_context == ["Point one.", "Point two."]

    def test_empty_additional_context_retained(self):
        captured: list[str] = []
        enricher = _make_enricher([[]], captured)
        output = enricher(_single_post_collection(), step_id="1")

        assert output.posts[0].additional_context == []
        assert len(output.posts) == 1  # post retained on empty enrichment

    def test_monitoring_contains_additional_context(self):
        captured: list[str] = []
        enricher = _make_enricher([["ctx alpha", "ctx beta"]], captured)
        enricher(_single_post_collection(), step_id="1")

        monitoring = captured[-1]
        assert "RAW POST TEXT" in monitoring
        assert "ctx alpha" in monitoring
        assert "ctx beta" in monitoring
