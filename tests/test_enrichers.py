"""Unit tests for the web enricher (additional context points flow)."""

import logging
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from comourat.models import EnrichmentFailure, EnrichmentSuccess
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


def _make_scripted_enricher(
    outcomes: list, captured: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[WebEnricher, MagicMock]:
    """Create a WebEnricher with scripted public Comourat outcomes."""
    core = MagicMock()
    core.enrich.side_effect = outcomes
    monkeypatch.setattr(
        "mourat.enrichers.web_enricher.EnrichmentAgent", MagicMock(return_value=core)
    )
    enricher = WebEnricher(
        monitoring_handler=_make_monitoring_handler(captured), model=MagicMock()
    )
    return enricher, core


def _single_post_collection(**overrides) -> RedditPostCollection:
    return RedditPostCollection(posts=[_make_post(**overrides)])


# --- Tests ---


class TestWebEnricherProxyConfiguration:
    def test_proxy_is_forwarded_to_comourat_core(self, monkeypatch):
        core = MagicMock()
        monkeypatch.setattr("mourat.enrichers.web_enricher.EnrichmentAgent", core)
        proxy = "socks5h://proxy.example:1080"
        WebEnricher(
            monitoring_handler=_make_monitoring_handler([]),
            model=MagicMock(),
            proxy=proxy,
        )
        assert core.call_args.kwargs["proxy"] == proxy

    def test_proxy_defaults_to_none(self, monkeypatch):
        core = MagicMock()
        monkeypatch.setattr("mourat.enrichers.web_enricher.EnrichmentAgent", core)
        WebEnricher(
            monitoring_handler=_make_monitoring_handler([]),
            model=MagicMock(),
        )
        assert core.call_args.kwargs.get("proxy") is None
        assert core.call_args.kwargs["overall_timeout"] == 120.0

    @pytest.mark.parametrize("proxy", [None, "socks5h://proxy.example:1080"])
    def test_overall_timeout_is_forwarded_to_comourat_core(self, monkeypatch, proxy):
        core = MagicMock()
        monkeypatch.setattr("mourat.enrichers.web_enricher.EnrichmentAgent", core)
        WebEnricher(
            monitoring_handler=_make_monitoring_handler([]),
            model=MagicMock(),
            proxy=proxy,
            overall_timeout=456.0,
        )
        assert core.call_args.kwargs["overall_timeout"] == 456.0


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


class TestWebEnricherFailures:
    @pytest.mark.parametrize(
        "category", ["search", "fetch", "provider", "usage_limit", "timeout"]
    )
    def test_returned_failure_skips_only_failed_post(
        self, monkeypatch, caplog, category
    ):
        captured: list[str] = []
        message = f"Sanitized {category} failure."
        failure = EnrichmentFailure(error_category=category, error_message=message)
        enricher, core = _make_scripted_enricher(
            [failure, EnrichmentSuccess(additional_context=["later context"])],
            captured,
            monkeypatch,
        )
        posts = RedditPostCollection(
            posts=[
                _make_post(submission_id="failed123", title="Failed Post"),
                _make_post(submission_id="later456", title="Later Post"),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="mourat.enrichers.web_enricher"):
            output = enricher(posts, step_id="1")

        assert core.enrich.call_count == 2
        assert [item.post.submission_id for item in output.posts] == ["later456"]
        assert output.posts[0].additional_context == ["later context"]
        assert "Later Post" in captured[-1]
        assert "Failed Post" not in captured[-1]
        assert any(
            "failed123" in record.getMessage()
            and category in record.getMessage()
            and message in record.getMessage()
            for record in caplog.records
        )

    def test_unexpected_exception_keeps_existing_skip_behavior(
        self, monkeypatch, caplog
    ):
        captured: list[str] = []
        enricher, core = _make_scripted_enricher(
            [
                RuntimeError("unexpected adapter failure"),
                EnrichmentSuccess(additional_context=["later context"]),
            ],
            captured,
            monkeypatch,
        )
        posts = RedditPostCollection(
            posts=[
                _make_post(submission_id="failed123", title="Failed Post"),
                _make_post(submission_id="later456", title="Later Post"),
            ]
        )

        with caplog.at_level(logging.ERROR, logger="mourat.enrichers.web_enricher"):
            output = enricher(posts, step_id="1")

        assert core.enrich.call_count == 2
        assert [item.post.submission_id for item in output.posts] == ["later456"]
        assert any(record.exc_info is not None for record in caplog.records)


class TestEnrichmentDependencies:
    def test_compatible_versions_and_reviewed_comourat_commit_are_explicit(self):
        project_root = Path(__file__).resolve().parent.parent
        with (project_root / "pyproject.toml").open("rb") as stream:
            project_dependencies = tomllib.load(stream)["project"]["dependencies"]

        def requirements(name: str) -> list[str]:
            lines = (project_root / name).read_text(encoding="utf-8").splitlines()
            return [
                line.strip()
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            ]

        runtime_requirements = requirements("requirements.txt")
        development_requirements = requirements("requirements_dev.txt")
        for dependencies in (
            project_dependencies,
            runtime_requirements,
            development_requirements,
        ):
            assert "pydantic-ai >= 2.64.0" in dependencies

        comourat = (
            "comourat @ git+https://github.com/anton-pershin/comourat.git@"
            "f15a0375889464d44f8ca93991fc32dd9b43a2a5"
        )
        assert comourat in project_dependencies
        assert comourat in runtime_requirements
