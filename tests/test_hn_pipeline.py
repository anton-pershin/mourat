"""Unit tests for the HN pipeline stages (specs 19 and 24)."""

import json
import logging
import threading
from unittest.mock import MagicMock

import pytest
import requests
from comourat.enrichment.page_fetch import ArticleFetchError

from mourat.data_models import HNPostCollection, HNPostInfo
from mourat.monitoring import MonitoringHandler
from mourat.processors.hn_fetcher import HNContentFetcher
from mourat.processors.hn_influence import HNInfluenceAssessor, HNInfluenceFilter
from mourat.processors.post_influence import compute_influence_score


def _post(**overrides):
    base = {
        "hn_id": "100",
        "title": "Test post",
        "author": "bob",
        "date": "2026-10-15T00:00:00+00:00",
        "url": "https://example.com/x",
        "text": "",
        "points": 50,
        "num_comments": 3,
    }
    base.update(overrides)
    return HNPostInfo(**base)


def _collection(posts):
    return HNPostCollection(posts=posts)


class _CapturingHandler(MonitoringHandler):
    """Captures monitoring text for assertions."""

    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


# -- Influence ----------------------------------------------------------------


class TestComputeHnInfluenceScore:
    def test_anchor_and_bounds(self):
        # T5 (R4, B2): at reference -> 50; 0 -> 0; 9x reference -> 90.
        assert compute_influence_score(100, 100) == 50
        assert compute_influence_score(0, 100) == 0
        assert compute_influence_score(900, 100) == 90

    def test_rounding(self):
        # T5 (R4): non-integer ratio rounding verified.
        # ratio = 1/3 -> 100 * (1/3)/(4/3) = 25
        assert compute_influence_score(1, 3) == 25


class TestHNInfluenceAssessor:
    def test_writes_influence_score(self):
        # T5 (R4): assessor writes influence_score onto posts.
        handler = _CapturingHandler()
        assessor = HNInfluenceAssessor(handler, reference_level=100)
        out = assessor(_collection([_post(points=100), _post(points=0)]), step_id="1.5")
        assert out.posts[0].influence_score == 50
        assert out.posts[1].influence_score == 0

    def test_nonpositive_reference_raises(self):
        # T6 (R4, B3): reference_level <= 0 -> ValueError naming the value.
        handler = _CapturingHandler()
        with pytest.raises(ValueError, match="0"):
            HNInfluenceAssessor(handler, reference_level=0)
        with pytest.raises(ValueError, match="-5"):
            HNInfluenceAssessor(handler, reference_level=-5)


class TestHNInfluenceFilter:
    def test_drops_below_threshold(self):
        # T7 (R4, B1): 49 dropped, 50 and 51 survive.
        handler = _CapturingHandler()
        f = HNInfluenceFilter(handler, min_influence=50)
        posts = [
            _post(hn_id="1", points=49),
            _post(hn_id="2", points=50),
            _post(hn_id="3", points=51),
        ]
        for p in posts:
            p.influence_score = p.points
        out = f(_collection(posts), step_id="1.6")
        assert [p.hn_id for p in out.posts] == ["2", "3"]
        assert any("filtered" in text or "1" in text for _, text in handler.calls)


# -- Fetcher -------------------------------------------------------------------


def _html(body: str, title: str | None = "Article title") -> str:
    title_html = "" if title is None else f"<head><title>{title}</title></head>"
    return f"<html>{title_html}<body>{body}</body></html>"


class _FakeResponse:
    def __init__(
        self,
        status_code=200,
        text=_html("hello world"),
        url="https://example.com/final-article",
    ):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class _FakeSession(requests.Session):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.proxies = {}

    def get(self, url, timeout=None, headers=None, allow_redirects=True):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestHNContentFetcher:
    def _run(self, handler, session, posts, **kwargs):
        fetcher = HNContentFetcher(handler, http_session=session, proxy=None, **kwargs)
        # Trafilatura is stubbed through the fetcher's extractor injection point.
        return fetcher(_collection(posts), step_id="2")

    def test_success_assigns_plain_text_and_serializes_a_string(self, monkeypatch):
        # Spec 24 T1: only ArticleResult.text reaches the HN model.
        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "  extracted text  ",
        )
        handler = _CapturingHandler()
        response = _FakeResponse(200)
        out = self._run(handler, _FakeSession([response]), [_post(hn_id="success")])

        assert [post.hn_id for post in out.posts] == ["success"]
        assert type(out.posts[0].text) is str
        assert out.posts[0].text == "extracted text"
        serialized = json.loads(out.posts[0].model_dump_json())
        assert serialized["text"] == "extracted text"
        assert type(serialized["text"]) is str
        assert response.closed

    def test_expected_failures_drop_only_failed_posts(self, monkeypatch, caplog):
        # Spec 24 T3: all expected categories are logged and later work continues.
        provider_detail = "provider-private-detail"
        extractor_detail = "extractor-private-detail"

        def extract(html):
            if "empty-text" in html:
                return "  \n"
            if "extractor-error" in html:
                raise RuntimeError(extractor_detail)
            return "  extracted text  "

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content", extract
        )
        responses = [
            _FakeResponse(404),
            _FakeResponse(200, _html("empty-text")),
            _FakeResponse(200, _html("hello world", title=None)),
            _FakeResponse(200, _html("extractor-error")),
            _FakeResponse(503),
            requests.ConnectionError(provider_detail),
            requests.ConnectTimeout("connection-private-detail"),
            _FakeResponse(200),
        ]
        expected_failures = [
            ("unavailable", "unavailable"),
            ("empty-text", "empty_extraction"),
            ("missing-title", "empty_extraction"),
            ("extractor-error", "extraction"),
            ("http-provider", "provider"),
            ("invalid-input", "invalid_input"),
            ("connection-provider", "provider"),
            ("connection-timeout", "timeout"),
        ]
        posts = [_post(hn_id=post_id) for post_id, _ in expected_failures]
        posts[5].url = "not a URL"
        posts.append(_post(hn_id="later-success"))
        handler = _CapturingHandler()

        with caplog.at_level(logging.DEBUG, logger="mourat.processors.hn_fetcher"):
            out = self._run(handler, _FakeSession(responses), posts)

        assert [post.hn_id for post in out.posts] == ["later-success"]
        assert out.posts[0].text == "extracted text"
        assert handler.calls[-1] == ("2", "HN content fetcher: 9 in, 1 out, 8 dropped")
        records = [
            record
            for record in caplog.records
            if record.name == "mourat.processors.hn_fetcher"
        ]
        assert len(records) == len(expected_failures)
        for record, (post_id, category) in zip(records, expected_failures):
            message = record.getMessage()
            assert post_id in message
            assert category in message
            assert str(ArticleFetchError(category)) in message
        assert provider_detail not in caplog.text
        assert extractor_detail not in caplog.text
        assert "connection-private-detail" not in caplog.text
        assert all(
            response.closed
            for response in responses
            if isinstance(response, _FakeResponse)
        )

    def test_overall_timeout_drops_only_failed_post(self, monkeypatch, caplog):
        # Spec 24 T4: the overall deadline includes extraction time.
        extraction_started = threading.Event()
        release_extraction = threading.Event()

        def extract(html):
            if "slow-extraction" in html:
                extraction_started.set()
                release_extraction.wait(1)
            return "extracted text"

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content", extract
        )
        handler = _CapturingHandler()
        session = _FakeSession(
            [
                _FakeResponse(200, _html("slow-extraction")),
                _FakeResponse(200),
            ]
        )
        posts = [_post(hn_id="slow"), _post(hn_id="later-success")]
        try:
            with caplog.at_level(logging.DEBUG, logger="mourat.processors.hn_fetcher"):
                out = self._run(handler, session, posts, timeout=0.02)
        finally:
            release_extraction.set()

        assert extraction_started.is_set()
        assert [post.hn_id for post in out.posts] == ["later-success"]
        assert handler.calls[-1] == ("2", "HN content fetcher: 2 in, 1 out, 1 dropped")
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "mourat.processors.hn_fetcher"
        ]
        assert len(messages) == 1
        assert "slow" in messages[0]
        assert "timeout" in messages[0]
        assert str(ArticleFetchError("timeout")) in messages[0]

    def test_unexpected_adapter_failure_propagates(self, monkeypatch, caplog):
        # Spec 24 T5: an adapter defect is not an expected article failure.
        fetcher = HNContentFetcher(_CapturingHandler(), http_session=_FakeSession([]))
        monkeypatch.setattr(
            fetcher.core,
            "fetch",
            MagicMock(side_effect=RuntimeError("unexpected adapter defect")),
        )
        with caplog.at_level(logging.DEBUG, logger="mourat.processors.hn_fetcher"):
            with pytest.raises(RuntimeError, match="unexpected adapter defect"):
                fetcher(_collection([_post()]), step_id="2")
        assert not [
            record
            for record in caplog.records
            if record.name == "mourat.processors.hn_fetcher"
        ]

    def test_timeout_defaults_and_custom_values(self):
        # Spec 24 T6: connection and overall operation scopes stay independent.
        default_fetcher = HNContentFetcher(
            _CapturingHandler(), http_session=_FakeSession([])
        )
        assert default_fetcher.core.timeout == 30
        assert default_fetcher.core.operation_timeout == 30
        assert default_fetcher.core.connection_timeout == 10

        custom_fetcher = HNContentFetcher(
            _CapturingHandler(),
            http_session=_FakeSession([]),
            timeout=45,
            connection_timeout=2.5,
        )
        assert custom_fetcher.core.timeout == 45
        assert custom_fetcher.core.operation_timeout == 45
        assert custom_fetcher.core.connection_timeout == 2.5

    def test_proxy_passed_to_session(self, monkeypatch):
        # T13 (R6): configured proxy lands in the session's proxies dict.
        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "t",
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200)])
        fetcher = HNContentFetcher(
            handler, http_session=session, proxy="http://proxy:3128"
        )
        fetcher(_collection([_post()]), step_id="2")
        assert session.proxies == {
            "http": "http://proxy:3128",
            "https": "http://proxy:3128",
        }


# -- Summarizer ----------------------------------------------------------------


class _StubLLM:
    """Stub pydantic-ai Model stand-in: returns queued strings."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class TestHNSummarizer:
    def test_summary_set_and_text_discarded(self, monkeypatch):
        # T14 (R7): description set; fetched text NOT retained on the model.
        from mourat.processors.hn_summarizer import HNSummarizer

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "DISTINCT FETCHED ARTICLE BODY",
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200)])
        fetcher = HNContentFetcher(handler, http_session=session, proxy=None)
        fetched = fetcher(_collection([_post(hn_id="1", points=60)]), step_id="2")

        llm = _StubLLM(['{"summary": "A compact summary of the article."}'])
        summarizer = HNSummarizer(handler, llm=llm, retries=2)
        out = summarizer(fetched, step_id="3")
        assert out.posts[0].description == "A compact summary of the article."
        # fetched content is not stored on the post (only title+description remain)
        assert "DISTINCT FETCHED ARTICLE BODY" not in out.posts[0].model_dump_json()

    def test_prompt_contains_title_and_content(self, monkeypatch):
        # T15 and spec 24 T2: the prompt contains only title + plain article text.
        from mourat.processors.hn_summarizer import HNSummarizer

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "DISTINCT FETCHED ARTICLE BODY",
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200)])
        fetched = HNContentFetcher(handler, http_session=session, proxy=None)(
            _collection([_post(hn_id="1", title="My title", points=60)]), step_id="2"
        )
        llm = _StubLLM(['{"summary": "summary"}'])
        HNSummarizer(handler, llm=llm, retries=1)(fetched, step_id="3")
        assert llm.prompts[0] == (
            "Title: My title\n\nContent:\nDISTINCT FETCHED ARTICLE BODY"
        )
        assert "source_url=" not in llm.prompts[0]
        assert "ArticleResult" not in llm.prompts[0]

    def test_configured_model_uses_agent_structured_output(self, monkeypatch):
        # R7 review regression: exercise model->agent boundary using stub Model.
        from types import SimpleNamespace

        from mourat.processors import hn_summarizer

        calls = []

        class StubAgent:
            def __init__(self, model, **kwargs):
                calls.append((model, kwargs))

            def run_sync(self, prompt):
                calls.append(prompt)
                return SimpleNamespace(
                    output=hn_summarizer._SummaryOutput(summary="summary")
                )

        monkeypatch.setattr(hn_summarizer, "Agent", StubAgent)
        handler = _CapturingHandler()
        model = object()  # stands in for configured pydantic-ai Model
        summarizer = hn_summarizer.HNSummarizer(handler, model=model, retries=0)
        post = _post(points=60)
        post.text = "DISTINCT FETCHED ARTICLE BODY"
        out = summarizer(_collection([post]), step_id="3")
        assert calls[0][0] is model
        assert out.posts[0].description == "summary"
        assert "DISTINCT FETCHED ARTICLE BODY" not in out.posts[0].model_dump_json()
        assert "DISTINCT FETCHED ARTICLE BODY" in calls[1]

    def test_logs_one_progress_line_per_llm_call(self, caplog):
        import logging
        from types import SimpleNamespace

        from mourat.processors import hn_summarizer

        calls = []

        class StubAgent:
            def run_sync(self, prompt):
                calls.append(prompt)
                if len(calls) == 1:
                    raise RuntimeError("temporary model failure")
                return SimpleNamespace(
                    output=hn_summarizer._SummaryOutput(summary="summary")
                )

        caplog.set_level(logging.INFO, logger="mourat.processors.hn_summarizer")
        summarizer = hn_summarizer.HNSummarizer(
            _CapturingHandler(), agent=StubAgent(), retries=1
        )
        out = summarizer(
            _collection([_post(hn_id="log-test"), _post(hn_id="log-test-2")]),
            step_id="3",
        )

        assert len(out.posts) == 2
        records = [
            record.getMessage()
            for record in caplog.records
            if "summarized HN post" in record.getMessage()
        ]
        assert len(records) == 3  # one line per model invocation, including retry
        assert "(1/2)" in records[0] and "post 'Test post'" in records[0]
        assert "failed" in records[0] and "attempt 1/2" in records[0]
        assert "(1/2)" in records[1] and "post 'Test post'" in records[1]
        assert "success" in records[1] and "attempt 2/2" in records[1]
        assert "(2/2)" in records[2] and "post 'Test post'" in records[2]
        assert "success" in records[2] and "attempt 1/2" in records[2]
        assert all("| " in record and "s |" in record for record in records)

    def test_retry_then_drop(self, monkeypatch):
        # T14 (R7, B6): retries configurable; drop after exhaustion; success on retry N honored.
        from mourat.processors.hn_summarizer import HNSummarizer

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content", lambda html: "t"
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200), _FakeResponse(200)])
        fetched = HNContentFetcher(handler, http_session=session, proxy=None)(
            _collection([_post(hn_id="1", points=60), _post(hn_id="2", points=61)]),
            step_id="2",
        )
        # post 1: fails twice then succeeds (retries=2); post 2: always fails (retries=1)
        llm = _StubLLM(["bad", "bad", '{"summary": "good summary"}', "bad"])
        out = HNSummarizer(handler, llm=llm, retries=2)(fetched, step_id="3")
        assert [p.hn_id for p in out.posts] == ["1"]
        assert out.posts[0].description == "good summary"
