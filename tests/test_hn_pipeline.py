"""Unit tests for the HN pipeline stages (spec 19)."""

import pytest

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


class _FakeResponse:
    def __init__(self, status_code=200, text="<html><body>hello world</body></html>"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.proxies = {}

    def get(self, url, timeout=None, headers=None):
        return self.responses.pop(0)


class TestHNContentFetcher:
    def _run(self, handler, session, posts):
        fetcher = HNContentFetcher(handler, http_session=session, proxy=None)
        # trafilatura is stubbed via the fetcher's extractor injection point
        return fetcher(_collection(posts), step_id="2")

    def test_success_and_failures(self, monkeypatch):
        # T12 (R6, B4): 200+content kept; HTTP error and empty extraction dropped.
        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "extracted text" if "hello" in html else None,
        )
        handler = _CapturingHandler()
        session = _FakeSession(
            [
                _FakeResponse(200, "hello world"),  # ok
                _FakeResponse(404),  # dropped
                _FakeResponse(200, "<html></html>"),  # empty extraction -> dropped
            ]
        )
        posts = [_post(hn_id=str(i)) for i in range(3)]
        out = self._run(handler, session, posts)
        assert [p.hn_id for p in out.posts] == ["0"]
        assert out.posts[0].text == "extracted text"

    def test_proxy_passed_to_session(self, monkeypatch):
        # T13 (R6): configured proxy lands in the session's proxies dict.
        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "t",
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200, "hello world")])
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
            "mourat.processors.hn_fetcher._extract_main_content", lambda html: "t"
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200, "hello world")])
        fetcher = HNContentFetcher(handler, http_session=session, proxy=None)
        fetched = fetcher(_collection([_post(hn_id="1", points=60)]), step_id="2")

        llm = _StubLLM(['{"summary": "A compact summary of the article."}'])
        summarizer = HNSummarizer(handler, llm=llm, retries=2)
        out = summarizer(fetched, step_id="3")
        assert out.posts[0].description == "A compact summary of the article."
        # fetched content is not stored on the post (only title+description remain)
        assert "hello" not in out.posts[0].model_dump_json()

    def test_prompt_contains_title_and_content(self, monkeypatch):
        # T15 (R7): prompt is exactly title + fetched content.
        from mourat.processors.hn_summarizer import HNSummarizer

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content",
            lambda html: "hello world",
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200, "hello world")])
        fetched = HNContentFetcher(handler, http_session=session, proxy=None)(
            _collection([_post(hn_id="1", title="My title", points=60)]), step_id="2"
        )
        llm = _StubLLM(['{"summary": "summary"}'])
        HNSummarizer(handler, llm=llm, retries=1)(fetched, step_id="3")
        assert "My title" in llm.prompts[0]
        assert "hello world" in llm.prompts[0]

    def test_retry_then_drop(self, monkeypatch):
        # T14 (R7, B6): retries configurable; drop after exhaustion; success on retry N honored.
        from mourat.processors.hn_summarizer import HNSummarizer

        monkeypatch.setattr(
            "mourat.processors.hn_fetcher._extract_main_content", lambda html: "t"
        )
        handler = _CapturingHandler()
        session = _FakeSession([_FakeResponse(200, "hello world")])
        fetched = HNContentFetcher(handler, http_session=session, proxy=None)(
            _collection([_post(hn_id="1", points=60), _post(hn_id="2", points=61)]),
            step_id="2",
        )
        # post 1: fails twice then succeeds (retries=2); post 2: always fails (retries=1)
        llm = _StubLLM(["bad", "bad", '{"summary": "good summary"}', "bad"])
        out = HNSummarizer(handler, llm=llm, retries=2)(fetched, step_id="3")
        assert [p.hn_id for p in out.posts] == ["1"]
        assert out.posts[0].description == "good summary"
