"""Tests for the slop classification components (spec 04).

These tests use pydantic_ai's FunctionModel so that the verdicts returned for each
batch are controlled by the test itself. No real LLM call is made.
"""

import ast
import json

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from mourat.classifiers import PostSlopClassifier
from mourat.data_models import (
    ClassifiedRedditPost,
    ClassifiedRedditPostCollection,
    RedditPostCollection,
    RedditPostInfo,
)
from mourat.monitoring import MonitoringHandler


class DummyHandler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _make_handler() -> DummyHandler:
    return DummyHandler()


def _make_post(
    submission_id: str, title: str = "A title", text: str = "Some text"
) -> RedditPostInfo:
    return RedditPostInfo(
        subreddit="testsub",
        submission_id=submission_id,
        title=title,
        author="author",
        date="2026-08-28T00:00:00+00:00",
        url=f"https://reddit.com/r/testsub/comments/{submission_id}",
        text=text,
        score=10,
    )


def _make_collection(n: int) -> RedditPostCollection:
    return RedditPostCollection(posts=[_make_post(f"id{i}") for i in range(n)])


def _verdict_response(verdicts: list[dict]) -> FunctionModel:
    """Build a FunctionModel that returns the given verdicts for every call."""

    def model_fn(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps({"verdicts": verdicts}))])

    return FunctionModel(model_fn)


def _echo_verdict_model(calls: list, slop: bool = True) -> FunctionModel:
    """A FunctionModel that returns one slop verdict per post in each batch."""

    def model_fn(messages, info):
        calls.append(messages)
        posts = ast.literal_eval(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart(json.dumps({
            "verdicts": [
                {
                    "submission_id": p["submission_id"],
                    "is_slop": slop,
                    "justification": "x",
                }
                for p in posts
            ]
        }))])
    return FunctionModel(model_fn)


class TestPostSlopClassifier:
    def test_marks_each_post_with_binary_verdict(self):
        handler = _make_handler()
        model = _verdict_response([
            {"submission_id": "id0", "is_slop": True, "justification": "meme"},
            {"submission_id": "id1", "is_slop": False, "justification": "substantive"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(_make_collection(2), "classify")

        assert isinstance(result, ClassifiedRedditPostCollection)
        by_id = {p.post.submission_id: p for p in result.posts}
        assert by_id["id0"].is_slop is True
        assert by_id["id0"].justification == "meme"
        assert by_id["id1"].is_slop is False
        assert by_id["id1"].justification == "substantive"

    def test_output_collection_preserves_all_posts(self):
        handler = _make_handler()
        model = _verdict_response([
            {"submission_id": "id0", "is_slop": True, "justification": "meme"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(_make_collection(3), "classify")

        assert len(result.posts) == 3
        assert all(isinstance(p, ClassifiedRedditPost) for p in result.posts)

    def test_batching_splits_by_batch_size(self):
        calls = []
        model = _echo_verdict_model(calls)
        classifier = PostSlopClassifier(
            monitoring_handler=_make_handler(), model=model, batch_size=2, max_text_chars=500
        )

        result = classifier(_make_collection(5), "classify")

        assert len(calls) == 3
        by_id = {p.post.submission_id: p for p in result.posts}
        assert len(by_id) == 5
        assert all(p.is_slop for p in by_id.values())

    def test_single_batch_when_below_batch_size(self):
        calls = []

        def model_fn(messages, info):
            calls.append(messages)
            return ModelResponse(parts=[TextPart(json.dumps({"verdicts": []}))])

        model = FunctionModel(model_fn)
        classifier = PostSlopClassifier(
            monitoring_handler=_make_handler(), model=model, batch_size=10, max_text_chars=500
        )

        classifier(_make_collection(3), "classify")

        assert len(calls) == 1

    def test_verdicts_matched_by_submission_id(self):
        handler = _make_handler()
        # Verdicts returned in reverse order.
        model = _verdict_response([
            {"submission_id": "id2", "is_slop": True, "justification": "third"},
            {"submission_id": "id1", "is_slop": False, "justification": "second"},
            {"submission_id": "id0", "is_slop": True, "justification": "first"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(_make_collection(3), "classify")

        by_id = {p.post.submission_id: p for p in result.posts}
        assert by_id["id0"].is_slop is True
        assert by_id["id0"].justification == "first"
        assert by_id["id1"].is_slop is False
        assert by_id["id2"].is_slop is True

    def test_missing_verdict_fails_open(self):
        handler = _make_handler()
        model = _verdict_response([
            {"submission_id": "id0", "is_slop": True, "justification": "meme"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(_make_collection(3), "classify")

        by_id = {p.post.submission_id: p for p in result.posts}
        assert by_id["id0"].is_slop is True
        assert by_id["id1"].is_slop is False
        assert by_id["id2"].is_slop is False

    def test_model_error_fails_open(self):
        handler = _make_handler()

        calls = []

        def model_fn(messages, info):
            calls.append(messages)
            if len(calls) == 1:
                raise UnexpectedModelBehavior("bad output")
            posts = ast.literal_eval(messages[-1].parts[-1].content)
            return ModelResponse(
                parts=[TextPart(json.dumps({
                    "verdicts": [
                        {
                            "submission_id": p["submission_id"],
                            "is_slop": True,
                            "justification": "slop",
                        }
                        for p in posts
                    ]
                }))]
            )

        model = FunctionModel(model_fn)
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=2, max_text_chars=500
        )

        result = classifier(_make_collection(4), "classify")

        by_id = {p.post.submission_id: p for p in result.posts}
        # First batch (id0, id1) failed -> fail-open, not slop.
        assert by_id["id0"].is_slop is False
        assert by_id["id1"].is_slop is False
        # Second batch processed normally.
        assert by_id["id2"].is_slop is True
        assert by_id["id3"].is_slop is True
        assert len(calls) == 2

    def test_unknown_submission_id_is_ignored(self):
        handler = _make_handler()
        model = _verdict_response([
            {"submission_id": "no_such_id", "is_slop": True, "justification": "ghost"},
            {"submission_id": "id0", "is_slop": False, "justification": "fine"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(_make_collection(2), "classify")

        by_id = {p.post.submission_id: p for p in result.posts}
        assert by_id["id0"].is_slop is False
        assert by_id["id1"].is_slop is False  # no verdict -> fail-open

    def test_empty_collection_makes_no_llm_call(self):
        calls = []

        def model_fn(messages, info):
            calls.append(messages)
            return ModelResponse(parts=[TextPart(json.dumps({"verdicts": []}))])

        model = FunctionModel(model_fn)
        classifier = PostSlopClassifier(
            monitoring_handler=_make_handler(), model=model, batch_size=10, max_text_chars=500
        )

        result = classifier(RedditPostCollection(posts=[]), "classify")

        assert isinstance(result, ClassifiedRedditPostCollection)
        assert len(result.posts) == 0
        assert len(calls) == 0

    def test_monitoring_reports_counts_and_slop_posts(self):
        handler = _make_handler()
        model = _verdict_response([
            {"submission_id": "id0", "is_slop": True, "justification": "meme post"},
            {"submission_id": "id1", "is_slop": False, "justification": "good"},
        ])
        classifier = PostSlopClassifier(
            monitoring_handler=handler, model=model, batch_size=10, max_text_chars=500
        )

        posts = _make_collection(2)
        classifier(posts, "3")

        assert len(handler.calls) == 1
        step, text = handler.calls[0]
        assert step == "3"
        assert "2" in text  # in count
        assert "1" in text  # kept / dropped counts
        assert "meme post" in text  # slop justification present
        assert posts.posts[0].url in text  # dropped post details