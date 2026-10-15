"""HN summarizer: one LLM call per post producing a compact description."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You summarize HackerNews-linked articles for a research-assistant pipeline.

Given the post title and the article content, produce a COMPACT summary of
the article: 2-4 self-contained sentences describing what it is about.
Do not add information that is not in the content. Return JSON:
{"summary": "..."}
"""


class _SummaryOutput(BaseModel):
    """Structured output of the summarizer LLM call."""

    summary: str


class _StringLLMAdapter:
    """Adapts a string-returning stub (or any object with .ask) for tests."""

    def __init__(self, llm):
        self.llm = llm

    def ask(self, prompt):
        return self.llm.ask(prompt)


class HNSummarizer(Function[HNPostCollection, HNPostCollection]):
    """Produces a compact description for each HN post via one LLM call.

    Unparseable output is retried up to `retries` times; after exhaustion the
    post is dropped with a logged warning. The fetched content is used only
    to build the prompt and is never stored on the output model.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        llm,
        retries: int = 1,
        system_prompt: str = SYSTEM_PROMPT,
        text_for_monitoring_template: str = (
            "HN summarizer: {before} in, {after} out, {dropped} dropped"
        ),
    ) -> None:
        self.llm = _StringLLMAdapter(llm)
        self.retries = retries
        self.system_prompt = system_prompt
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _build_prompt(self, post) -> str:
        return f"Title: {post.title}\n\nContent:\n{post.text}"

    def _summarize_one(self, post) -> str:
        prompt = self._build_prompt(post)
        last_error = None
        for attempt in range(self.retries + 1):
            raw = self.llm.ask(prompt)
            try:
                parsed = json.loads(raw)
                summary = parsed["summary"]
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("empty summary")
                return summary.strip()
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.debug(
                    "HN summarizer attempt %d failed for %s: %s",
                    attempt + 1,
                    post.hn_id,
                    exc,
                )
        raise RuntimeError(
            f"HN summarizer failed for post {post.hn_id} after "
            f"{self.retries + 1} attempts: {last_error}"
        )

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        kept = []
        dropped = 0
        for post in data.posts:
            try:
                post.description = self._summarize_one(post)
            except RuntimeError as exc:
                logger.warning("%s", exc)
                dropped += 1
                continue
            kept.append(post)
        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), dropped=dropped
        )
        return HNPostCollection(posts=kept), text_for_monitoring
