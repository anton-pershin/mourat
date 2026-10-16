"""HN summarizer: one structured LLM call per post."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You summarize HackerNews-linked articles for a research-assistant pipeline.

Given the post title and the article content, produce a COMPACT summary of
the article: 2-4 self-contained sentences describing what it is about.
Do not add information that is not in the content.
"""


class _SummaryOutput(BaseModel):
    """Structured output of the summarizer LLM call."""

    summary: str


class _StringLLMAgent:
    """Test adapter for a string-returning stub; production uses pydantic-ai Agent."""

    def __init__(self, llm):
        self.llm = llm

    def run_sync(self, prompt):
        raw = self.llm.ask(prompt)
        parsed = json.loads(raw)
        return SimpleNamespace(output=_SummaryOutput.model_validate(parsed))


def _create_summary_agent(
    model: Model,
    system_prompt: str = SYSTEM_PROMPT,
    model_settings: dict | None = None,
) -> Agent:
    """Create an agent with structured summary output."""
    return Agent(
        model,
        output_type=_SummaryOutput,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=0,
    )


class HNSummarizer(Function[HNPostCollection, HNPostCollection]):
    """Summarize fetched page content; discard the full text after success.

    The configured pydantic-ai Model is wrapped in an Agent with structured
    `_SummaryOutput`. Failed requests or invalid outputs are retried up to
    `retries` times; after exhaustion the post is dropped.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model | None = None,
        retries: int = 1,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        llm=None,
        agent=None,
        text_for_monitoring_template: str = (
            "HN summarizer: {before} in, {after} out, {dropped} dropped"
        ),
    ) -> None:
        if agent is not None:
            self.agent = agent
        elif model is not None:
            self.agent = _create_summary_agent(
                model,
                system_prompt=system_prompt,
                model_settings=model_settings,
            )
        elif llm is not None:
            # Retain a narrow test seam for string-returning unit-test stubs.
            self.agent = _StringLLMAgent(llm)
        else:
            raise ValueError("HNSummarizer requires a pydantic-ai model")
        self.retries = retries
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _build_prompt(self, post) -> str:
        return f"Title: {post.title}\n\nContent:\n{post.text}"

    def _summarize_one(self, post) -> str:
        prompt = self._build_prompt(post)
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                result = self.agent.run_sync(prompt)
                summary = result.output.summary
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("empty summary")
                return summary.strip()
            except Exception as exc:
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
            # The full fetched article is only a summarization input.
            post.text = ""
            kept.append(post)
        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), dropped=dropped
        )
        return HNPostCollection(posts=kept), text_for_monitoring
