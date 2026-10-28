"""Structured LLM summarization agent mechanics."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from types import SimpleNamespace

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

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


def summarize_with_retries(
    agent,
    prompt: str,
    retries: int = 1,
    label: str = "",
) -> str:
    """Run one structured summarization call with bounded retries.

    Returns the stripped summary; raises RuntimeError after exhaustion.
    Failed requests or invalid outputs are retried up to `retries` times.
    One log line per attempt, labelled with `label`.
    """
    last_error = None
    for attempt in range(retries + 1):
        started_at = perf_counter()
        try:
            result = agent.run_sync(prompt)
            summary = result.output.summary
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("empty summary")
            logger.info(
                "summarize '%s' | success | attempt %d/%d | %.2fs",
                label,
                attempt + 1,
                retries + 1,
                perf_counter() - started_at,
            )
            return summary.strip()
        except Exception as exc:
            last_error = exc
            logger.warning(
                "summarize '%s' | failed | attempt %d/%d | %.2fs | %s",
                label,
                attempt + 1,
                retries + 1,
                perf_counter() - started_at,
                exc,
            )
    raise RuntimeError(
        f"summarizer failed for '{label}' after {retries + 1} attempts: {last_error}"
    )
