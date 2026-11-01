"""Paper discovery: Function adapter over comourat."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model

from comourat.collectors.paper_discoverer import (
    SYSTEM_PROMPT,
    PaperDiscovererCore,
)
from comourat.models import PaperCandidateCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

__all__ = ["PaperDiscoverer", "SYSTEM_PROMPT"]


class PaperDiscoverer(Function[Any, PaperCandidateCollection]):
    """Discovers candidate papers with a web-search-equipped LLM agent.

    The agent is the authority on *which* papers are worth collecting and on
    nothing else: its output model carries title, authors, description and
    urls only — no identifier fields exist on `PaperCandidate`, so any
    identifier the model emits is rejected by validation rather than dropped
    by a discard step (spec decision 4.3, enforced structurally).
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        attribute_description: str,
        constraints_description: str = "",
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
        request_limit: int = 30,
        tool_calls_limit: int = 15,
        tool_call_rejection_buffer: int = 4,
    ) -> None:
        self.core = PaperDiscovererCore(
            model=model,
            attribute_description=attribute_description,
            constraints_description=constraints_description,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
            request_limit=request_limit,
            tool_calls_limit=tool_calls_limit,
            tool_call_rejection_buffer=tool_call_rejection_buffer,
        )
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[PaperCandidateCollection, str]:
        try:
            result, lines = self.core.discover()
        except UsageLimitExceeded:
            logger.exception("Tool or request usage limit exceeded during discovery")
            raise
        return result, "\n".join(lines)
