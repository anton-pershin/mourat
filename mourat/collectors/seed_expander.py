"""Seed expansion: Function adapter over comourat."""

from __future__ import annotations

from typing import Any

from comourat.collectors.seed_expander import (
    GEN_BACKWARD,
    GEN_FORWARD,
    GEN_SEARCH,
    SeedExpanderCore,
)
from comourat.clients.paper_graph import PaperGraphClient
from comourat.models import PaperCandidateCollection
from mourat.base import Function
from mourat.data_models import PaperCandidate, SeedCollection
from mourat.monitoring import MonitoringHandler


class SeedExpander(Function[SeedCollection, PaperCandidateCollection]):
    """Run forward, relevance-search, and backward graph generators."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        paper_graph_client: PaperGraphClient | None = None,
        forward_budget: int = 100,
        search_budget: int = 25,
        backward_budget: int = 100,
        openalex_client: PaperGraphClient | None = None,
    ) -> None:
        self.core = SeedExpanderCore(
            paper_graph_client=paper_graph_client,
            forward_budget=forward_budget,
            search_budget=search_budget,
            backward_budget=backward_budget,
            openalex_client=openalex_client,
            candidate_model=PaperCandidate,
        )
        super().__init__(monitoring_handler)

    def _run(self, data: SeedCollection) -> tuple[PaperCandidateCollection, str]:
        output, lines = self.core.expand(data.seeds)
        return output, "\n".join(lines)
