"""InfluenceAssessor: normalised influence onto the 0-100 schema range.

Function adapter over comourat's citation-influence assessor. The mapping
(fwci preferred, citations-per-year fallback, breakpoint interpolation)
lives in comourat.
"""

from __future__ import annotations

from comourat.influence.citation import (
    MEASURE_CITATIONS_PER_YEAR,
    MEASURE_FWCI,
    CitationInfluenceAssessor,
    _interpolate,
    _years_since,
)
from mourat.base import Function
from mourat.data_models import ResolvedPaperCollection
from mourat.monitoring import MonitoringHandler

__all__ = [
    "InfluenceAssessor",
    "MEASURE_CITATIONS_PER_YEAR",
    "MEASURE_FWCI",
    "_interpolate",
    "_years_since",
]


class InfluenceAssessor(Function[ResolvedPaperCollection, ResolvedPaperCollection]):
    """Assesses field- and age-normalised influence for resolved papers.

    Pass-through by count: every resolved paper leaves with an
    `influence_score` in 0-100 and `influence_measure_used` set — or, when no
    measure is computable, both stay None with a warning (a paper the resolver
    accepted but that carries no citation signal at all).
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        fwci_breakpoints: list[tuple[float, int]],
        citations_per_year_breakpoints: list[tuple[float, int]],
    ) -> None:
        self.core = CitationInfluenceAssessor(
            fwci_breakpoints=fwci_breakpoints,
            citations_per_year_breakpoints=citations_per_year_breakpoints,
        )
        super().__init__(monitoring_handler)

    def _assess_one(self, item, today) -> tuple[int | None, str | None]:
        return self.core._assess_one(item, today)

    def _run(
        self, data: ResolvedPaperCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        _, lines = self.core.assess(data.papers)
        return data, "\n".join(lines)
