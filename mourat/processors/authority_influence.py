"""Authority influence assessment: Function adapter over comourat."""

from __future__ import annotations

import logging
import time

from comourat.influence.authority import (
    DEFAULT_SYSTEM_PROMPT,
    AuthorityInfluenceAssessorCore,
    AuthorityResult,
    AuthorityVerdict,
)
from comourat.models import PaperCandidateCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

__all__ = ["AuthorityInfluenceAssessor", "AuthorityResult", "AuthorityVerdict"]


class AuthorityInfluenceAssessor(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Writes the path's influence_score onto candidates (R4, option (b)).

    Pass-through by count: every candidate is returned with
    `influence_score` filled (0 when unmeasurable or unassessable). Which
    measure produced the score is decided per path via the DB writer's
    `influence_metric_id` ("authority" on this path).
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model,
        batch_size: int = 25,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        request_retries: int = 2,
        request_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.core = AuthorityInfluenceAssessorCore(
            model=model,
            batch_size=batch_size,
            system_prompt=system_prompt,
            request_retries=request_retries,
            request_retry_delay_seconds=request_retry_delay_seconds,
        )
        self.core.time = time  # patch point: tests patch time.sleep here
        self.core.logger = logger  # patch point: tests patch logger.warning here
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        # Affiliations are a mourat-side in-flight field the neutral comourat
        # model cannot carry; bridge them to the core by paper index.
        affiliations = {
            i: affs
            for i, paper in enumerate(data.papers)
            if (affs := getattr(paper, "affiliations", None))
        }
        output, lines = self.core.assess(data, affiliations)
        return output, "\n".join(lines)
