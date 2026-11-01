"""ArxivPdfVerifier: Function adapter over comourat verification core."""

from __future__ import annotations

import logging
import time

from comourat.clients.arxiv import ArxivClient
from comourat.models import PaperRecord
from comourat.verification.arxiv_pdf import ArxivPdfVerifierCore
from mourat.base import Function
from mourat.data_models import ResolvedPaperCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

REASON_NOT_ON_ARXIV = "not_on_arxiv"
REASON_TITLE_MISMATCH = "arxiv_title_mismatch"
REASON_API_ERROR = "api_error"


class ArxivPdfVerifier(Function[ResolvedPaperCollection, ResolvedPaperCollection]):
    """Set mourat URL fields from comourat's neutral verification outcomes."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        arxiv_client: ArxivClient,
        title_similarity_threshold: float,
    ) -> None:
        self.core = ArxivPdfVerifierCore(
            arxiv_client=arxiv_client,
            title_similarity_threshold=title_similarity_threshold,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: ResolvedPaperCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        t_start = time.monotonic()
        papers = data.papers
        records = [PaperRecord(title=paper.title) for paper in papers]
        result = self.core.verify(records)
        verified = 0
        reasons: dict[str, int] = {}
        lines: list[str] = []

        for paper, outcome in zip(papers, result.outcomes):
            if outcome.pdf_url is not None:
                paper.url = outcome.pdf_url
                verified += 1
                continue

            reason = outcome.reason or REASON_API_ERROR
            if reason == REASON_API_ERROR:
                logger.warning(
                    "arXiv verification request failed for '%s'", paper.title
                )
            paper.url = ""
            paper.url_absent_reason = reason
            reasons[reason] = reasons.get(reason, 0) + 1
            lines.append(f"### URL ABSENT: {paper.title}\nReason: {reason}\n")

        lines.insert(
            0,
            f"Papers in: {len(papers)}, verified: {verified}, "
            f"url absent: {len(papers) - verified}"
            + (f" | reasons: {reasons}" if reasons else ""),
        )
        logger.debug(
            "verified %d/%d papers | %.2fs",
            verified,
            len(papers),
            time.monotonic() - t_start,
        )
        return data, "\n".join(lines)
