"""ArxivPdfVerifier: FR4 — a paper is verified when arXiv has its preprint.

The verification is a single arXiv title search per paper: if arXiv has a
preprint whose title matches the resolved title (cross-checked, threshold as
configured), the paper's url is the arXiv PDF url for the found id; if not,
the paper is left unverified with the recorded reason. No OpenAlex-derived
arXiv ids and no ranged PDF probe are involved — an id found via the arXiv
search guarantees the PDF url form is valid.
"""

import logging
import time

from mourat.base import Function
from mourat.clients.arxiv import ARXIV_PDF_URL_TEMPLATE, ArxivClient
from mourat.data_models import ResolvedPaperCollection
from mourat.monitoring import MonitoringHandler
from mourat.utils.similarity import titles_match

logger = logging.getLogger(__name__)

REASON_NOT_ON_ARXIV = "not_on_arxiv"
REASON_TITLE_MISMATCH = "arxiv_title_mismatch"
REASON_API_ERROR = "api_error"


class ArxivPdfVerifier(Function[ResolvedPaperCollection, ResolvedPaperCollection]):
    """Sets `url` only when an arXiv preprint matches the paper's title.

    For each resolved paper, `search_by_title` queries arXiv for the paper's
    own title. Verification requires: a preprint found AND its title matching
    the resolved title at the configured threshold. On success the paper's url
    is the arXiv PDF url for the found id; on any failed leg the url stays
    empty and the reason is recorded (FR4: recorded reason, never an
    empty-but-unexplained url). A paper whose arXiv request fails after
    retries is recorded as `api_error` and never aborts the whole step.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        arxiv_client: ArxivClient,
        title_similarity_threshold: float,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.title_similarity_threshold = title_similarity_threshold
        super().__init__(monitoring_handler)

    def _verify_one(self, item) -> tuple[str | None, str | None]:
        """Return (url, reason): url set on success, reason on failure."""
        found = self.arxiv_client.search_by_title(item.title)
        if found is None:
            return None, REASON_NOT_ON_ARXIV
        arxiv_id, arxiv_title = found
        if not titles_match(item.title, arxiv_title, self.title_similarity_threshold):
            return None, REASON_TITLE_MISMATCH
        return ARXIV_PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id), None

    def _run(
        self, data: ResolvedPaperCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        t_start = time.monotonic()
        verified = 0
        reasons: dict[str, int] = {}
        lines: list[str] = []

        for item in data.papers:
            try:
                url, reason = self._verify_one(item)
            except Exception:
                # One paper's API failure never aborts the step (FR4 records
                # a reason for every empty url; api_error is that reason).
                logger.exception(
                    "arXiv verification request failed for '%s'", item.title
                )
                url, reason = None, REASON_API_ERROR
            if url is not None:
                item.url = url
                verified += 1
            else:
                assert reason is not None  # FR4: a reason for every empty url
                item.url = ""
                item.url_absent_reason = reason
                reasons[reason] = reasons.get(reason, 0) + 1
                lines.append(f"### URL ABSENT: {item.title}\nReason: {reason}\n")

        lines.insert(
            0,
            f"Papers in: {len(data.papers)}, verified: {verified}, "
            f"url absent: {len(data.papers) - verified}"
            + (f" | reasons: {reasons}" if reasons else ""),
        )
        logger.debug(
            "verified %d/%d papers | %.2fs",
            verified,
            len(data.papers),
            time.monotonic() - t_start,
        )
        return data, "\n".join(lines)
