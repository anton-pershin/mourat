"""PaperResolver: candidates -> resolved papers with canonical metadata.

Per FR1: lookup by arXiv id when a candidate url carries one, else by title
via OpenAlex; a resolved record whose title does not match the claimed title
above the configured threshold is rejected wholesale — no field from a
mismatched record reaches the output. Candidates no API indexes are marked
unresolved and dropped (FR5), with the reason in monitoring.
"""

import logging
import re
import time
from typing import Any

from mourat.base import Function
from mourat.clients.arxiv import ArxivClient
from mourat.clients.openalex import OpenAlexClient
from mourat.data_models import (
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_title_preview
from mourat.utils.similarity import title_similarity, titles_match

logger = logging.getLogger(__name__)

# Matches an arXiv id inside a url: abs/<id>, pdf/<id>, arxiv.org/<id>.
_ARXIV_ID_IN_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?/?"
)

STATUS_RESOLVED = "resolved"
STATUS_UNRESOLVED_NO_MATCH = "unresolved_title_mismatch"
STATUS_UNRESOLVED_NOT_FOUND = "unresolved_not_found"


def _extract_arxiv_id(urls: list[str]) -> str | None:
    """The first arXiv id found in the candidate's as-encountered urls."""
    for url in urls:
        match = _ARXIV_ID_IN_URL.search(url)
        if match:
            return match.group(1)
    return None


def _abstract_from_inverted_index(inv: dict[str, list[int]] | None) -> str:
    """Reconstruct the abstract text from OpenAlex's inverted index."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(word for _, word in positions)


class PaperResolver(Function[PaperCandidateCollection, ResolvedPaperCollection]):
    """Resolves candidates into canonical records; drops unresolved ones.

    A plain (non-`Function`) `OpenAlexClient` and `ArxivClient` do the API
    work; this component owns the resolution policy (FR1, FR2, FR5). doi,
    arXiv id and work id are carried in-flight on the resolved record and
    reported in monitoring, never persisted (FR2). The arXiv id comes only
    from the candidate's as-encountered urls (web-search path); arXiv
    verification of seed-expanded candidates (which carry no urls) is
    done later by title search in ArxivPdfVerifier, never here.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        openalex_client: OpenAlexClient,
        arxiv_client: ArxivClient,
        title_similarity_threshold: float = 0.9,
    ) -> None:
        self.openalex = openalex_client
        self.arxiv = arxiv_client
        self.title_similarity_threshold = title_similarity_threshold
        super().__init__(monitoring_handler)

    def _resolve_work(
        self, claimed_title: str, arxiv_id: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Find the OpenAlex work record for a candidate.

        Returns (work_record, None) on success, (None, reason) on failure.
        With an arXiv id: confirm the id's own title against the claimed
        title via arXiv, then take OpenAlex's top title-search hit — a
        candidate whose id names a different paper is a mismatch (FR1).
        Without one: OpenAlex title search decides, same threshold.
        """
        if arxiv_id is not None:
            arxiv_title = self.arxiv.get_title_by_id(arxiv_id)
            if arxiv_title is None:
                return None, f"arxiv_id_{arxiv_id}_not_found"
            if not titles_match(
                claimed_title, arxiv_title, self.title_similarity_threshold
            ):
                return None, "arxiv_title_mismatch"
        envelope = self.openalex.search_works_by_title(claimed_title)
        results = envelope.get("results") or []
        if not results:
            return None, "no_work_found"
        best = results[0]
        work_title = best.get("display_name") or best.get("title") or ""
        if not titles_match(claimed_title, work_title, self.title_similarity_threshold):
            return None, "openalex_title_mismatch"
        return best, None

    def _resolve_one(self, candidate) -> tuple[ResolvedPaper | None, str]:
        """Resolve one candidate; returns (resolved paper or None, reason)."""
        arxiv_id = _extract_arxiv_id(candidate.urls_seen)
        try:
            work, reason = self._resolve_work(candidate.title, arxiv_id)
        except Exception:
            logger.exception(
                "resolution request failed for candidate '%s'", candidate.title
            )
            work, reason = None, "api_error"
        if work is None:
            assert reason is not None
            return None, reason

        title = work.get("display_name") or work.get("title") or candidate.title
        doi = work.get("doi") or None
        work_id = work.get("id") or None
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        authors = [name for name in authors if name]
        pub_date = work.get("publication_date") or None
        abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        resolved = ResolvedPaper(
            title=title,
            abstract=abstract,
            authors=authors,
            publication_date=pub_date,
            url="",  # set by ArxivPdfVerifier (FR4), stays empty until then
            provenance=candidate.provenance,  # carried through (FR5)
            doi=doi,
            arxiv_id=arxiv_id,
            work_id=work_id,
            influence_fwci=work.get("fwci"),
            influence_cited_by_count=work.get("cited_by_count"),
            resolution_status=STATUS_RESOLVED,
        )
        return resolved, "resolved"

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        t_start = time.monotonic()
        resolved: list[ResolvedPaper] = []
        dropped: list[tuple[str, str]] = []
        total_candidates = len(data.papers)
        for i, candidate in enumerate(data.papers):
            item, reason = self._resolve_one(candidate)
            title_preview = to_title_preview(candidate.title)
            if item is None:
                logger.error("unresolved candidate '%s': %s", candidate.title, reason)
                dropped.append((candidate.title, reason))
            else:
                logger.debug(
                    "resolved '%s' (%d/%d)",
                    title_preview,
                    i + 1,
                    total_candidates,
                )
                resolved.append(item)

        lines = [
            f"Candidates in: {len(data.papers)}, resolved: {len(resolved)}, "
            f"unresolved (dropped): {len(dropped)}"
        ]
        for title, reason in dropped:
            lines.append(f"### UNRESOLVED: {title}\nReason: {reason}\n")
        for item in resolved:
            ids = (
                f"doi={item.doi or '-'} arxiv={item.arxiv_id or '-'} "
                f"work={item.work_id or '-'}"
            )
            lines.append(f"### {item.title}\nIdentifiers: {ids}\n")
        logger.debug(
            "resolved %d/%d candidates | %.2fs",
            len(resolved),
            len(data.papers),
            time.monotonic() - t_start,
        )
        return ResolvedPaperCollection(papers=resolved), "\n".join(lines)
