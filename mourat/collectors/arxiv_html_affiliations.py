"""Author affiliation extraction from arXiv HTML renderings (spec 14, R3).

arXiv's cheap metadata surfaces (RSS feed, abs page, Atom API) carry no
affiliations. The HTML rendering at https://arxiv.org/html/<id>v1 does:
each author block may carry ltx_role_affiliation spans. Measured on recent
cs.LG submissions: 9/9 rendered, 8/9 with parseable affiliation nodes.

Deterministic fetching only — no LLM. Papers whose render is missing
(HTTP error) or carries zero affiliation nodes get affiliations=None:
an explicit "unknown" state, not an empty result (R3).
"""

import logging
import re
import time
from typing import Any

import httpx

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

# Author blocks in the HTML render look like:
# <span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">
# Affiliation: </span>University of Somewhere</span>
_AFFILIATION_SPAN_RE = re.compile(
    r'<span class="ltx_contact ltx_role_affiliation"[^>]*>.*?'
    r'<span class="ltx_contact_name">\s*Affiliation:\s*</span>'
    r"(.*?)</span>",
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_affiliations(html: str) -> dict[str, list[str]]:
    """Author -> affiliation pairs from an arXiv HTML rendering.

    The render interleaves author names and affiliation spans in document
    order: each affiliation span belongs to the most recently seen author
    name. Multiple affiliation spans per author are common (dual
    appointments) and accumulate.
    """
    # Author names appear as <span class="ltx_personname">Full Name</span>
    author_spans = re.findall(
        r'<span class="ltx_personname"[^>]*>(.*?)</span>', html, re.S
    )

    # Walk the html in order, tracking the current author.
    author_to_affs: dict[str, list[str]] = {}
    pattern = re.compile(
        r'<span class="ltx_personname"[^>]*>(.*?)</span>'
        r'|<span class="ltx_contact ltx_role_affiliation"[^>]*>.*?'
        r'<span class="ltx_contact_name">\s*Affiliation:\s*</span>'
        r"(.*?)</span>",
        re.S,
    )
    current_author: str | None = None
    for m in pattern.finditer(html):
        person, affil = m.group(1), m.group(2)
        if person is not None:
            current_author = _TAG_RE.sub(" ", person).strip()
            author_to_affs.setdefault(current_author, [])
        elif affil is not None and current_author is not None:
            aff = _TAG_RE.sub(" ", affil).strip()
            if aff and aff not in author_to_affs[current_author]:
                author_to_affs[current_author].append(aff)
    return author_to_affs


class ArxivHtmlAffiliationFetcher(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Attaches author affiliations to candidates (R3).

    Pass-through by count: every candidate is returned; the only change is
    the in-flight `affiliations` field (None when unknown). Fetching is
    paced: `request_delay_seconds` between HTML fetches.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_client: httpx.Client,
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.http_client = http_client
        self.request_delay_seconds = request_delay_seconds
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        t_start = time.monotonic()
        fetched = 0
        with_affiliations = 0
        unknown: list[str] = []
        total = len(data.papers)

        for i, paper in enumerate(data.papers, 1):
            if paper.arxiv_id is None:
                unknown.append(paper.title)
                continue
            url = f"https://arxiv.org/html/{paper.arxiv_id}v1"
            try:
                r: httpx.Response = self.http_client.get(url)
                r.raise_for_status()
                html = r.text
            except Exception as exc:
                logger.warning(
                    "affiliation fetch failed for '%s' (%s): %s",
                    paper.title,
                    url,
                    exc,
                )
                unknown.append(paper.title)
                paper.affiliations = None
                continue

            fetched += 1
            affs = _extract_affiliations(html)
            if affs:
                paper.affiliations = affs
                with_affiliations += 1
            else:
                paper.affiliations = None
                unknown.append(paper.title)

            if i < total:
                time.sleep(self.request_delay_seconds)

        lines = [
            (
                f"Papers: {total}, fetched: {fetched}, with affiliations: "
                f"{with_affiliations}, unknown: {len(unknown)}"
            ),
            "Affiliations unknown for:",
        ]
        lines.extend(f"- {t}" for t in unknown)
        return (
            data,
            "\n".join(lines) + f"\n(stage took {time.monotonic() - t_start:.1f}s)",
        )
