"""HN content fetcher: fetch the linked page and extract main content."""

from __future__ import annotations

import logging

import requests
from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler
from trafilatura import extract

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def _extract_main_content(html: str) -> str | None:
    """Extract main article content from HTML; None when nothing extractable."""
    return extract(html, include_comments=False, include_tables=True)


class HNContentFetcher(Function[HNPostCollection, HNPostCollection]):
    """Fetches each HN post's URL via the configured proxy and extracts content.

    A post whose URL fails to fetch (HTTP error, timeout, connection error)
    or whose extracted content is empty is DROPPED from the collection.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_session=None,
        proxy: str | None = None,
        timeout: int = 30,
        text_for_monitoring_template: str = (
            "HN content fetcher: {before} in, {after} out, {dropped} dropped"
        ),
    ) -> None:
        self.session = http_session if http_session is not None else requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.timeout = timeout
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        kept = []
        dropped = 0
        for post in data.posts:
            try:
                resp = self.session.get(
                    post.url,
                    timeout=self.timeout,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.debug(
                    "HN fetch failed for %s (%s): %s", post.hn_id, post.url, exc
                )
                dropped += 1
                continue
            try:
                content = _extract_main_content(resp.text)
            except Exception:
                logger.exception(
                    "HN content extraction failed unexpectedly for %s (%s)",
                    post.hn_id,
                    post.url,
                )
                dropped += 1
                continue
            if not content:
                logger.debug(
                    "HN fetch yielded no content for %s (%s)", post.hn_id, post.url
                )
                dropped += 1
                continue
            post.text = content
            kept.append(post)

        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), dropped=dropped
        )
        return HNPostCollection(posts=kept), text_for_monitoring
