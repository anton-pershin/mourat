"""Page fetching and main-content extraction."""

from __future__ import annotations

import logging

import requests
from trafilatura import extract

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def extract_main_content(html: str) -> str | None:
    """Extract main article content from HTML; None when nothing extractable."""
    return extract(html, include_comments=False, include_tables=True)


class PageFetcher:
    """Fetches a page URL via the configured session and extracts content.

    Returns None when the fetch fails, extraction fails, or nothing is
    extractable; the caller decides how to treat a None result. The
    extractor is injectable (defaults to `extract_main_content`).
    """

    def __init__(
        self,
        http_session=None,
        proxy: str | None = None,
        timeout: int = 30,
        extractor=extract_main_content,
    ) -> None:
        self.session = http_session if http_session is not None else requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.timeout = timeout
        self.extractor = extractor

    def fetch(self, url: str) -> str | None:
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("fetch failed for %s: %s", url, exc)
            return None
        try:
            content = self.extractor(resp.text)
        except Exception:
            logger.exception("content extraction failed unexpectedly for %s", url)
            return None
        if not content:
            logger.debug("fetch yielded no content for %s", url)
            return None
        return content
