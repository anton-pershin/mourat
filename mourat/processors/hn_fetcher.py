"""HN content fetcher: Function adapter over comourat."""

from __future__ import annotations

from comourat.enrichment.page_fetch import PageFetcher
from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler

DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def _extract_main_content(html: str) -> str | None:
    """Patch point retained for tests: delegates to comourat's extractor."""
    from comourat.enrichment.page_fetch import extract_main_content

    return extract_main_content(html)


class HNContentFetcher(Function[HNPostCollection, HNPostCollection]):
    """Fetches each HN post's URL via the configured proxy and extracts content.

    Posts whose fetch or extraction fail or yield nothing extractable are
    dropped (debug-logged); the survivors carry the extracted content in
    their `text` field.
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
        self.core = PageFetcher(
            http_session=http_session,
            proxy=proxy,
            timeout=timeout,
            # resolved at call time so tests can patch this module's function
            extractor=lambda html: _extract_main_content(html),
        )
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        kept = []
        dropped = 0
        for post in data.posts:
            content = self.core.fetch(post.url)
            if content is None:
                dropped += 1
                continue
            post.text = content
            kept.append(post)

        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), dropped=dropped
        )
        return HNPostCollection(posts=kept), text_for_monitoring
