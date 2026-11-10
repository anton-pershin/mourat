"""HN content fetcher: Function adapter over comourat."""

from __future__ import annotations

import logging

from comourat.enrichment.page_fetch import (
    ArticleFetchError,
    PageFetcher,
    extract_main_content,
)

from mourat.base import Function
from mourat.data_models import HNPostCollection
from mourat.monitoring import MonitoringHandler

DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

logger = logging.getLogger(__name__)


def _extract_main_content(html: str) -> str | None:
    """Patch point retained for tests: delegates to comourat's extractor."""
    return extract_main_content(html)


class HNContentFetcher(Function[HNPostCollection, HNPostCollection]):
    """Fetch plain article text for each HN post through Comourat.

    An article must have a non-empty extracted title and non-empty text. An
    article without a title is dropped as `empty_extraction`, even when its
    text is extractable. Other expected article failures are also debug-logged
    and dropped one post at a time. Unexpected adapter exceptions propagate.

    `timeout` limits the complete fetch and extraction operation. The separate
    `connection_timeout` caps each connection attempt. Survivors carry only
    `ArticleResult.text` in their `text` field.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        http_session=None,
        proxy: str | None = None,
        timeout: float = 30,
        connection_timeout: float = 10,
        text_for_monitoring_template: str = (
            "HN content fetcher: {before} in, {after} out, {dropped} dropped"
        ),
    ) -> None:
        self.core = PageFetcher(
            http_session=http_session,
            proxy=proxy,
            timeout=timeout,
            connection_timeout=connection_timeout,
            # Resolve at call time so tests can patch this module's function.
            # pylint: disable-next=unnecessary-lambda
            extractor=lambda html: _extract_main_content(html),
        )
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: HNPostCollection) -> tuple[HNPostCollection, str]:
        kept = []
        dropped = 0
        for post in data.posts:
            try:
                article = self.core.fetch(post.url)
            except ArticleFetchError as exc:
                dropped += 1
                logger.debug(
                    "article fetch failed for HN post %s | category=%s | %s",
                    post.hn_id,
                    exc.category,
                    exc,
                )
                continue
            post.text = article.text
            kept.append(post)

        text_for_monitoring = self.text_for_monitoring_template.format(
            before=len(data.posts), after=len(kept), dropped=dropped
        )
        return HNPostCollection(posts=kept), text_for_monitoring
