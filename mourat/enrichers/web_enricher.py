"""Web enrichment: Function adapter over comourat's enrichment agent."""

from __future__ import annotations

import logging
import time

from comourat.enrichment.enricher import (
    DEFAULT_SYSTEM_PROMPT,
    EnrichmentAgent,
)
from comourat.models import EnrichmentResult
from mourat.base import Function
from mourat.data_models import (
    EnrichedRedditPost,
    EnrichedRedditPostCollection,
    RedditPostCollection,
)
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


class WebEnricher(Function[RedditPostCollection, EnrichedRedditPostCollection]):
    """Enriches Reddit posts by fetching additional context from the web."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
        request_limit: int = 20,
        tool_calls_limit: int = 10,
        tool_call_rejection_buffer: int = 4,
    ) -> None:
        self.core = EnrichmentAgent(
            model=model,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
            request_limit=request_limit,
            tool_calls_limit=tool_calls_limit,
            tool_call_rejection_buffer=tool_call_rejection_buffer,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: RedditPostCollection
    ) -> tuple[EnrichedRedditPostCollection, str]:
        enriched_posts = []
        monitoring_lines = []
        n_posts = len(data.posts)
        for i, post_info in enumerate(data.posts, 1):
            prompt = (
                f"Gather additional context for the following post:\n\n"
                f"Title: {post_info.title}\n"
                f"Author: {post_info.author}\n"
                f"URL: {post_info.url}\n"
                f"Score: {post_info.score}\n"
                f"Text: {post_info.text or '(no text)'}\n\n"
                "Use extract_url to fetch the linked article, or web_search to find related info. "
                "Return the list of additional context points (facts not already evident in the post)."
            )

            t_post = time.monotonic()
            try:
                result: EnrichmentResult = self.core.enrich(prompt)
            except Exception as exc:
                # UsageLimitExceeded and transport failures skip this post.
                logger.exception(
                    "enrichment failed for post %s: %s", post_info.submission_id, exc
                )
                continue
            post_s = time.monotonic() - t_post
            logger.debug(
                "enrich %d/%d | '%s' | total=%.2fs",
                i,
                n_posts,
                post_info.submission_id,
                post_s,
            )

            enriched_posts.append(
                EnrichedRedditPost(
                    post=post_info,
                    additional_context=result.additional_context,
                )
            )
            context_lines = "\n".join(
                f"{i}. {point}" for i, point in enumerate(result.additional_context, 1)
            )
            monitoring_lines.append(
                f"### {post_info.title}\n"
                f"URL: {post_info.url}\n"
                f"Original text: {post_info.text or '(none)'}\n"
                f"Additional context:\n{context_lines}\n"
            )

        output = EnrichedRedditPostCollection(posts=enriched_posts)
        text_for_monitoring = "\n---\n".join(monitoring_lines)
        return output, text_for_monitoring
