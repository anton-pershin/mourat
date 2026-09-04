"""Web enrichment module — LLM agent with URL extraction and web search tools."""

from __future__ import annotations

import logging
import time

import requests
from bs4 import BeautifulSoup
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.tools import RunContext
from trafilatura import extract

from mourat.base import Function
from mourat.data_models import (
    EnrichedRedditPost,
    EnrichedRedditPostCollection,
    EnrichmentResult,
    RedditPostCollection,
)
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant gathering additional context for social media posts.

For each post provided, your task is to:
1. Use the available tools to gather relevant information:
   - `extract_url`: Fetch and extract the full content of a linked article.
   - `web_search`: Find related information online.
2. Return a list of additional context points: short self-contained facts
   obtained from the web that are NOT already evident in the post.
   Each point must stand alone, phrased like "X is ...".
3. If the web search and article fetch yield nothing new, return an empty list.

Return at most 5 context points.

Note that you have **strict tool usage limits**: one `web_search` call and three `extract_url` calls.
Be frugal and smart in your choices and do not attempt to bypass these limits.
"""


def _create_enrichment_agent(
    model: Model,
    system_prompt: str = SYSTEM_PROMPT,
    model_settings: dict | None = None,
    retries: int | None = None,
    tool_call_rejection_buffer: int = 4,
) -> Agent:
    """Create an enrichment agent with URL extraction and web search tools."""
    tool_time_total = 0.0

    agent = Agent(
        model,
        output_type=EnrichmentResult,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
        deps_type=dict,
    )

    def _ensure_tool_call_within_buffer(ctx: RunContext[dict]) -> str | None:
        usage: RunUsage = ctx.usage
        usage_limits: UsageLimits = ctx.usage_limits
        if usage.tool_calls + tool_call_rejection_buffer > usage_limits.tool_calls_limit:
            logger.debug(
                "attempted to exceed tool call limit",
            )
            return (
                "Tool call limit is exceeded. "
                "Do not call tools anymore. "
                "Answer the request based on the available information"
            )
        else:
            return None


    @agent.tool
    def extract_url(ctx: RunContext[dict], url: str) -> str:
        """Extract main article content from a URL."""

        if error_msg := _ensure_tool_call_within_buffer(ctx):
            return error_msg

        nonlocal tool_time_total
        t_tool = time.monotonic()
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            content = extract(
                resp.text, url=url, include_comments=False, include_tables=True
            )
            if content:
                return content
            return f"No extractable content found at {url}"
        except Exception as e:
            return f"Error extracting {url}: {e}"
        finally:
            tool_time_total += time.monotonic() - t_tool
            logger.debug(
                "extract_url '%s' | %.2fs (tool total %.2fs)",
                url,
                time.monotonic() - t_tool,
                tool_time_total,
            )

    @agent.tool
    def web_search(ctx: RunContext[dict], query: str, max_results: int = 5) -> str:
        """Search the web for information related to a query."""

        if error_msg := _ensure_tool_call_within_buffer(ctx):
            return error_msg

        nonlocal tool_time_total
        t_tool = time.monotonic()
        try:
            resp = requests.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                },
                timeout=30,
            )
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("a.result-link")
            results = []
            for a in links[:max_results]:
                title = a.get_text(strip=True)
                href = a.get("href", "")
                results.append(f"Title: {title}\nURL: {href}")

            if not results:
                return f"No results found for query: {query}"
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching for '{query}': {e}"
        finally:
            tool_time_total += time.monotonic() - t_tool
            logger.debug(
                "web_search '%s' | %.2fs (tool total %.2fs)",
                query,
                time.monotonic() - t_tool,
                tool_time_total,
            )

    agent._mourat_tool_time_total = tool_time_total  # type: ignore[attr-defined]

    return agent


class WebEnricher(Function[RedditPostCollection, EnrichedRedditPostCollection]):
    """Enriches Reddit posts by fetching additional context from the web."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
        request_limit: int = 20,
        tool_calls_limit: int = 10,
        tool_call_rejection_buffer: int = 4,
    ) -> None:
        self.agent = _create_enrichment_agent(
            model,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
            tool_call_rejection_buffer=tool_call_rejection_buffer,
        )
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
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

            tools_before = self.agent._mourat_tool_time_total  # type: ignore[attr-defined]
            t_post = time.monotonic()
            try:
                run_result: AgentRunResult = self.agent.run_sync(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=self.request_limit,
                        tool_calls_limit=self.tool_calls_limit
                    ),
                )
            except UsageLimitExceeded as e:
                logger.exception(
                    "Tool or request usage limit exceeded, skip this post: %s", e
                )
                continue
            post_s = time.monotonic() - t_post
            tools_s = (
                self.agent._mourat_tool_time_total - tools_before  # type: ignore[attr-defined]
            )
            logger.debug(
                "enrich %d/%d | '%s' | total=%.2fs tools=%.2fs llm=%.2fs",
                i,
                n_posts,
                post_info.submission_id,
                post_s,
                tools_s,
                post_s - tools_s,
            )

            result: EnrichmentResult = run_result.output
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
