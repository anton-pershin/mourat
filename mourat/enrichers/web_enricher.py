"""Web enrichment module — LLM agent with URL extraction and web search tools."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models import Model
from trafilatura import extract

from mourat.base import Function
from mourat.data_models import (
    EnrichedRedditPost,
    EnrichedRedditPostCollection,
    EnrichmentResult,
    RedditPostCollection,
)
from mourat.monitoring import MonitoringHandler

SYSTEM_PROMPT = """\
You are a research assistant helping to enrich social media posts with additional context.

For each post provided, your task is to:
1. Determine if the post needs enrichment (e.g., link-only post with no body text, post mentioning a topic that needs more context).
2. If enrichment is needed, use the available tools to gather relevant information:
   - `extract_url`: Fetch and extract the full content of a linked article.
   - `web_search`: Find related information online.
3. After enrichment, return the enriched text and a brief summary of what you did.

If the post already has sufficient text, you may leave it unchanged and provide a short summary.
Be concise and focused. Only enrich posts that genuinely need it.
"""


def _create_enrichment_agent(
    model: Model,
    system_prompt: str = SYSTEM_PROMPT,
    model_settings: dict | None = None,
    retries: int | None = None,
) -> Agent:
    """Create an enrichment agent with URL extraction and web search tools."""
    agent = Agent(
        model,
        output_type=EnrichmentResult,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
    )

    @agent.tool_plain
    def extract_url(url: str) -> str:
        """Extract main article content from a URL."""
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

    @agent.tool_plain
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for information related to a query."""
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
    ) -> None:
        self.agent = _create_enrichment_agent(
            model,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: RedditPostCollection
    ) -> tuple[EnrichedRedditPostCollection, str]:
        enriched_posts = []
        monitoring_lines = []
        for post_info in data.posts:
            prompt = (
                f"Enrich the following post if it needs more context:\n\n"
                f"Title: {post_info.title}\n"
                f"Author: {post_info.author}\n"
                f"URL: {post_info.url}\n"
                f"Score: {post_info.score}\n"
                f"Text: {post_info.text or '(no text)'}\n\n"
                "Use extract_url to fetch the linked article, or web_search to find related info. "
                "Return the enriched text and a brief summary of what enrichment was done."
            )

            run_result: AgentRunResult = self.agent.run_sync(prompt)
            result: EnrichmentResult = run_result.output
            enriched_posts.append(
                EnrichedRedditPost(
                    post=post_info, enrichment_summary=result.enrichment_summary
                )
            )
            monitoring_lines.append(
                f"### {post_info.title}\n"
                f"URL: {post_info.url}\n"
                f"Original text: {post_info.text or '(none)'}\n"
                f"Enriched text: {result.enriched_text}\n"
                f"Enrichment: {result.enrichment_summary}\n"
            )

        output = EnrichedRedditPostCollection(posts=enriched_posts)
        text_for_monitoring = "\n---\n".join(monitoring_lines)
        return output, text_for_monitoring
