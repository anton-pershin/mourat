"""Post relevance scoring module using pydantic-ai."""

from __future__ import annotations

import json

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models import Model
from rich.progress import track

from mourat.base import Function
from mourat.data_models import (
    EnrichedRedditPostCollection,
    ScoredRedditPost,
    ScoredRedditPostCollection,
    ScoreEntry,
    ScoringResult,
)
from mourat.monitoring import MonitoringHandler

SYSTEM_PROMPT = """\
You are a research relevance scorer. Given a content item (post) and a list of research questions (RQs),
technical challenges (TCs), and research topics, score how relevant the post is to each one on a scale of 0-100.

For each RQ, TC, and topic:
- Provide a relevance score (0-100, where 0 = completely irrelevant, 100 = directly addresses the RQ/TC/topic).
- Provide a brief justification for the score.

Be strict with scoring — most posts will score below 50.
Return only scores for items that are actually relevant to the post (score > 0).
"""


def _build_scoring_prompt(
    post_info,
    enriched_text: str = "",
    rq_list: list[dict] | None = None,
    tc_list: list[dict] | None = None,
    topic_list: list[dict] | None = None,
) -> str:
    """Build the scoring prompt for a single post."""
    text = enriched_text or post_info.text or "(no text)"

    lines = [
        "Score the following post against the research attributes below.",
        "",
        f"Title: {post_info.title}",
        f"Author: {post_info.author}",
        f"URL: {post_info.url}",
        f"Content: {text}",
        "",
    ]

    if rq_list:
        lines.append(f"Research Questions: {json.dumps(rq_list)}")
        lines.append("")

    if tc_list:
        lines.append(f"Technical Challenges: {json.dumps(tc_list)}")
        lines.append("")

    if topic_list:
        lines.append(f"Research Topics: {json.dumps(topic_list)}")
        lines.append("")

    lines.append(
        "Return a JSON object with a 'scores' array containing entries with 'id', 'type' ('rq'/'tc'/'topic'), 'score' (0-100), and 'justification'."
    )
    return "\n".join(lines)


class PostScorer(Function[EnrichedRedditPostCollection, ScoredRedditPostCollection]):
    """Scores enriched Reddit posts against RQs, TCs, and topics."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=ScoringResult,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
        )
        self.rq_list = rq_list or []
        self.tc_list = tc_list or []
        self.topic_list = topic_list or []
        super().__init__(monitoring_handler)

    def _run(
        self, data: EnrichedRedditPostCollection
    ) -> tuple[ScoredRedditPostCollection, str]:
        scored_posts = []
        monitoring_lines = []

        for ep in track(data.posts, description="Scoring posts"):
            prompt = _build_scoring_prompt(
                ep.post,
                ep.enrichment_summary or "",
                self.rq_list,
                self.tc_list,
                self.topic_list,
            )
            run_result: AgentRunResult = self.agent.run_sync(prompt)
            result: ScoringResult = run_result.output

            relevance_scores = [ScoreEntry.model_validate(e) for e in result.scores]
            max_score = max(
                (e.score for e in result.scores),
                default=0,
            )

            # Convert to dicts for monitoring output (json serializable)
            score_dicts = [s.model_dump() for s in relevance_scores]

            monitoring_lines.append(
                f"### {ep.post.title}\n"
                f"URL: {ep.post.url}\n"
                f"Max score: {max_score}\n"
                f"Scores: {json.dumps(score_dicts, indent=2)}\n"
            )

            scored_posts.append(
                ScoredRedditPost(
                    post=ep.post,
                    enrichment_summary=ep.enrichment_summary or "",
                    relevance_scores=relevance_scores,
                    max_score=float(max_score),
                )
            )

        output = ScoredRedditPostCollection(posts=scored_posts)
        text_for_monitoring = "\n---\n".join(monitoring_lines)
        return output, text_for_monitoring
