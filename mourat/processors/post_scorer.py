"""Post relevance scoring module using pydantic-ai."""

from __future__ import annotations

import json
import logging
import time

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models import Model

from mourat.base import Function
from mourat.data_models import (
    EnrichedRedditPostCollection,
    ScoredRedditPost,
    ScoredRedditPostCollection,
    ScoreEntry,
    ScoringResult,
)
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research relevance scorer. Given a content item (post) and a list of research questions (RQs),
technical challenges (TCs), research topics, and constraints, score how relevant the post is to each one
on a scale of 0-100.

For each RQ, TC, topic, and constraint:
- Provide a relevance score (0-100, where 0 = completely irrelevant, 100 = directly addresses the RQ/TC/topic).
- Provide a brief justification for the score.

Be strict with scoring — most posts will score below 50.
Return only scores for items that are actually relevant to the post (score > 0).
"""


def _build_scoring_prompt(
    post_info,
    additional_context: list[str] | None = None,
    rq_list: list[dict] | None = None,
    tc_list: list[dict] | None = None,
    topic_list: list[dict] | None = None,
    constraint_list: list[dict] | None = None,
) -> str:
    """Build the scoring prompt for a single post.

    Composes the original post text followed by the numbered additional
    context points (when any are present).
    """
    text = post_info.text or "(no text)"

    lines = [
        "Score the following post against the research attributes below.",
        "",
        f"Title: {post_info.title}",
        f"Author: {post_info.author}",
        f"URL: {post_info.url}",
        f"Content: {text}",
    ]

    if additional_context:
        lines.append("")
        lines.append("Additional context:")
        for i, point in enumerate(additional_context, 1):
            lines.append(f"{i}. {point}")
        lines.append("")

    if rq_list:
        lines.append(f"Research Questions: {json.dumps(rq_list)}")
        lines.append("")

    if tc_list:
        lines.append(f"Technical Challenges: {json.dumps(tc_list)}")
        lines.append("")

    if topic_list:
        lines.append(f"Research Topics: {json.dumps(topic_list)}")
        lines.append("")

    if constraint_list:
        lines.append(f"Constraints: {json.dumps(constraint_list)}")
        lines.append("")

    lines.append(
        "Return a JSON object with a 'scores' array containing entries with 'id', 'type' ('rq'/'tc'/'topic'/'constraint'), 'score' (0-100), and 'justification'."
    )
    return "\n".join(lines)


class PostScorer(Function[EnrichedRedditPostCollection, ScoredRedditPostCollection]):
    """Scores enriched Reddit posts against RQs, TCs, topics, and constraints."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        constraint_list: list[dict] | None = None,
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
        self.constraint_list = constraint_list or []
        self.valid_id_type_pairs = [
            (entity["id"], entity["type"])
            for entity in sum(
                [self.rq_list, self.tc_list, self.topic_list, self.constraint_list],
                start=[],
            )
        ]
        super().__init__(monitoring_handler)

    def _run(
        self, data: EnrichedRedditPostCollection
    ) -> tuple[ScoredRedditPostCollection, str]:
        scored_posts = []
        monitoring_lines = []

        n_posts = len(data.posts)
        for i, ep in enumerate(data.posts, 1):
            t_post = time.monotonic()
            prompt = _build_scoring_prompt(
                ep.post,
                ep.additional_context,
                self.rq_list,
                self.tc_list,
                self.topic_list,
                self.constraint_list,
            )
            run_result: AgentRunResult = self.agent.run_sync(prompt)
            result: ScoringResult = run_result.output
            logger.debug(
                "score post %d/%d | '%s' | %.2fs",
                i,
                n_posts,
                ep.post.submission_id,
                time.monotonic() - t_post,
            )

            relevance_scores = [
                ScoreEntry.model_validate(e)
                for e in result.scores
                if any((e.id, e.type) == p for p in self.valid_id_type_pairs)
            ]
            max_score = max(
                (
                    e.score
                    for e in relevance_scores
                    if e.type != "constraint"
                ),
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
                    additional_context=ep.additional_context,
                    relevance_scores=relevance_scores,
                    max_score=float(max_score),
                )
            )

        output = ScoredRedditPostCollection(posts=scored_posts)
        text_for_monitoring = "\n---\n".join(monitoring_lines)
        return output, text_for_monitoring
