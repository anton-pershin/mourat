"""Content item relevance scoring using pydantic-ai.

One scoring implementation shared by the post and paper collection paths:
an item-agnostic core scores any item rendered as a title, a body text and
optional context points; thin domain bindings map domain collections onto the
neutral shape and back, mapping results positionally.
"""

from __future__ import annotations

import json
import logging
import time

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models import Model

from mourat.base import Function
from mourat.data_models import (
    ContentItemScoringInput,
    ContentItemScoringInputCollection,
    EnrichedRedditPost,
    EnrichedRedditPostCollection,
    PaperCandidate,
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
    ScoredPaper,
    ScoredPaperCollection,
    ScoredRedditPost,
    ScoredRedditPostCollection,
    ScoreEntry,
    ScoringResult,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_title_preview

logger = logging.getLogger(__name__)


def _candidate_as_resolved(candidate: PaperCandidate) -> ResolvedPaper:
    """Wrap an unresolved candidate as a ResolvedPaper for the scorer's output.

    Interim shape used only on the discovery-only path: the agent's description
    stands in as the abstract (spec section 5.3's forward-compatibility note),
    and every resolved/in-flight field stays at its default until resolution
    populates it.
    """
    return ResolvedPaper(
        title=candidate.title,
        abstract=candidate.description,
        authors=candidate.authors,
        url=candidate.urls_seen[0] if candidate.urls_seen else "",
        resolution_status="candidate",
    )


SYSTEM_PROMPT = """\
You are a research relevance scorer. Given a content item and a list of research questions (RQs),
technical challenges (TCs), research topics, and constraints, score how relevant the item is to each
one on a scale of 0-100.

For each RQ, TC, topic, and constraint:
- Provide a relevance score (0-100, where 0 = completely irrelevant, 100 = directly addresses the RQ/TC/topic).
- Provide a brief justification for the score.

Be strict with scoring — most items will score below 50.
Return only scores for items that are actually relevant to the item (score > 0).
"""


def _build_scoring_prompt(
    title: str,
    body_text: str,
    context_points: list[str] | None = None,
    rq_list: list[dict] | None = None,
    tc_list: list[dict] | None = None,
    topic_list: list[dict] | None = None,
    constraint_list: list[dict] | None = None,
) -> str:
    """Build the scoring prompt for a single item.

    Composes the item title and body text followed by the numbered
    context points (when any are present), then the research attributes.
    """
    lines = [
        "Score the following content item against the research attributes below.",
        "",
        f"Title: {title}",
        f"Content: {body_text or '(no text)'}",
    ]

    if context_points:
        lines.append("")
        lines.append("Additional context:")
        for i, point in enumerate(context_points, 1):
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


class _ScoredNeutral:
    """One scored neutral item: the input plus its validated scores."""

    def __init__(
        self,
        item: ContentItemScoringInput,
        relevance_scores: list[ScoreEntry],
        filtering_score: float,
    ) -> None:
        self.item = item
        self.relevance_scores = relevance_scores
        self.filtering_score = filtering_score


class ContentItemScorer(
    Function[ContentItemScoringInputCollection, ContentItemScoringInputCollection]
):
    """Scores neutral content items against RQs, TCs, topics, and constraints.

    `constraints_contribute_to_filtering_score` decides whether constraint
    scores count toward `filtering_score`: false for post collection (a post
    may be relevant to a TC while failing constraints), true for paper
    collection (constraints are supplied deliberately per input, so a paper
    satisfying none of them is not a wanted result).

    The most recent `_run` leaves `self.scored_items` populated so a domain
    binding can map scores back onto its domain objects positionally.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        constraint_list: list[dict] | None = None,
        constraints_contribute_to_filtering_score: bool = False,
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
        self.constraints_contribute_to_filtering_score = (
            constraints_contribute_to_filtering_score
        )
        self.scored_items: list[_ScoredNeutral] = []
        self.valid_id_type_pairs = [
            (entity["id"], entity["type"])
            for entity in sum(
                [self.rq_list, self.tc_list, self.topic_list, self.constraint_list],
                start=[],
            )
        ]
        super().__init__(monitoring_handler)

    def _score_one(
        self, item: ContentItemScoringInput
    ) -> tuple[list[ScoreEntry], float]:
        """Score one neutral item; returns validated entries and filtering score."""
        prompt = _build_scoring_prompt(
            item.title,
            item.body_text,
            item.context_points,
            self.rq_list,
            self.tc_list,
            self.topic_list,
            self.constraint_list,
        )
        run_result: AgentRunResult = self.agent.run_sync(prompt)
        result: ScoringResult = run_result.output
        relevance_scores = [
            ScoreEntry.model_validate(e)
            for e in result.scores
            if any((e.id, e.type) == p for p in self.valid_id_type_pairs)
        ]
        score_pool = (
            relevance_scores
            if self.constraints_contribute_to_filtering_score
            else [e for e in relevance_scores if e.type != "constraint"]
        )
        filtering_score = max((e.score for e in score_pool), default=0)
        return relevance_scores, float(filtering_score)

    def _run(
        self, data: ContentItemScoringInputCollection
    ) -> tuple[ContentItemScoringInputCollection, str]:
        monitoring_lines = []
        self.scored_items = []

        total_items = len(data.items)
        for i, item in enumerate(data.items):
            t_item = time.monotonic()
            relevance_scores, filtering_score = self._score_one(item)
            self.scored_items.append(
                _ScoredNeutral(item, relevance_scores, filtering_score)
            )

            title_preview = to_title_preview(item.title)
            logger.debug(
                "scored '%s' (%d/%d) | %.2fs",
                title_preview,
                i + 1,
                total_items,
                time.monotonic() - t_item,
            )

            # Convert to dicts for monitoring output (json serializable)
            score_dicts = [s.model_dump() for s in relevance_scores]

            monitoring_lines.append(
                f"### {item.title}\n"
                f"Filtering score: {filtering_score}\n"
                f"Scores: {json.dumps(score_dicts, indent=2)}\n"
            )

        output = ContentItemScoringInputCollection(
            items=[s.item for s in self.scored_items]
        )
        text_for_monitoring = "\n---\n".join(monitoring_lines)
        return output, text_for_monitoring


class PostContentItemScorer(
    Function[EnrichedRedditPostCollection, ScoredRedditPostCollection]
):
    """Scores enriched Reddit posts via the shared content item scorer."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        constraint_list: list[dict] | None = None,
        constraints_contribute_to_filtering_score: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
    ) -> None:
        self.core = ContentItemScorer(
            monitoring_handler,
            model=model,
            rq_list=rq_list,
            tc_list=tc_list,
            topic_list=topic_list,
            constraint_list=constraint_list,
            constraints_contribute_to_filtering_score=(
                constraints_contribute_to_filtering_score
            ),
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
        )
        super().__init__(monitoring_handler)

    def _run(
        self, data: EnrichedRedditPostCollection
    ) -> tuple[ScoredRedditPostCollection, str]:
        neutral = ContentItemScoringInputCollection(
            items=[
                ContentItemScoringInput(
                    id=ep.post.submission_id,
                    title=ep.post.title,
                    body_text=ep.post.text or "",
                    context_points=ep.additional_context,
                )
                for ep in data.posts
            ]
        )
        _, text_for_monitoring = self.core._run(neutral)

        scored_posts = [
            ScoredRedditPost(
                post=ep.post,
                additional_context=ep.additional_context,
                relevance_scores=scored.relevance_scores,
                filtering_score=scored.filtering_score,
            )
            for ep, scored in zip(data.posts, self.core.scored_items, strict=True)
        ]
        output = ScoredRedditPostCollection(posts=scored_posts)
        return output, text_for_monitoring


class PaperContentItemScorer(Function[ResolvedPaperCollection, ScoredPaperCollection]):
    """Scores resolved papers via the shared content item scorer."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        constraint_list: list[dict] | None = None,
        constraints_contribute_to_filtering_score: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
        model_settings: dict | None = None,
        retries: int | None = None,
    ) -> None:
        self.core = ContentItemScorer(
            monitoring_handler,
            model=model,
            rq_list=rq_list,
            tc_list=tc_list,
            topic_list=topic_list,
            constraint_list=constraint_list,
            constraints_contribute_to_filtering_score=(
                constraints_contribute_to_filtering_score
            ),
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
        )
        super().__init__(monitoring_handler)

    def _run(self, data: ResolvedPaperCollection) -> tuple[ScoredPaperCollection, str]:
        neutral = ContentItemScoringInputCollection(
            items=[
                ContentItemScoringInput(
                    id=f"paper_{i}",
                    title=p.title,
                    body_text=p.abstract,
                    context_points=[],
                )
                for i, p in enumerate(data.papers, 1)
            ]
        )
        _, text_for_monitoring = self.core._run(neutral)

        scored_papers = [
            ScoredPaper(
                paper=resolved_paper,
                relevance_scores=scored.relevance_scores,
                filtering_score=scored.filtering_score,
            )
            for resolved_paper, scored in zip(
                data.papers, self.core.scored_items, strict=True
            )
        ]
        output = ScoredPaperCollection(papers=scored_papers)
        return output, text_for_monitoring
