"""Classifier modules using pydantic_ai."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import pydantic_ai
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.exceptions import UnexpectedModelBehavior

from mourat.base import Function
from mourat.data_models import (
    ClassifiedRedditPost,
    ClassifiedRedditPostCollection,
    PaperInfoCollection,
    RedditPostCollection,
    SlopClassificationResult,
)
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import to_text_description

logger = logging.getLogger(__name__)

SLOP_SYSTEM_PROMPT = """\
You are a content-quality classifier. You are given a batch of social media posts as a
JSON array; each entry has a "submission_id", "subreddit", "title" and "text".

For each post decide whether it bears any meaningful content worth deeper analysis.
Mark a post as slop (is_slop=true) if it carries no substantive information: memes and
jokes, low-effort one-liners, vents, karma farming, self-promotion without content,
bare link drops with no commentary, support/troubleshooting requests, bot-generated
posts, community meta threads.

Mark a post as not slop (is_slop=false) if it contains substantive information,
arguments, experience reports, technical details, or a well-formed discussion prompt —
even if it is short or informal.

Return one verdict per post in the batch, each with the post's "submission_id", the
boolean "is_slop" and a one-sentence "justification".
"""


class BinaryPaperClassifier(Function[PaperInfoCollection, PaperInfoCollection]):
    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        topic_name: str,
        user_prompt_template: str,
        text_for_monitoring_template: str,
        system_prompt: str,
        progress_title: Optional[str] = None,
    ) -> None:
        self.agent = Agent(model, output_type=bool, system_prompt=system_prompt)
        self.topic_name = topic_name
        self.user_prompt_template = user_prompt_template
        self.text_for_monitoring_template = text_for_monitoring_template
        self.progress_title = progress_title
        super().__init__(monitoring_handler)

    def _run(self, data: PaperInfoCollection) -> tuple[PaperInfoCollection, str]:
        text_for_monitoring = ""
        output = data

        n_papers = len(output.papers)
        for i, p in enumerate(output.papers[:], 1):
            t_item = time.monotonic()
            try:
                result = self.agent.run_sync(
                    self.user_prompt_template.format(
                        title=p.title,
                        abstract=p.abstract,
                        topic_name=self.topic_name,
                    )
                )
            except UnexpectedModelBehavior:
                logger.exception(
                    "Failed to validate model answer. Remove paper '%s'", p.title
                )
                output.papers.remove(p)
                continue

            logger.debug(
                "classify %d/%d | '%s' | %.2fs",
                i,
                n_papers,
                p.title,
                time.monotonic() - t_item,
            )

            relevant = result.output

            if relevant:
                text_for_monitoring += to_text_description(
                    template=self.text_for_monitoring_template,
                    paper_info=p,
                )
            else:
                output.papers.remove(p)

        return output, text_for_monitoring


class PostSlopClassifier(
    Function[RedditPostCollection, ClassifiedRedditPostCollection]
):
    """Classifies Reddit posts as slop / not slop in batches via an LLM.

    Marks every post with a binary verdict; never drops posts. Fails open: a post
    with no verdict (missing id, malformed batch, failed request) is marked not slop.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: pydantic_ai.models.Model,
        system_prompt: str = SLOP_SYSTEM_PROMPT,
        batch_size: int = 20,
        max_text_chars: int = 800,
        model_settings: dict | None = None,
        retries: int | None = None,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=SlopClassificationResult,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
        )
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        super().__init__(monitoring_handler)

    def _build_batch_prompt(self, posts: list[ClassifiedRedditPost]) -> str:
        entries = [
            {
                "submission_id": cp.post.submission_id,
                "subreddit": cp.post.subreddit,
                "title": cp.post.title,
                "text": (cp.post.text or "")[: self.max_text_chars],
            }
            for cp in posts
        ]
        return json.dumps(entries, ensure_ascii=False)

    def _run(
        self, data: RedditPostCollection
    ) -> tuple[ClassifiedRedditPostCollection, str]:
        marked = [ClassifiedRedditPost(post=p) for p in data.posts]
        if not marked:
            return ClassifiedRedditPostCollection(posts=[]), (
                "Posts in: 0, kept: 0, dropped: 0 (0.0% dropped)"
            )

        batches = [
            marked[i : i + self.batch_size]
            for i in range(0, len(marked), self.batch_size)
        ]

        slop_posts: list[ClassifiedRedditPost] = []
        n_batches = len(batches)
        for b_i, batch in enumerate(batches, 1):
            t_batch = time.monotonic()
            try:
                run_result: AgentRunResult = self.agent.run_sync(
                    self._build_batch_prompt(batch)
                )
                result: SlopClassificationResult = run_result.output
            except UnexpectedModelBehavior:
                # Fail open: keep every post of the failed batch.
                logger.warning(
                    "Slop classification failed for batch %d/%d (%d posts); "
                    "keeping all posts of the batch (fail-open)",
                    b_i,
                    n_batches,
                    len(batch),
                )
                continue

            logger.debug(
                "classify slop batch %d/%d | %d posts | %.2fs",
                b_i,
                n_batches,
                len(batch),
                time.monotonic() - t_batch,
            )

            verdict_by_id = {v.submission_id: v for v in result.verdicts}
            batch_ids = {cp.post.submission_id for cp in batch}
            unknown_ids = set(verdict_by_id) - batch_ids
            if unknown_ids:
                logger.warning(
                    "Slop classification returned verdicts for %d unknown "
                    "submission_id(s); ignoring them: %s",
                    len(unknown_ids),
                    sorted(unknown_ids),
                )
            for cp in batch:
                verdict = verdict_by_id.get(cp.post.submission_id)
                if verdict is None:
                    logger.warning(
                        "No slop verdict returned for post '%s' (%s); "
                        "keeping it (fail-open)",
                        cp.post.title,
                        cp.post.submission_id,
                    )
                    continue  # fail open
                cp.is_slop = verdict.is_slop
                cp.justification = verdict.justification
                if verdict.is_slop:
                    slop_posts.append(cp)

        total = len(marked)
        dropped = sum(1 for cp in marked if cp.is_slop)
        pct = (100.0 * dropped / total) if total else 0.0
        lines = [
            f"Posts in: {total}, kept: {total - dropped}, dropped: {dropped} "
            f"({pct:.1f}% dropped)"
        ]
        for cp in slop_posts:
            lines.append(
                f"### {cp.post.title}\nURL: {cp.post.url}\n"
                f"Marked as slop: {cp.justification}\n"
            )

        return ClassifiedRedditPostCollection(posts=marked), "\n".join(lines)
