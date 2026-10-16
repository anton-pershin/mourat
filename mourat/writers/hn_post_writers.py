"""Configurable writers for scored HN posts."""

from __future__ import annotations

import json
import logging

from mourat.base import Function
from mourat.data_models import ScoredHNPostCollection
from mourat.database import content_item as ci
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)


class HNContentItemDbWriter(Function[ScoredHNPostCollection, ScoredHNPostCollection]):
    """Write HN posts and relevance links to the content-item database."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        conn,
        source_type_id: str = "post",
        platform_id: str = "hackernews",
        influence_metric_id: str = "hn_points",
    ) -> None:
        self.conn = conn
        self.source_type_id = source_type_id
        self.platform_id = platform_id
        self.influence_metric_id = influence_metric_id
        super().__init__(monitoring_handler)

    def _ensure_reference_rows(self) -> None:
        for create, args in (
            (
                ci.create_source_type,
                (self.conn, self.source_type_id, "Post", "Social media or blog post"),
            ),
            (
                ci.create_platform,
                (
                    self.conn,
                    self.platform_id,
                    "HackerNews",
                    "HackerNews platform",
                ),
            ),
            (
                ci.create_influence_metric,
                (
                    self.conn,
                    self.influence_metric_id,
                    "HN points",
                    "HackerNews story points",
                ),
            ),
        ):
            try:
                create(*args)
            except Exception:
                logger.debug("reference row already exists: %s", args[1])

    def _run(self, data: ScoredHNPostCollection) -> tuple[ScoredHNPostCollection, str]:
        self._ensure_reference_rows()
        saved = 0
        skipped_existing = 0
        failed_inserts = 0
        failed_links = 0
        link_writers = {
            "rq": ci.add_item_research_question,
            "tc": ci.add_item_technical_challenge,
            "topic": ci.add_item_research_topic,
            "constraint": ci.add_item_constraint,
        }

        for scored in data.posts:
            post = scored.post
            item_id = f"hn_{post.hn_id}"
            try:
                if ci.get_content_item(self.conn, item_id) is not None:
                    skipped_existing += 1
                    continue
                ci.create_content_item(
                    self.conn,
                    id=item_id,
                    name=post.title,
                    source_type_id=self.source_type_id,
                    platform_id=self.platform_id,
                    influence_metric_id=self.influence_metric_id,
                    description=post.description,
                    url=post.url,
                    published_at=post.date,
                    authors=post.author,
                    influence_score=(
                        post.influence_score
                        if post.influence_score is not None
                        else min(100, post.points)
                    ),
                )
            except Exception:
                failed_inserts += 1
                logger.exception("failed to insert HN post %s", item_id)
                continue

            saved += 1
            for entry in scored.relevance_scores:
                add_link = link_writers.get(entry.type)
                if add_link is None:
                    logger.warning(
                        "unknown score entry type %s for post %s", entry.type, item_id
                    )
                    continue
                try:
                    add_link(
                        self.conn,
                        item_id,
                        entry.id,
                        entry.justification,
                        int(entry.score),
                    )
                except Exception:
                    failed_links += 1
                    logger.exception(
                        "failed to add %s link for HN post %s", entry.type, item_id
                    )

        message = (
            f"HN DB writer: saved {saved}, skipped {skipped_existing}, "
            f"failed inserts {failed_inserts}, failed links {failed_links}"
        )
        logger.info(message)
        return data, message


class HNPostJsonlWriter(Function[ScoredHNPostCollection, ScoredHNPostCollection]):
    """Write one JSONL record for each scored HN post."""

    def __init__(self, monitoring_handler: MonitoringHandler, output_path: str) -> None:
        self.output_path = output_path
        super().__init__(monitoring_handler)

    def _run(self, data: ScoredHNPostCollection) -> tuple[ScoredHNPostCollection, str]:
        with open(self.output_path, "w", encoding="utf-8") as output:
            for scored in data.posts:
                post = scored.post
                record = {
                    "hn_id": post.hn_id,
                    "title": post.title,
                    "author": post.author,
                    "date": post.date,
                    "url": post.url,
                    "points": post.points,
                    "num_comments": post.num_comments,
                    "influence_score": (
                        post.influence_score
                        if post.influence_score is not None
                        else min(100, post.points)
                    ),
                    "description": post.description,
                    "relevance_scores": [
                        entry.model_dump() for entry in scored.relevance_scores
                    ],
                    "filtering_score": scored.filtering_score,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        message = f"Wrote {len(data.posts)} HN posts to {self.output_path}"
        logger.info(message)
        return data, message
