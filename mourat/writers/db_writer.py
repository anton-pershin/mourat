"""Database writer pipeline stage for scored papers."""

from __future__ import annotations

import logging
import sqlite3

from mourat.base import Function
from mourat.data_models import ScoredPaperCollection
from mourat.database import apply_schema
from mourat.database import content_item as ci
from mourat.database.paper_identity import (
    paper_content_item_id,
    register_paper_aliases,
    resolve_paper_content_item_id,
)
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)


def _content_item_id(
    title: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> str:
    """Derive a stable paper ID with arXiv, DOI, then title precedence."""
    return paper_content_item_id(title=title, arxiv_id=arxiv_id, doi=doi)


class ContentItemDbWriter(Function[ScoredPaperCollection, ScoredPaperCollection]):
    """Write scored papers with atomic metadata, link, and identity updates.

    A pass-through stage returns its input unchanged. A re-run refreshes the
    stored paper and its relevance links. Canonical arXiv and DOI aliases join
    runs, and arXiv metadata remains primary. If ``fail_on_error`` is enabled,
    the first failed paper raises after its transaction rolls back. Otherwise,
    the writer reports the failed item and continues with later papers.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        conn,
        source_type_id: str = "paper",
        platform_id: str = "web",
        influence_metric_id: str = "citations",
        fail_on_error: bool = False,
    ) -> None:
        self.conn = conn
        self.source_type_id = source_type_id
        self.platform_id = platform_id
        self.influence_metric_id = influence_metric_id
        self.fail_on_error = fail_on_error
        super().__init__(monitoring_handler)

    def _ensure_reference_rows(self) -> None:
        """Create the reference rows the content item FKs point at (idempotent)."""
        for create, args in [
            (
                ci.create_source_type,
                (self.conn, self.source_type_id, "Paper", "Academic paper"),
            ),
            (
                ci.create_platform,
                (self.conn, self.platform_id, "Web", "Discovered via web search"),
            ),
            (
                ci.create_influence_metric,
                (self.conn, self.influence_metric_id, "Citations", "Citation count"),
            ),
        ]:
            try:
                create(*args)
            except sqlite3.Error:
                logger.debug("reference row already exists: %s", args[1])

    def _write_item(self, sp) -> str:
        """Write one scored paper in one transaction and return its outcome."""
        arxiv_id = getattr(sp.paper, "arxiv_id", None)
        doi = getattr(sp.paper, "doi", None)
        try:
            with self.conn:
                resolution = resolve_paper_content_item_id(
                    self.conn,
                    title=sp.paper.title,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    manage_transaction=False,
                )
                item_id = resolution.content_item_id
                if resolution.preserve_existing:
                    self._refresh_links(item_id, sp)
                    register_paper_aliases(
                        self.conn,
                        content_item_id=item_id,
                        arxiv_id=arxiv_id,
                        doi=doi,
                        manage_transaction=False,
                    )
                    logger.info(
                        "kept arXiv-primary metadata for paper '%s'; "
                        "refreshed links from DOI-only update",
                        sp.paper.title,
                    )
                    return "skipped"

                authors = "; ".join(sp.paper.authors)
                url = sp.paper.url or None
                existing = ci.get_content_item(self.conn, item_id)
                if existing is None:
                    ci.create_content_item(
                        self.conn,
                        id=item_id,
                        name=sp.paper.title,
                        source_type_id=self.source_type_id,
                        platform_id=self.platform_id,
                        influence_metric_id=self.influence_metric_id,
                        description=sp.paper.abstract,
                        url=url,
                        published_at=sp.paper.publication_date,
                        authors=authors or None,
                        influence_score=sp.paper.influence_score,
                        commit=False,
                    )
                else:
                    ci.update_content_item(
                        self.conn,
                        id=item_id,
                        name=sp.paper.title,
                        description=sp.paper.abstract,
                        url=url,
                        published_at=sp.paper.publication_date,
                        authors=authors or None,
                        influence_score=sp.paper.influence_score,
                        commit=False,
                    )

                self._refresh_links(item_id, sp)
                register_paper_aliases(
                    self.conn,
                    content_item_id=item_id,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    manage_transaction=False,
                )
                return "updated" if existing is not None else "created"
        except sqlite3.Error:
            logger.exception("failed to write paper '%s'", sp.paper.title)
            if self.fail_on_error:
                raise
            return "failed"

    def _refresh_links(self, item_id: str, sp) -> None:
        """Replace relevance junction rows with the current scores."""
        remove_dispatch = {
            "rq": ci.remove_item_research_question,
            "tc": ci.remove_item_technical_challenge,
            "topic": ci.remove_item_research_topic,
            "constraint": ci.remove_item_constraint,
        }
        add_dispatch = {
            "rq": ci.add_item_research_question,
            "tc": ci.add_item_technical_challenge,
            "topic": ci.add_item_research_topic,
            "constraint": ci.add_item_constraint,
        }

        present_types = {se.type for se in sp.relevance_scores}
        for link_type in present_types:
            remove = remove_dispatch.get(link_type)
            if remove is None:
                logger.warning(
                    "unknown score entry type '%s' for paper '%s'; skipped",
                    link_type,
                    sp.paper.title,
                )
                present_types.discard(link_type)

        list_dispatch = {
            "rq": ci.list_item_research_questions,
            "tc": ci.list_item_technical_challenges,
            "topic": ci.list_item_research_topics,
            "constraint": ci.list_item_constraints,
        }
        for link_type in present_types:
            existing_ids = [
                row["id"] for row in list_dispatch[link_type](self.conn, item_id)
            ]
            for old_id in existing_ids:
                remove_dispatch[link_type](self.conn, item_id, old_id, commit=False)

        for se in sp.relevance_scores:
            try:
                add_dispatch[se.type](
                    self.conn,
                    item_id,
                    se.id,
                    se.justification,
                    int(se.score),
                    commit=False,
                )
            except sqlite3.Error:
                if self.fail_on_error:
                    raise
                # Legacy pipelines retain the paper when only a link cannot be
                # written. The newest-paper pipeline uses strict mode instead.
                logger.exception(
                    "failed to link paper '%s' to %s '%s'",
                    sp.paper.title,
                    se.type,
                    se.id,
                )

    def _run(self, data: ScoredPaperCollection) -> tuple[ScoredPaperCollection, str]:
        apply_schema(self.conn)
        self._ensure_reference_rows()

        counts = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
        monitoring_lines = []

        for sp in data.papers:
            outcome = self._write_item(sp)
            counts[outcome] += 1
            if outcome == "failed":
                monitoring_lines.append(
                    f"### FAILED: {sp.paper.title}\n"
                    f"(write to the database failed; see the log for the reason)\n"
                )

        lines = [
            f"Papers in: {len(data.papers)}, created: {counts['created']}, "
            f"updated: {counts['updated']}, skipped: {counts['skipped']}, "
            f"failed: {counts['failed']}"
        ]
        lines.extend(monitoring_lines)
        return data, "\n".join(lines)
