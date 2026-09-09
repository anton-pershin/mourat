"""Database writer pipeline stage for scored papers."""

from __future__ import annotations

import logging
import re

from mourat.base import Function
from mourat.data_models import ScoredPaperCollection
from mourat.database import content_item as ci
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

# Valid id prefixes for content_items.id, derived from the normalised title.
_ID_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _content_item_id(title: str) -> str:
    """Derive a stable content item id from the paper title.

    Normalising the title (lowercase, alphanumeric runs joined by single
    underscores) makes a re-run over the same paper produce the same id, which
    is what the upsert path keys on. Two different spellings of one paper
    produce different ids — an accepted, stated limitation (4.3).
    """
    slug = _ID_NON_ALNUM.sub("_", title.strip().lower()).strip("_")
    return f"paper_{slug}"


class ContentItemDbWriter(Function[ScoredPaperCollection, ScoredPaperCollection]):
    """Writes scored papers to the content database.

    A pass-through stage: returns its input unchanged so it composes after
    the filter like any other stage and is independently enable-able.

    Upserts on the derived content item id: the first run creates the
    content item and its relevance links; a re-run refreshes the relevance
    links of the already-stored paper instead of skipping it (FR4). The
    influence score is left unset — no influence value is computed on this
    path, and none is used to filter papers here.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        conn,
        source_type_id: str = "paper",
        platform_id: str = "web",
        influence_metric_id: str = "citations",
    ) -> None:
        self.conn = conn
        self.source_type_id = source_type_id
        self.platform_id = platform_id
        self.influence_metric_id = influence_metric_id
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
            except Exception:
                logger.debug("reference row already exists: %s", args[1])

    def _write_item(self, sp) -> str:
        """Write one scored paper; returns 'created', 'updated' or 'failed'."""
        item_id = _content_item_id(sp.paper.title)
        authors = "; ".join(sp.paper.authors)
        url = sp.paper.url or None

        existing = ci.get_content_item(self.conn, item_id)
        if existing is None:
            try:
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
                )
            except Exception:
                logger.exception(
                    "failed to create content item for paper '%s'", sp.paper.title
                )
                return "failed"
        else:
            try:
                ci.update_content_item(
                    self.conn,
                    id=item_id,
                    name=sp.paper.title,
                    description=sp.paper.abstract,
                    url=url,
                    published_at=sp.paper.publication_date,
                    authors=authors or None,
                    influence_score=sp.paper.influence_score,
                )
            except Exception:
                logger.exception(
                    "failed to update content item for paper '%s'", sp.paper.title
                )
                return "failed"

        # Refresh relevance links: remove stale ones, then insert current.
        self._refresh_links(item_id, sp)
        return "updated" if existing is not None else "created"

    def _refresh_links(self, item_id: str, sp) -> None:
        """Replace the paper's relevance junction rows with the current scores.

        All existing rows of the types present in this run's scores are removed
        first, then the current rows are inserted — a re-run updates the
        paper's relevance links instead of leaving them stale (FR4).
        """
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

        # Remove all stale rows of the types this run supplies...
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
                remove_dispatch[link_type](self.conn, item_id, old_id)

        # ...then insert the current rows.
        for se in sp.relevance_scores:
            try:
                add_dispatch[se.type](
                    self.conn, item_id, se.id, se.justification, int(se.score)
                )
            except Exception:
                logger.exception(
                    "failed to link paper '%s' to %s '%s'",
                    sp.paper.title,
                    se.type,
                    se.id,
                )

    def _run(self, data: ScoredPaperCollection) -> tuple[ScoredPaperCollection, str]:
        self._ensure_reference_rows()

        counts = {"created": 0, "updated": 0, "failed": 0}
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
            f"updated: {counts['updated']}, failed: {counts['failed']}"
        ]
        lines.extend(monitoring_lines)
        return data, "\n".join(lines)
