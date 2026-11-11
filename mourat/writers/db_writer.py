"""Database writer pipeline stage for scored papers."""

from __future__ import annotations

import logging
import re

from comourat.models import normalize_arxiv_id, normalize_doi

from mourat.base import Function
from mourat.data_models import ResolvedPaper, ScoredPaperCollection
from mourat.database import apply_schema
from mourat.database import content_item as ci
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

# Valid id prefixes for content_items.id, derived from the normalised title.
_ID_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalized_doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    return normalized.lower() if normalized else None


def _paper_identities(paper: ResolvedPaper) -> list[tuple[str, str]]:
    """Return normalized provider aliases in precedence order."""
    identities = []
    arxiv_id = normalize_arxiv_id(paper.arxiv_id)
    doi = _normalized_doi(paper.doi)
    if arxiv_id:
        identities.append(("arxiv", arxiv_id))
    if doi:
        identities.append(("doi", doi))
    return identities


def _content_item_id(
    title: str, arxiv_id: str | None = None, doi: str | None = None
) -> str:
    """Derive an ID by arXiv, DOI, then the legacy title fallback."""
    arxiv_id = normalize_arxiv_id(arxiv_id)
    doi = _normalized_doi(doi)
    if arxiv_id:
        return f"paper_arxiv-{arxiv_id}"
    if doi:
        return f"paper_doi-{doi}"
    slug = _ID_NON_ALNUM.sub("_", title.strip().lower()).strip("_")
    return f"paper_{slug}"


def _derived_content_item_ids(paper: ResolvedPaper) -> list[str]:
    """Return current and legacy IDs that can identify an existing row."""
    ids = []
    for arxiv_id, doi in (
        (paper.arxiv_id, paper.doi),
        (paper.arxiv_id, None),
        (None, paper.doi),
        (None, None),
    ):
        item_id = _content_item_id(paper.title, arxiv_id, doi)
        if item_id not in ids:
            ids.append(item_id)
    return ids


def _has_identity_table(conn) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'paper_identities'"
        ).fetchone()
        is not None
    )


def _existing_content_item_ids(conn, paper: ResolvedPaper) -> list[str]:
    """Find existing rows through current IDs, legacy IDs, and aliases."""
    existing = []

    def add(item_id: str | None) -> None:
        if item_id and item_id not in existing:
            if ci.get_content_item(conn, item_id) is not None:
                existing.append(item_id)

    for item_id in _derived_content_item_ids(paper):
        add(item_id)
    if _has_identity_table(conn):
        for identity_type, identity_value in _paper_identities(paper):
            rows = conn.execute(
                "SELECT content_item_id FROM paper_identities "
                "WHERE identity_type = ? AND identity_value = ?",
                (identity_type, identity_value),
            ).fetchall()
            for row in rows:
                add(row["content_item_id"])
    return existing


def _existing_content_item_id(conn, paper: ResolvedPaper) -> str | None:
    """Read-only identity lookup shared with the JSONL writer."""
    existing = _existing_content_item_ids(conn, paper)
    return existing[0] if existing else None


def _existing_arxiv_identity(conn, existing_ids: list[str]) -> str | None:
    """Retain arXiv precedence when an incoming record has only a DOI."""
    if not _has_identity_table(conn):
        return None
    for item_id in existing_ids:
        row = conn.execute(
            "SELECT identity_value FROM paper_identities "
            "WHERE content_item_id = ? AND identity_type = 'arxiv' "
            "ORDER BY identity_value LIMIT 1",
            (item_id,),
        ).fetchone()
        if row is not None:
            return row["identity_value"]
    return None


def _reconcile_content_item_id(conn, paper: ResolvedPaper) -> str:
    """Make the preferred ID canonical and merge any legacy duplicates."""
    existing = _existing_content_item_ids(conn, paper)
    arxiv_id = normalize_arxiv_id(paper.arxiv_id)
    if not arxiv_id:
        arxiv_id = _existing_arxiv_identity(conn, existing)
    canonical = _content_item_id(paper.title, arxiv_id, paper.doi)
    if not existing:
        return canonical
    if canonical not in existing:
        ci.rekey_content_item(conn, existing[0], canonical)
    for duplicate_id in existing:
        if duplicate_id != canonical and ci.get_content_item(conn, duplicate_id):
            ci.merge_content_items(conn, duplicate_id, canonical)
    return canonical


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
        apply_schema(conn)
        self.conn = conn
        self.source_type_id = source_type_id
        self.platform_id = platform_id
        self.influence_metric_id = influence_metric_id
        super().__init__(monitoring_handler)

    def _ensure_reference_rows(self) -> None:
        """Create reference rows without leaving a failed duplicate transaction."""
        reference_rows = [
            (
                "source_types",
                (self.source_type_id, "Paper", "Academic paper"),
            ),
            (
                "platforms",
                (self.platform_id, "Web", "Discovered via web search"),
            ),
            (
                "influence_metrics",
                (self.influence_metric_id, "Citations", "Citation count"),
            ),
        ]
        with self.conn:
            for table, values in reference_rows:
                self.conn.execute(
                    f"INSERT OR IGNORE INTO {table} (id, name, description) "
                    "VALUES (?, ?, ?)",
                    values,
                )

    def _write_item(self, sp) -> str:
        """Write one scored paper; returns 'created', 'updated' or 'failed'."""
        item_id = _reconcile_content_item_id(self.conn, sp.paper)
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
        for identity_type, identity_value in _paper_identities(sp.paper):
            self.conn.execute(
                "INSERT INTO paper_identities "
                "(identity_type, identity_value, content_item_id) VALUES (?, ?, ?) "
                "ON CONFLICT(identity_type, identity_value) DO UPDATE SET "
                "content_item_id = excluded.content_item_id",
                (identity_type, identity_value, item_id),
            )
        self.conn.commit()
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
