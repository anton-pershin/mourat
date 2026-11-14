"""Collect newest papers from configured sources, process them, and write output.

Entry point for the newest-papers paper collection pipeline (specs 14, 28, and 36):

    research attributes (loaded from the DB by configured id)
      -> arXiv RSS and/or Semantic Scholar publication-date collectors
      -> canonical identity merge with arXiv precedence
      -> RelevanceTriageClassifier (cheap batched boolean filter)
      -> ArxivHtmlAffiliationFetcher (arXiv HTML or Semantic Scholar authors)
      -> AuthorityInfluenceAssessor (batched 0-100 authority score)
      -> CandidateToResolvedConverter (candidate -> ResolvedPaper)
      -> PaperContentItemScorer (0-100 vs every supplied attribute)
      -> PaperScoreFilter (filtering_score threshold)
      -> ContentItemDbWriter and/or JsonlWriter (independently enabled)
"""

import logging
from pathlib import Path
from typing import Any, cast

import httpx
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic_ai.models import Model

from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.semantic_scholar_newest import (
    SemanticScholarNewestPaperCollector,
)
from mourat.data_models import PaperCandidateCollection, ScoredPaperCollection
from mourat.database import apply_schema
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.database.paper_identity import (
    list_semantic_scholar_ledger_ids,
    record_semantic_scholar_processed,
)
from mourat.monitoring import MonitoringHandler
from mourat.processors.content_item_scorer import PaperContentItemScorer
from mourat.processors.newest_paper_merger import (
    NewestPaperMergeResult,
    merge_newest_paper_candidates,
)
from mourat.utils.common import get_config_path
from mourat.utils.config import read_enabled

logger = logging.getLogger("mourat")

CONFIG_NAME = "config_collect_newest_papers"
_SOURCE_ORDER = ("arxiv", "semantic_scholar")


def _selected_sources(cfg: DictConfig) -> tuple[str, ...]:
    """Validate source selection and return it in deterministic order."""
    sources = list(cfg.get("sources", []))
    if not sources:
        raise ValueError("sources must contain at least one source")
    unknown = [source for source in sources if source not in _SOURCE_ORDER]
    if unknown:
        raise ValueError(f"Unknown newest-paper sources: {unknown}")
    if len(set(sources)) != len(sources):
        raise ValueError("sources must not contain duplicate values")
    return tuple(source for source in _SOURCE_ORDER if source in sources)


def _make_http_client(proxy: str | None = None) -> httpx.Client:
    return httpx.Client(
        verify=False,
        timeout=httpx.Timeout(
            timeout=600,
            connect=5,
        ),
        proxy=proxy,
    )


def _load_attributes(
    conn, cfg: DictConfig
) -> tuple[list[str], dict[str, list[dict]], list[dict]]:
    """Load research attributes and constraints from the database by id.

    Returns (attribute blocks for the triage/scorer prompts, scoring rq/tc/topic
    list, scoring constraint list). An id absent from the database is reported
    (logger.error) and skipped, not silently ignored.
    """
    attribute_blocks: list[str] = []
    scoring_lists: dict[str, list[dict]] = {
        "rq": [],
        "tc": [],
        "topic": [],
    }

    getters = {
        "research_question": (rd.get_research_question, "rq", "research question"),
        "technical_challenge": (
            bd.get_technical_challenge,
            "tc",
            "technical challenge",
        ),
        "research_topic": (rd.get_research_topic, "topic", "research topic"),
    }

    for kind, cfg_ids in [
        ("research_question", cfg.research_question_ids),
        ("technical_challenge", cfg.technical_challenge_ids),
        ("research_topic", cfg.research_topic_ids),
    ]:
        getter, type_key, label = getters[kind]
        for attr_id in cfg_ids:
            record = getter(conn, attr_id)
            if record is None:
                logger.error(
                    "%s id '%s' not found in the database; skipped", label, attr_id
                )
                continue
            description = record["description"] or "(no description)"
            attribute_blocks.append(
                f"{label.capitalize()} '{record['name']}': {description}"
            )
            scoring_lists[type_key].append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "type": type_key,
                    "description": record["description"] or "",
                }
            )

    constraint_list: list[dict] = []
    for constraint_id in cfg.get("constraint_ids", []):
        record = bd.get_constraint(conn, constraint_id)
        if record is None:
            logger.error(
                "constraint id '%s' not found in the database; skipped", constraint_id
            )
            continue
        constraint_list.append(
            {
                "id": record["id"],
                "name": record["name"],
                "type": "constraint",
                "description": record["description"] or "",
            }
        )

    return attribute_blocks, scoring_lists, constraint_list


def _collect_arxiv(
    cfg: DictConfig,
    monitoring_handler: MonitoringHandler,
    http_client: httpx.Client,
) -> PaperCandidateCollection:
    collection = PaperCandidateCollection(papers=[])
    for name, feed_cfg in cfg.feeds.items():
        collector: ArxivPaperCollector = hydra.utils.instantiate(feed_cfg)(
            monitoring_handler, http_client
        )
        batch: PaperCandidateCollection = collector({}, step_id=f"1-arxiv-{name}")
        for paper in batch.papers:
            paper.provenance = ["arxiv"]
        collection.papers.extend(batch.papers)
        logger.info("Feed '%s' contributed %d papers", name, len(batch.papers))
    return collection


def _collect_semantic_scholar(
    cfg: DictConfig,
    monitoring_handler: MonitoringHandler,
) -> PaperCandidateCollection:
    semantic_scholar_cfg = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg.semantic_scholar_newest, resolve=True),
    )
    proxy = semantic_scholar_cfg.pop("proxy", None)
    http_client = _make_http_client(proxy=proxy)
    collector: SemanticScholarNewestPaperCollector = hydra.utils.instantiate(
        semantic_scholar_cfg
    )(monitoring_handler, http_client)
    collection: PaperCandidateCollection = collector({}, step_id="1-semantic_scholar")
    logger.info(
        "Semantic Scholar contributed %d papers before ledger and identity filters",
        len(collection.papers),
    )
    return collection


def _collect_and_merge(
    cfg: DictConfig,
    selected_sources: tuple[str, ...],
    seen_provider_ids: set[str],
    monitoring_handler: MonitoringHandler,
    http_client: httpx.Client,
) -> NewestPaperMergeResult:
    arxiv_candidates = PaperCandidateCollection(papers=[])
    semantic_scholar_candidates = PaperCandidateCollection(papers=[])
    if "arxiv" in selected_sources:
        arxiv_candidates = _collect_arxiv(cfg, monitoring_handler, http_client)
    if "semantic_scholar" in selected_sources:
        semantic_scholar_candidates = _collect_semantic_scholar(cfg, monitoring_handler)

    merge = merge_newest_paper_candidates(
        arxiv=arxiv_candidates,
        semantic_scholar=semantic_scholar_candidates,
        seen_provider_ids=seen_provider_ids,
    )
    merge_lines = [
        "# Source merge",
        f"arXiv collected: {merge.arxiv_collected}",
        f"Semantic Scholar collected: {merge.semantic_scholar_collected}",
        f"Excluded by ledger: {merge.ledger_excluded}",
        f"Excluded without provider ID: {merge.missing_provider_id_excluded}",
        f"Semantic Scholar-only excluded without DOI: {merge.missing_doi_excluded}",
        f"Duplicate candidates excluded: {merge.duplicates_excluded}",
        f"Cross-source records merged: {merge.merged_cross_source}",
        f"Candidates after merge: {len(merge.collection.papers)}",
    ]
    monitoring_handler("1-merge", "\n".join(merge_lines))
    logger.info("Newest-paper source merge: %s", "; ".join(merge_lines[1:]))
    return merge


def _write_outputs(
    cfg: DictConfig,
    monitoring_handler: MonitoringHandler,
    db_path: Path,
    filtered: ScoredPaperCollection,
) -> None:
    db_writer_cfg = cfg.db_writer.copy()
    jsonl_writer_cfg = cfg.jsonl_writer.copy()
    db_enabled = read_enabled(db_writer_cfg)
    jsonl_enabled = read_enabled(jsonl_writer_cfg)
    needs_conn = db_enabled or (
        jsonl_enabled and bool(jsonl_writer_cfg.get("skip_existing", False))
    )
    writer_conn = create_connection(db_path) if needs_conn else None
    try:
        if jsonl_enabled:
            jsonl_writer = hydra.utils.instantiate(jsonl_writer_cfg)(
                monitoring_handler, conn=writer_conn
            )
            jsonl_writer(filtered, step_id="8")
        if db_enabled:
            db_writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=writer_conn
            )
            db_writer(filtered, step_id="8")
    finally:
        if writer_conn is not None:
            writer_conn.close()


def _record_ledger(
    db_path: Path,
    provider_ids: list[str],
    monitoring_handler: MonitoringHandler,
) -> None:
    ledger_conn = create_connection(db_path)
    try:
        record_semantic_scholar_processed(ledger_conn, provider_ids)
    finally:
        ledger_conn.close()
    ledger_text = (
        f"Recorded {len(provider_ids)} Semantic Scholar provider IDs in the "
        "collection ledger"
    )
    monitoring_handler("9", ledger_text)
    logger.info("%s", ledger_text)


def collect_newest_papers_main(cfg: DictConfig) -> None:
    """Main pipeline: collect -> triage -> authority -> score -> filter -> write."""
    selected_sources = _selected_sources(cfg)
    db_path = cfg.get("db_path")
    if db_path is None:
        raise ValueError("db_path not set in config")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )

    conn = create_connection(db_path)
    try:
        apply_schema(conn)
        _, scoring_lists, constraint_list = _load_attributes(conn, cfg)
        seen_provider_ids = (
            list_semantic_scholar_ledger_ids(conn)
            if "semantic_scholar" in selected_sources
            else set()
        )
    finally:
        conn.close()

    if not any(scoring_lists[k] for k in ("rq", "tc", "topic")):
        raise ValueError(
            "No research attributes resolved from the database; "
            "nothing to collect for. Check the configured ids and the log "
            "for the ones that were not found."
        )

    http_client = _make_http_client()

    # Step 1: collect selected sources and merge canonical paper identities.
    merge = _collect_and_merge(
        cfg,
        selected_sources,
        seen_provider_ids,
        monitoring_handler,
        http_client,
    )
    candidates = merge.collection

    # Step 2: cheap boolean triage before any expensive per-item work (R9)
    step_id = "2"
    triage_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
    triage = hydra.utils.instantiate(cfg.relevance_triage)(
        monitoring_handler,
        model=triage_llm,
        rq_list=scoring_lists["rq"],
        tc_list=scoring_lists["tc"],
        topic_list=scoring_lists["topic"],
    )
    candidates = triage(candidates, step_id=step_id)

    # Step 3: affiliations from arXiv HTML or Semantic Scholar author data
    step_id = "3"
    fetcher = hydra.utils.instantiate(cfg.arxiv_html_affiliations)(
        monitoring_handler, http_client
    )
    candidates = fetcher(candidates, step_id=step_id)

    # Step 4: authority influence (batched LLM, score 0 on failure, R4)
    step_id = "4"
    authority_llm: Model = hydra.utils.instantiate(cfg.authority_llm)
    assessor = hydra.utils.instantiate(cfg.authority_influence)(
        monitoring_handler,
        model=authority_llm,
    )
    candidates = assessor(candidates, step_id=step_id)

    # Step 5: candidate -> resolved lineage for the shared tail
    step_id = "5"
    converter = hydra.utils.instantiate(cfg.candidate_converter)(monitoring_handler)
    resolved = converter(candidates, step_id=step_id)

    # Step 6: relevance scoring vs every supplied attribute
    step_id = "6"
    scoring_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
    scorer: PaperContentItemScorer = hydra.utils.instantiate(cfg.content_item_scorer)(
        monitoring_handler,
        model=scoring_llm,
        rq_list=scoring_lists["rq"],
        tc_list=scoring_lists["tc"],
        topic_list=scoring_lists["topic"],
        constraint_list=constraint_list,
        constraints_contribute_to_filtering_score=True,
    )
    scored: ScoredPaperCollection = scorer(resolved, step_id=step_id)

    # Step 7: threshold filter
    step_id = "7"
    score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
    filtered: ScoredPaperCollection = score_filter(scored, step_id=step_id)

    # Step 8: writers, independently enabled (both off = nothing written)
    _write_outputs(cfg, monitoring_handler, db_path, filtered)

    # Step 9: mark provider IDs only after every enabled writer returns.
    if "semantic_scholar" in selected_sources:
        _record_ledger(db_path, merge.new_semantic_scholar_ids, monitoring_handler)

    logger.info(
        "Pipeline complete: %d scored, %d after filter",
        len(scored.papers),
        len(filtered.papers),
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_newest_papers_main)()
