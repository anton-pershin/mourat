"""Collect newest papers from arXiv, Semantic Scholar, or both.

research attributes (loaded from the DB by configured id)
  -> enabled arXiv and Semantic Scholar producers
  -> PaperCandidateDeduplicator (canonical arXiv and DOI identities)
  -> RelevanceTriageClassifier (cheap batched boolean filter)
  -> ArxivHtmlAffiliationFetcher (only records with an arXiv ID)
  -> AuthorityInfluenceAssessor (batched 0-100 authority score)
  -> CandidateToResolvedConverter (candidate -> ResolvedPaper)
  -> PaperContentItemScorer (0-100 vs every supplied attribute)
  -> PaperScoreFilter (filtering_score threshold)
  -> ContentItemDbWriter and/or JsonlWriter (independently enabled)
  -> Semantic Scholar ledger and cursor commit
"""

import logging
from pathlib import Path

import httpx
import hydra
from omegaconf import DictConfig
from pydantic_ai.models import Model

from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.arxiv_html_affiliations import ArxivHtmlAffiliationFetcher
from mourat.collectors.semantic_scholar_newest import (
    SemanticScholarNewestPaperCollector,
)
from mourat.data_models import PaperCandidateCollection, ScoredPaperCollection
from mourat.database import apply_schema
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.authority_influence import AuthorityInfluenceAssessor
from mourat.processors.candidate_converter import CandidateToResolvedConverter
from mourat.processors.content_item_scorer import PaperContentItemScorer
from mourat.processors.paper_deduplicator import PaperCandidateDeduplicator
from mourat.processors.relevance_triage import RelevanceTriageClassifier
from mourat.utils.common import get_config_path
from mourat.utils.config import read_enabled

logger = logging.getLogger("mourat")

CONFIG_NAME = "config_collect_newest_papers"
_SUPPORTED_SOURCES = ("arxiv", "semantic_scholar")


def _validate_sources(cfg: DictConfig) -> list[str]:
    """Return enabled producers in their deterministic collection order."""
    sources = list(cfg.get("sources", ["arxiv"]))
    if not sources:
        raise ValueError("sources must enable arxiv, semantic_scholar, or both")
    unknown = [source for source in sources if source not in _SUPPORTED_SOURCES]
    if unknown:
        raise ValueError(f"Unknown newest-paper sources: {unknown}")
    if len(set(sources)) != len(sources):
        raise ValueError("sources must not contain duplicates")
    # arXiv is always collected first, but deduplication also enforces its
    # precedence independently of producer order.
    return [source for source in _SUPPORTED_SOURCES if source in sources]


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
            attribute_blocks.append(
                f"{label.capitalize()} '{record['name']}': {record['description'] or '(no description)'}"
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


def collect_newest_papers_main(cfg: DictConfig) -> None:
    """Collect, deduplicate, assess, score, filter, write, then commit state."""
    sources = _validate_sources(cfg)
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
        _, scoring_lists, constraint_list = _load_attributes(conn, cfg)
    finally:
        conn.close()

    if not any(scoring_lists[k] for k in ("rq", "tc", "topic")):
        raise ValueError(
            "No research attributes resolved from the database; nothing to collect for. "
            "Check the configured ids and the log for the ones that were not found."
        )

    http_client = httpx.Client(
        verify=True,
        timeout=httpx.Timeout(
            timeout=600,
            connect=5,
        ),
    )
    semantic_scholar_conn = None
    semantic_scholar_collector: SemanticScholarNewestPaperCollector | None = None
    writer_conn = None
    try:
        # Step 1: collect every enabled producer into one candidate collection.
        step_id = "1"
        candidates = PaperCandidateCollection(papers=[])
        if "arxiv" in sources:
            for name, feed_cfg in cfg.feeds.items():
                collector: ArxivPaperCollector = hydra.utils.instantiate(feed_cfg)(
                    monitoring_handler, http_client
                )
                batch: PaperCandidateCollection = collector({}, step_id=step_id)
                candidates.papers.extend(batch.papers)
                logger.info("Feed '%s' contributed %d papers", name, len(batch.papers))

        if "semantic_scholar" in sources:
            semantic_scholar_conn = create_connection(db_path)
            apply_schema(semantic_scholar_conn)
            semantic_scholar_collector = hydra.utils.instantiate(
                cfg.semantic_scholar_newest
            )(
                monitoring_handler,
                http_client,
                conn=semantic_scholar_conn,
            )
            batch = semantic_scholar_collector({}, step_id=step_id)
            candidates.papers.extend(batch.papers)
            logger.info("Semantic Scholar contributed %d papers", len(batch.papers))

        # Step 2: apply source invariants before any LLM or per-paper requests.
        step_id = "2"
        deduplicator: PaperCandidateDeduplicator = hydra.utils.instantiate(
            cfg.paper_deduplicator
        )(monitoring_handler)
        candidates = deduplicator(candidates, step_id=step_id)

        # Step 3: cheap boolean triage before any expensive per-item work.
        step_id = "3"
        triage_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
        triage = hydra.utils.instantiate(cfg.relevance_triage)(
            monitoring_handler,
            model=triage_llm,
            rq_list=scoring_lists["rq"],
            tc_list=scoring_lists["tc"],
            topic_list=scoring_lists["topic"],
        )
        candidates = triage(candidates, step_id=step_id)

        # Step 4: DOI-only candidates keep affiliations unknown; arXiv IDs are fetched.
        step_id = "4"
        fetcher = hydra.utils.instantiate(cfg.arxiv_html_affiliations)(
            monitoring_handler, http_client
        )
        candidates = fetcher(candidates, step_id=step_id)

        # Step 5: authority influence is stored, never used as a filter.
        step_id = "5"
        authority_llm: Model = hydra.utils.instantiate(cfg.authority_llm)
        assessor = hydra.utils.instantiate(cfg.authority_influence)(
            monitoring_handler,
            model=authority_llm,
        )
        candidates = assessor(candidates, step_id=step_id)

        # Step 6: candidate -> resolved lineage for the shared tail.
        step_id = "6"
        converter = hydra.utils.instantiate(cfg.candidate_converter)(monitoring_handler)
        resolved = converter(candidates, step_id=step_id)

        # Step 7: relevance scoring against every supplied attribute.
        step_id = "7"
        scoring_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
        scorer: PaperContentItemScorer = hydra.utils.instantiate(
            cfg.content_item_scorer
        )(
            monitoring_handler,
            model=scoring_llm,
            rq_list=scoring_lists["rq"],
            tc_list=scoring_lists["tc"],
            topic_list=scoring_lists["topic"],
            constraint_list=constraint_list,
            constraints_contribute_to_filtering_score=True,
        )
        scored: ScoredPaperCollection = scorer(resolved, step_id=step_id)

        # Step 8: threshold filter.
        step_id = "8"
        score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
        filtered: ScoredPaperCollection = score_filter(scored, step_id=step_id)

        # Step 9: writers are independently enabled.
        step_id = "9"
        db_writer_cfg = cfg.db_writer.copy()
        jsonl_writer_cfg = cfg.jsonl_writer.copy()
        db_enabled = read_enabled(db_writer_cfg)
        jsonl_enabled = read_enabled(jsonl_writer_cfg)
        needs_conn = db_enabled or (
            jsonl_enabled and bool(jsonl_writer_cfg.get("skip_existing", False))
        )
        writer_conn = create_connection(db_path) if needs_conn else None
        if jsonl_enabled:
            jsonl_writer = hydra.utils.instantiate(jsonl_writer_cfg)(
                monitoring_handler, conn=writer_conn
            )
            jsonl_writer(filtered, step_id=step_id)

        if db_enabled:
            db_writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=writer_conn
            )
            db_writer(filtered, step_id=step_id)

        # This is the only Semantic Scholar state commit point. Any exception
        # above leaves both the seen-paper ledger and query cursors unchanged.
        if semantic_scholar_collector is not None:
            semantic_scholar_collector.commit()
            logger.info("Semantic Scholar collection state committed")

        logger.info(
            "Pipeline complete: %d scored, %d after filter",
            len(scored.papers),
            len(filtered.papers),
        )
    finally:
        if writer_conn is not None:
            writer_conn.close()
        if semantic_scholar_conn is not None:
            semantic_scholar_conn.close()
        http_client.close()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_newest_papers_main)()
