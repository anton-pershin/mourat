"""Collect newest papers: RSS collect, triage, authority, score, filter, write.

Entry point for the newest-papers paper collection pipeline (spec 14):

    research attributes (loaded from the DB by configured id)
      -> ArxivPaperCollector (RSS feeds, newest mode, N feeds combined)
      -> RelevanceTriageClassifier (cheap batched boolean filter)
      -> ArxivHtmlAffiliationFetcher (per-paper HTML render)
      -> AuthorityInfluenceAssessor (batched 0-100 authority score)
      -> CandidateToResolvedConverter (candidate -> ResolvedPaper)
      -> PaperContentItemScorer (0-100 vs every supplied attribute)
      -> PaperScoreFilter (filtering_score threshold)
      -> ContentItemDbWriter and/or JsonlWriter (independently enabled)
"""

import logging
from pathlib import Path

import httpx
import hydra
from omegaconf import DictConfig
from pydantic_ai.models import Model


from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.arxiv_html_affiliations import ArxivHtmlAffiliationFetcher
from mourat.data_models import PaperCandidateCollection, ScoredPaperCollection
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.authority_influence import AuthorityInfluenceAssessor
from mourat.processors.candidate_converter import CandidateToResolvedConverter
from mourat.processors.content_item_scorer import PaperContentItemScorer
from mourat.processors.relevance_triage import RelevanceTriageClassifier
from mourat.utils.common import get_config_path
from mourat.utils.config import read_enabled

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_collect_newest_papers"


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
    """Main pipeline: collect -> triage -> authority -> score -> filter -> write."""
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
        verify=False,
        timeout=httpx.Timeout(
            timeout=600,
            connect=5,
        ),
    )

    # Step 1: collect the recent feeds from arxiv (R8: combined across feeds)
    step_id = "1"
    candidates = PaperCandidateCollection(papers=[])
    for name, feed_cfg in cfg.feeds.items():
        collector: ArxivPaperCollector = hydra.utils.instantiate(feed_cfg)(
            monitoring_handler, http_client
        )
        batch: PaperCandidateCollection = collector({}, step_id=step_id)
        candidates.papers.extend(batch.papers)
        logger.info("Feed '%s' contributed %d papers", name, len(batch.papers))

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

    # Step 3: affiliations from the arXiv HTML render (deterministic)
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
    step_id = "8"
    db_writer_cfg = cfg.db_writer.copy()
    if read_enabled(db_writer_cfg):
        conn = create_connection(db_path)
        try:
            db_writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=conn
            )
            db_writer(filtered, step_id=step_id)
        finally:
            conn.close()

    jsonl_writer_cfg = cfg.jsonl_writer.copy()
    if read_enabled(jsonl_writer_cfg):
        jsonl_writer = hydra.utils.instantiate(jsonl_writer_cfg)(monitoring_handler)
        jsonl_writer(filtered, step_id=step_id)

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
