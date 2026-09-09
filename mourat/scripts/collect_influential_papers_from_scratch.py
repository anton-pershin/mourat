"""Collect influential papers from scratch: discover, resolve, score, filter, write.

Entry point for the from-scratch paper collection pipeline:

    research attributes (loaded from the DB by configured id)
      -> PaperDiscoverer (LLM agent with web search)
      -> PaperResolver (OpenAlex/arXiv metadata resolution, drop unresolved)
      -> InfluenceAssessor (fwci / citations-per-year -> 0-100)
      -> ArxivPdfVerifier (verified arXiv PDF url or empty with reason)
      -> PaperContentItemScorer (0-100 vs every supplied attribute)
      -> PaperScoreFilter (filtering_score threshold)
      -> ContentItemDbWriter and/or JsonlWriter (independently enabled)
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf


@contextmanager
def omegaconf_open_dict(cfg: DictConfig):
    """Temporarily unseal a struct-mode DictConfig so keys can be deleted."""
    old = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        yield cfg
    finally:
        OmegaConf.set_struct(cfg, old)


from pydantic_ai.models import Model

from mourat.collectors.paper_discoverer import PaperDiscoverer
from mourat.data_models import (
    PaperCandidateCollection,
    ResolvedPaperCollection,
    ScoredPaperCollection,
)
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.arxiv_pdf_verifier import ArxivPdfVerifier
from mourat.processors.content_item_scorer import PaperContentItemScorer
from mourat.processors.influence_assessor import InfluenceAssessor
from mourat.resolvers.paper_resolver import PaperResolver
from mourat.utils.common import get_config_path

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_collect_influential_papers_from_scratch"


def _load_attributes(conn, cfg: DictConfig) -> tuple[list[str], list[dict], list[dict]]:
    """Load research attributes and constraints from the database by id.

    Returns (attribute blocks for the discoverer prompt, scoring rq/tc/topic
    list, scoring constraint list). An id absent from the database is reported
    (logger.error) and skipped, not silently ignored (FR1).
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
    constraint_blocks: list[str] = []
    for constraint_id in cfg.get("constraint_ids", []):
        record = bd.get_constraint(conn, constraint_id)
        if record is None:
            logger.error(
                "constraint id '%s' not found in the database; skipped", constraint_id
            )
            continue
        constraint_blocks.append(
            f"Constraint '{record['name']}': {record['description'] or '(no description)'}"
        )
        constraint_list.append(
            {
                "id": record["id"],
                "name": record["name"],
                "type": "constraint",
                "description": record["description"] or "",
            }
        )

    return attribute_blocks + constraint_blocks, scoring_lists, constraint_list


def _read_enabled(writer_cfg: DictConfig) -> bool:
    """Read the writer's enabled flag without mutating the struct-mode config."""
    enabled = writer_cfg.get("enabled", False)
    if enabled and "enabled" in writer_cfg:
        with omegaconf_open_dict(writer_cfg):
            del writer_cfg["enabled"]
    return bool(enabled)


def collect_influential_papers_from_scratch_main(cfg: DictConfig) -> None:
    """Main pipeline: discover -> score -> filter -> write."""
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
        attribute_descriptions, scoring_lists, constraint_list = _load_attributes(
            conn, cfg
        )
    finally:
        conn.close()

    if not attribute_descriptions:
        raise ValueError(
            "No research attributes resolved from the database; nothing to collect for. "
            "Check the configured ids and the log for the ones that were not found."
        )

    discovery_llm: Model = hydra.utils.instantiate(cfg.discovery_llm)
    discoverer: PaperDiscoverer = hydra.utils.instantiate(cfg.paper_discoverer)(
        monitoring_handler,
        model=discovery_llm,
        attribute_description="\n\n".join(attribute_descriptions),
    )
    step_id = "1"
    candidates: PaperCandidateCollection = discoverer({}, step_id=step_id)

    resolver: PaperResolver = hydra.utils.instantiate(cfg.paper_resolver)(
        monitoring_handler
    )
    step_id = "2"
    resolved: ResolvedPaperCollection = resolver(candidates, step_id=step_id)

    assessor: InfluenceAssessor = hydra.utils.instantiate(cfg.influence_assessor)(
        monitoring_handler
    )
    step_id = "3"
    assessed: ResolvedPaperCollection = assessor(resolved, step_id=step_id)

    verifier: ArxivPdfVerifier = hydra.utils.instantiate(cfg.arxiv_pdf_verifier)(
        monitoring_handler
    )
    step_id = "4"
    verified: ResolvedPaperCollection = verifier(assessed, step_id=step_id)

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
    step_id = "5"
    scored: ScoredPaperCollection = scorer(verified, step_id=step_id)

    score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
    step_id = "6"
    filtered: ScoredPaperCollection = score_filter(scored, step_id=step_id)

    step_id = "7"
    db_writer_cfg = cfg.db_writer.copy()
    if _read_enabled(db_writer_cfg):
        conn = create_connection(db_path)
        try:
            db_writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=conn
            )
            db_writer(filtered, step_id=step_id)
        finally:
            conn.close()

    jsonl_writer_cfg = cfg.jsonl_writer.copy()
    if _read_enabled(jsonl_writer_cfg):
        jsonl_writer = hydra.utils.instantiate(jsonl_writer_cfg)(monitoring_handler)
        jsonl_writer(filtered, step_id=step_id)

    logger.info(
        "Pipeline complete: %d discovered, %d resolved, %d scored, %d after filter",
        len(candidates.papers),
        len(resolved.papers),
        len(scored.papers),
        len(filtered.papers),
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_influential_papers_from_scratch_main)()
