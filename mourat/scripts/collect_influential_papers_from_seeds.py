"""Collect influential papers from seeds: resolve seeds, expand, score, write.

Entry point for the seed-based expansion pipeline (spec 11):

    seed content items (retrieved from the DB by research attribute id)
      -> SeedResolver (stored items -> seeds; skip-and-report unresolvable)
      -> SeedExpander (forward citations + relevance search + backward refs,
         per-generator budgets, dedup by work id, provenance)
      -> PaperResolver (unchanged: canonical resolution, drop unresolved)
      -> InfluenceAssessor (unchanged: fwci / citations-per-year -> 0-100)
      -> InfluenceFloorFilter (one-sided percentile floor of the seed set)
      -> ArxivPdfVerifier (unchanged: verified arXiv PDF url)
      -> PaperContentItemScorer (unchanged)
      -> PaperScoreFilter (unchanged)
      -> ContentItemDbWriter and/or JsonlWriter (unchanged)

Candidates that pass every stage are written exactly as the from-scratch
collection script writes them, so a paper discovered by expansion and a
paper discovered by web search are indistinguishable once stored, aside
from their recorded provenance.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mourat.collectors.seed_expander import SeedExpander
from mourat.data_models import (
    ContentItem,
    ContentItemCollection,
    PaperCandidateCollection,
    ResolvedPaperCollection,
    ScoredPaperCollection,
)
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.database.query_engine import (
    search_by_research_question,
    search_by_technical_challenge,
)
from mourat.filters import InfluenceFloorFilter
from mourat.monitoring import MonitoringHandler
from mourat.processors.arxiv_pdf_verifier import ArxivPdfVerifier
from mourat.processors.content_item_scorer import PaperContentItemScorer
from mourat.processors.influence_assessor import InfluenceAssessor
from mourat.resolvers.paper_resolver import PaperResolver
from mourat.resolvers.seed_resolver import SeedResolver
from mourat.utils.common import get_config_path

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_collect_influential_papers_from_seeds"


@contextmanager
def omegaconf_open_dict(cfg: DictConfig):
    """Temporarily unseal a struct-mode DictConfig so keys can be deleted."""
    old = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        yield cfg
    finally:
        OmegaConf.set_struct(cfg, old)


def _read_enabled(writer_cfg: DictConfig) -> bool:
    """Read the writer's enabled flag without mutating the struct-mode config."""
    enabled = writer_cfg.get("enabled", False)
    if enabled and "enabled" in writer_cfg:
        with omegaconf_open_dict(writer_cfg):
            del writer_cfg["enabled"]
    return bool(enabled)


def _retrieve_seed_items(conn, cfg: DictConfig) -> ContentItemCollection:
    """Seed content items linked to the configured research attributes.

    Merges the results of `search_by_research_question` and
    `search_by_technical_challenge` over the configured ids, de-duplicating
    by content-item id. An id with no linked items is reported and skipped,
    not silently ignored (FR1).
    """
    items: dict[str, ContentItem] = {}

    def _absorb(rows: list[dict], source: str) -> None:
        for row in rows:
            if row["id"] in items:
                continue
            try:
                items[row["id"]] = ContentItem.model_validate(dict(row))
            except Exception:
                logger.exception(
                    "seed content item '%s' from %s failed validation; skipped",
                    row.get("id"),
                    source,
                )

    for question_id in cfg.seed_research_question_ids:
        rows = search_by_research_question(conn, question_id)
        if not rows:
            logger.error(
                "no seed content items linked to research question '%s'", question_id
            )
        _absorb(rows, f"research question '{question_id}'")

    for challenge_id in cfg.seed_technical_challenge_ids:
        rows = search_by_technical_challenge(conn, challenge_id)
        if not rows:
            logger.error(
                "no seed content items linked to technical challenge '%s'",
                challenge_id,
            )
        _absorb(rows, f"technical challenge '{challenge_id}'")

    return ContentItemCollection(items=list(items.values()))


def _load_scoring_attributes(
    conn, cfg: DictConfig
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Load the research attributes the scorer scores candidates against.

    By default these are the same research questions and technical
    challenges the seeds came from (`seed_research_question_ids` /
    `seed_technical_challenge_ids`), plus the optional
    `research_topic_ids` / `constraint_ids` when configured. An id absent
    from the database is reported (logger.error) and skipped, never
    silently ignored — same convention as the from-scratch script.

    Returns (rq_list, tc_list, topic_list, constraint_list).
    """
    rq_list: list[dict] = []
    tc_list: list[dict] = []
    topic_list: list[dict] = []
    constraint_list: list[dict] = []

    def _absorb(record: dict | None, type_key: str, label: str, attr_id) -> None:
        if record is None:
            logger.error(
                "%s id '%s' not found in the database; skipped", label, attr_id
            )
            return
        entry = {
            "id": record["id"],
            "name": record["name"],
            "type": type_key,
            "description": record["description"] or "",
        }
        {
            "rq": rq_list,
            "tc": tc_list,
            "topic": topic_list,
            "constraint": constraint_list,
        }[type_key].append(entry)

    for attr_id in cfg.seed_research_question_ids:
        _absorb(
            rd.get_research_question(conn, attr_id), "rq", "research question", attr_id
        )
    for attr_id in cfg.seed_technical_challenge_ids:
        _absorb(
            bd.get_technical_challenge(conn, attr_id),
            "tc",
            "technical challenge",
            attr_id,
        )
    for attr_id in cfg.get("research_topic_ids", []):
        _absorb(
            rd.get_research_topic(conn, attr_id), "topic", "research topic", attr_id
        )
    for attr_id in cfg.get("constraint_ids", []):
        _absorb(bd.get_constraint(conn, attr_id), "constraint", "constraint", attr_id)

    return rq_list, tc_list, topic_list, constraint_list


def collect_influential_papers_from_seeds_main(cfg: DictConfig) -> None:
    """Main pipeline: seeds -> expand -> resolve -> floor -> score -> write."""
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
        seed_items = _retrieve_seed_items(conn, cfg)
        rq_list, tc_list, topic_list, constraint_list = _load_scoring_attributes(
            conn, cfg
        )
    finally:
        conn.close()

    if not seed_items.items:
        raise ValueError(
            "No seed content items resolved from the database; nothing to expand from. "
            "Check the configured seed attribute ids and the log for details."
        )

    seed_resolver: SeedResolver = hydra.utils.instantiate(cfg.seed_resolver)(
        monitoring_handler
    )
    step_id = "1"
    seeds = seed_resolver(seed_items, step_id=step_id)

    expander: SeedExpander = hydra.utils.instantiate(cfg.seed_expander)(
        monitoring_handler
    )
    step_id = "2"
    candidates: PaperCandidateCollection = expander(seeds, step_id=step_id)

    resolver: PaperResolver = hydra.utils.instantiate(cfg.paper_resolver)(
        monitoring_handler
    )
    step_id = "3"
    resolved: ResolvedPaperCollection = resolver(candidates, step_id=step_id)

    assessor: InfluenceAssessor = hydra.utils.instantiate(cfg.influence_assessor)(
        monitoring_handler
    )
    step_id = "4"
    assessed: ResolvedPaperCollection = assessor(resolved, step_id=step_id)

    # Seed-derived floor (FR4): percentile of the resolved seeds' influence.
    seed_influences = [
        seed.influence_value for seed in seeds.seeds if seed.influence_value is not None
    ]
    floor_filter: InfluenceFloorFilter = hydra.utils.instantiate(
        cfg.influence_floor_filter
    )(
        monitoring_handler,
        seed_influences=seed_influences,
    )
    step_id = "5"
    after_floor: ResolvedPaperCollection = floor_filter(assessed, step_id=step_id)

    verifier: ArxivPdfVerifier = hydra.utils.instantiate(cfg.arxiv_pdf_verifier)(
        monitoring_handler
    )
    step_id = "6"
    verified: ResolvedPaperCollection = verifier(after_floor, step_id=step_id)

    scoring_llm = hydra.utils.instantiate(cfg.scoring_llm)
    scorer: PaperContentItemScorer = hydra.utils.instantiate(cfg.content_item_scorer)(
        monitoring_handler,
        model=scoring_llm,
        rq_list=rq_list,
        tc_list=tc_list,
        topic_list=topic_list,
        constraint_list=constraint_list,
    )
    step_id = "7"
    scored: ScoredPaperCollection = scorer(verified, step_id=step_id)

    score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
    step_id = "8"
    filtered: ScoredPaperCollection = score_filter(scored, step_id=step_id)

    step_id = "9"
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
        "Pipeline complete: %d seeds, %d candidates, %d resolved, %d after floor, "
        "%d scored, %d after filter",
        len(seeds.seeds),
        len(candidates.papers),
        len(resolved.papers),
        len(after_floor.papers),
        len(scored.papers),
        len(filtered.papers),
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_influential_papers_from_seeds_main)()
