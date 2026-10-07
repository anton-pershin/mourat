"""Collect posts from web resources, enrich, score, and save to database."""

import logging
from contextlib import contextmanager
from pathlib import Path

import hydra
import praw
from omegaconf import DictConfig, OmegaConf
from pydantic_ai.models import Model

from mourat.data_models import RedditPostCollection, ScoredRedditPostCollection
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.content_item_scorer import PostContentItemScorer
from mourat.utils.common import get_config_path

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_collect_posts"


@contextmanager
def omegaconf_open_dict(cfg: DictConfig):
    old = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        yield cfg
    finally:
        OmegaConf.set_struct(cfg, old)


def _read_enabled(writer_cfg: DictConfig) -> bool:
    enabled = writer_cfg.get("enabled", False)
    if enabled and "enabled" in writer_cfg:
        with omegaconf_open_dict(writer_cfg):
            del writer_cfg["enabled"]
    return bool(enabled)


def _write_posts(
    cfg: DictConfig,
    monitoring_handler: MonitoringHandler,
    db_path: Path,
    filtered_posts: ScoredRedditPostCollection,
    step_id: str,
) -> None:
    """Run each independently enabled writer on the filtered posts."""
    db_writer_cfg = cfg.db_writer.copy()
    if _read_enabled(db_writer_cfg):
        conn = create_connection(db_path)
        try:
            writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=conn
            )
            writer(filtered_posts, step_id=step_id)
        finally:
            conn.close()

    jsonl_writer_cfg = cfg.jsonl_writer.copy()
    if _read_enabled(jsonl_writer_cfg):
        writer = hydra.utils.instantiate(jsonl_writer_cfg)(monitoring_handler)
        writer(filtered_posts, step_id=step_id)


def collect_posts_main(cfg: DictConfig) -> None:
    """Main pipeline: collect -> enrich -> score -> save."""
    db_path = cfg.get("db_path")
    if db_path is None:
        raise ValueError("db_path not set in config")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )

    # Initialize Reddit client
    reddit = praw.Reddit(
        client_id=cfg.user_settings.reddit.client_id,
        client_secret=cfg.user_settings.reddit.client_secret,
        user_agent=cfg.user_settings.reddit.user_agent,
    )

    # Step 1: Collect
    collector = hydra.utils.instantiate(cfg.collector)(
        monitoring_handler, reddit_client=reddit
    )
    step_id = "1"
    raw_posts: RedditPostCollection = collector({}, step_id=step_id)

    # Step 2: Heuristic slop filter
    heuristic_slop_filter = hydra.utils.instantiate(cfg.slop_filter)(monitoring_handler)
    step_id = "2"
    heuristic_posts: RedditPostCollection = heuristic_slop_filter(
        raw_posts, step_id=step_id
    )

    # Step 3: Slop classification
    classification_llm: Model = hydra.utils.instantiate(cfg.classification_llm)
    slop_classifier = hydra.utils.instantiate(cfg.slop_classifier)(
        monitoring_handler, model=classification_llm
    )
    step_id = "3"
    classified_posts = slop_classifier(heuristic_posts, step_id=step_id)

    # Step 4: Slop verdict filter
    slop_verdict_filter = hydra.utils.instantiate(cfg.slop_verdict_filter)(
        monitoring_handler
    )
    step_id = "4"
    posts: RedditPostCollection = slop_verdict_filter(classified_posts, step_id=step_id)

    # Step 5: Enrich
    enrichment_llm: Model = hydra.utils.instantiate(cfg.enrichment_llm)
    enricher = hydra.utils.instantiate(cfg.enricher)(
        monitoring_handler, model=enrichment_llm
    )
    step_id = "5"
    enriched_posts = enricher(posts, step_id=step_id)

    # Step 5.5: Load research attributes from database
    conn = create_connection(db_path)
    try:
        rq_list = rd.list_research_questions(conn)
        tc_list = bd.list_technical_challenges(conn)
        topic_list = rd.list_research_topics(conn)
        constraint_list = bd.list_constraints(conn)
        # Format for the scorer: each entry needs id, name, description
        scoring_rq_list = [
            {
                "id": r["id"],
                "name": r["name"],
                "type": "rq",
                "description": r["description"],
            }
            for r in rq_list
        ]
        scoring_tc_list = [
            {
                "id": t["id"],
                "name": t["name"],
                "type": "tc",
                "description": t["description"],
            }
            for t in tc_list
        ]
        scoring_topic_list = [
            {
                "id": t["id"],
                "name": t["name"],
                "type": "topic",
                "description": t["description"],
            }
            for t in topic_list
        ]
        scoring_constraint_list = [
            {
                "id": c["id"],
                "name": c["name"],
                "type": "constraint",
                "description": c["description"],
            }
            for c in constraint_list
        ]
    finally:
        conn.close()

    # Step 6: Score
    scoring_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
    scorer: PostContentItemScorer = hydra.utils.instantiate(cfg.scorer)(
        monitoring_handler,
        model=scoring_llm,
        rq_list=scoring_rq_list,
        tc_list=scoring_tc_list,
        topic_list=scoring_topic_list,
        constraint_list=scoring_constraint_list,
    )
    step_id = "6"
    scored_posts: ScoredRedditPostCollection = scorer(enriched_posts, step_id=step_id)

    # Step 7: Filter by score
    score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
    step_id = "7"
    filtered_posts: ScoredRedditPostCollection = score_filter(
        scored_posts, step_id=step_id
    )

    # Step 8: Write independently configured outputs.
    step_id = "8"
    _write_posts(cfg, monitoring_handler, db_path, filtered_posts, step_id)

    logger.info(
        "Pipeline complete: %d collected, %d after heuristic filter, "
        "%d after slop filter, %d enriched, %d scored, %d after score filter",
        len(raw_posts.posts),
        len(heuristic_posts.posts),
        len(posts.posts),
        len(enriched_posts.posts),
        len(scored_posts.posts),
        len(filtered_posts.posts),
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_posts_main)()
