"""Collect posts from HackerNews, fetch, summarize, score, and save."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic_ai.models import Model

from mourat.clients.hn_algolia import HNAlgoliaClient
from mourat.data_models import HNPostCollection, ScoredHNPostCollection
from mourat.database import business_domain as bd
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.content_item_scorer import HNContentItemScorer
from mourat.processors.hn_fetcher import HNContentFetcher
from mourat.processors.hn_influence import HNInfluenceAssessor, HNInfluenceFilter
from mourat.processors.hn_summarizer import HNSummarizer
from mourat.utils.common import get_config_path
from mourat.utils.config import read_enabled

logger = logging.getLogger("mourat.scripts.collect_hn_posts")

CONFIG_NAME = "config_collect_hn_posts"


def _log_post_counts(step_id: str, stage: str, before: int, after: int) -> None:
    logger.info(
        "step %s | %s | posts in: %d, posts out: %d, filtered: %d",
        step_id,
        stage,
        before,
        after,
        before - after,
    )


def _write_posts(
    cfg: DictConfig,
    monitoring_handler: MonitoringHandler,
    db_path: Path,
    filtered_posts: ScoredHNPostCollection,
    step_id: str,
) -> None:
    """Run each independently enabled writer on the filtered posts."""
    db_writer_cfg = cfg.db_writer.copy()
    if read_enabled(db_writer_cfg):
        conn = create_connection(db_path)
        try:
            writer = hydra.utils.instantiate(db_writer_cfg)(
                monitoring_handler, conn=conn
            )
            writer(filtered_posts, step_id=step_id)
        finally:
            conn.close()

    jsonl_writer_cfg = cfg.jsonl_writer.copy()
    if read_enabled(jsonl_writer_cfg):
        writer = hydra.utils.instantiate(jsonl_writer_cfg)(monitoring_handler)
        writer(filtered_posts, step_id=step_id)


def collect_hn_posts_main(cfg: DictConfig) -> None:
    """Main pipeline: collect -> influence -> fetch -> summarize -> score -> save."""
    db_path = cfg.get("db_path")
    if db_path is None:
        raise ValueError("db_path not set in config")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )

    hn_client = HNAlgoliaClient()

    # Step 1: Collect
    collector = hydra.utils.instantiate(cfg.collector)(
        monitoring_handler, hn_client=hn_client
    )
    step_id = "1"
    raw_posts: HNPostCollection = collector({}, step_id=step_id)
    collected_count = len(raw_posts.posts)
    influence_filtered_count = collected_count
    logger.info(
        "step %s | HNPostCollector | collected %d posts", step_id, len(raw_posts.posts)
    )

    # Steps 1.5/1.6: normalized influence assessment and filter
    hn_influence_cfg = cfg.hn_influence
    if read_enabled(hn_influence_cfg):
        assessor = hydra.utils.instantiate(hn_influence_cfg.assessor)(
            monitoring_handler
        )
        raw_posts = assessor(raw_posts, step_id="1.5")
        influence_filter = hydra.utils.instantiate(hn_influence_cfg.filter)(
            monitoring_handler
        )
        filtered_raw = influence_filter(raw_posts, step_id="1.6")
        _log_post_counts(
            "1.6", "HNInfluenceFilter", len(raw_posts.posts), len(filtered_raw.posts)
        )
        raw_posts = filtered_raw
        influence_filtered_count = len(raw_posts.posts)

    # Step 2: Fetch linked content (posts with unavailable URLs are dropped)
    fetcher = hydra.utils.instantiate(cfg.hn_fetcher)(monitoring_handler)
    step_id = "2"
    fetched_posts: HNPostCollection = fetcher(raw_posts, step_id=step_id)
    _log_post_counts(
        step_id, "HNContentFetcher", len(raw_posts.posts), len(fetched_posts.posts)
    )

    # Step 3: Summarize (LLM); unparseable output retried then dropped
    summarization_llm: Model = hydra.utils.instantiate(cfg.summarization_llm)
    summarizer = hydra.utils.instantiate(cfg.hn_summarizer)(
        monitoring_handler, model=summarization_llm
    )
    step_id = "3"
    summarized_posts: HNPostCollection = summarizer(fetched_posts, step_id=step_id)
    _log_post_counts(
        step_id, "HNSummarizer", len(fetched_posts.posts), len(summarized_posts.posts)
    )

    # Step 4: Score (sees title + description only)
    conn = create_connection(db_path)
    try:
        rq_list = rd.list_research_questions(conn)
        tc_list = bd.list_technical_challenges(conn)
        topic_list = rd.list_research_topics(conn)
        constraint_list = bd.list_constraints(conn)
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

    scoring_llm: Model = hydra.utils.instantiate(cfg.scoring_llm)
    scorer: HNContentItemScorer = hydra.utils.instantiate(cfg.scorer)(
        monitoring_handler,
        model=scoring_llm,
        rq_list=scoring_rq_list,
        tc_list=scoring_tc_list,
        topic_list=scoring_topic_list,
        constraint_list=scoring_constraint_list,
    )
    step_id = "4"
    scored_posts: ScoredHNPostCollection = scorer(summarized_posts, step_id=step_id)
    _log_post_counts(
        step_id,
        "HNContentItemScorer",
        len(summarized_posts.posts),
        len(scored_posts.posts),
    )

    # Step 5: Filter by score
    score_filter = hydra.utils.instantiate(cfg.score_filter)(monitoring_handler)
    step_id = "5"
    filtered_posts: ScoredHNPostCollection = score_filter(scored_posts, step_id=step_id)
    _log_post_counts(
        step_id, "HNScoreFilter", len(scored_posts.posts), len(filtered_posts.posts)
    )

    # Step 6: Write independently configured outputs.
    step_id = "6"
    _write_posts(cfg, monitoring_handler, db_path, filtered_posts, step_id)

    logger.info(
        "Pipeline complete: %d collected, %d after influence filter, "
        "%d fetched, %d summarized, %d scored, %d after score filter",
        collected_count,
        influence_filtered_count,
        len(fetched_posts.posts),
        len(summarized_posts.posts),
        len(scored_posts.posts),
        len(filtered_posts.posts),
    )


def main(cfg: DictConfig) -> None:
    collect_hn_posts_main(cfg)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
