"""Collect posts from web resources, enrich, score, and save to database."""

import logging
from pathlib import Path

import hydra
import praw
from omegaconf import DictConfig
from pydantic_ai.models import Model

from mourat.data_models import RedditPostCollection, ScoredRedditPostCollection
from mourat.database import business_domain as bd
from mourat.database import content_item as ci
from mourat.database import create_connection
from mourat.database import research_domain as rd
from mourat.monitoring import MonitoringHandler
from mourat.processors.post_scorer import PostScorer
from mourat.utils.common import get_config_path

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_collect_posts"


def save_posts_to_db(conn, scored: ScoredRedditPostCollection) -> int:
    """Save scored posts to the database as ContentItem records."""
    saved = 0

    # Ensure source type and platform exist
    try:
        ci.create_source_type(conn, "post", "Post", "Social media or blog post")
    except Exception:
        pass
    try:
        ci.create_platform(conn, "reddit", "Reddit", "Reddit platform")
    except Exception:
        pass
    try:
        ci.create_influence_metric(conn, "upvotes", "Upvotes", "Reddit upvote count")
    except Exception:
        pass

    for sp in scored.posts:
        item_id = f"reddit_{sp.post.submission_id}"
        influence_score = min(100, sp.post.score)

        try:
            ci.create_content_item(
                conn,
                id=item_id,
                name=sp.post.title,
                source_type_id="post",
                platform_id="reddit",
                influence_metric_id="upvotes",
                description=sp.post.text or "",
                url=sp.post.url,
                published_at=sp.post.date,
                authors=sp.post.author,
                influence_score=influence_score,
            )
        except Exception:
            continue  # Already exists, skip

        for score_entry in sp.relevance_scores:
            rid = score_entry.id
            rtype = score_entry.type
            score_val = score_entry.score
            justification = score_entry.justification

            if rid and rtype and score_val is not None:
                if rtype == "rq":
                    try:
                        ci.add_item_research_question(
                            conn, item_id, rid, justification, int(score_val)
                        )
                    except Exception:
                        pass
                elif rtype == "tc":
                    try:
                        ci.add_item_technical_challenge(
                            conn, item_id, rid, justification, int(score_val)
                        )
                    except Exception:
                        pass
                elif rtype == "topic":
                    try:
                        ci.add_item_research_topic(
                            conn, item_id, rid, justification, int(score_val)
                        )
                    except Exception:
                        pass
                elif rtype == "constraint":
                    try:
                        ci.add_item_constraint(
                            conn, item_id, rid, justification, int(score_val)
                        )
                    except Exception:
                        pass

        saved += 1

    return saved


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
    scorer: PostScorer = hydra.utils.instantiate(cfg.scorer)(
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

    # Step 8: Save
    conn = create_connection(db_path)
    try:
        saved = save_posts_to_db(conn, filtered_posts)
    finally:
        conn.close()

    logger.info(
        "Pipeline complete: %d collected, %d after heuristic filter, "
        "%d after slop filter, %d enriched, %d scored, %d saved to database",
        len(raw_posts.posts),
        len(heuristic_posts.posts),
        len(posts.posts),
        len(enriched_posts.posts),
        len(scored_posts.posts),
        saved,
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_posts_main)()
