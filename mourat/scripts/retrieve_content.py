"""Retrieve content items from the database via CLI."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.table import Table

from mourat.database import create_connection
from mourat.database.query_engine import (
    search_by_influence_score,
    search_by_keywords,
    search_by_research_question,
    search_by_research_topic,
    search_by_technical_challenge,
)
from mourat.utils.common import get_config_path
from mourat.utils.console import console

logger = logging.getLogger(__name__)

CONFIG_NAME = "config_retrieve_content"


def run_query(conn, cfg):
    """Execute the configured query and display results."""
    query_type = cfg.get("query_type", "keywords")

    if query_type == "keywords":
        results = search_by_keywords(
            conn,
            cfg.keyword_query,
            min_relevance=cfg.get("min_relevance_score"),
        )
    elif query_type == "research_question":
        results = search_by_research_question(
            conn,
            cfg.research_question_id,
            min_score=cfg.get("min_relevance_score"),
        )
    elif query_type == "technical_challenge":
        results = search_by_technical_challenge(
            conn,
            cfg.technical_challenge_id,
            min_score=cfg.get("min_relevance_score"),
        )
    elif query_type == "research_topic":
        results = search_by_research_topic(
            conn,
            cfg.research_topic_id,
            min_score=cfg.get("min_relevance_score"),
        )
    elif query_type == "influence_score":
        results = search_by_influence_score(
            conn,
            cfg.min_influence_score,
            max_score=cfg.get("max_influence_score", 100),
        )
    else:
        logger.error("Unknown query type: %s", query_type)
        return

    if not results:
        console.print("No content items found.")
        return

    table = Table(title=f"Found {len(results)} content item(s)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Score", justify="right")
    table.add_column("URL")

    for item in results:
        score = (
            str(item["influence_score"])
            if item["influence_score"] is not None
            else "N/A"
        )
        url = item["url"] or ""
        table.add_row(item["id"], item["name"], score, url)

    console.print(table)


def retrieve_content(cfg: DictConfig) -> None:
    db_path = cfg.get("db_path")
    if db_path is None:
        logger.error(
            "Database path not set in config (db_path). "
            "Set db_path in config_retrieve_content.yaml or via Hydra override."
        )
        return

    db_path = Path(db_path)
    if not db_path.exists():
        logger.error(
            "Database not found: %s. "
            "Create the database first before running retrieve_content.",
            db_path,
        )
        return

    conn = create_connection(db_path)
    try:
        run_query(conn, cfg)
    finally:
        conn.close()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(retrieve_content)()
