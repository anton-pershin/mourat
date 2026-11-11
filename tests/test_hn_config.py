"""Test HN configuration and dependency selection (specs 19 and 24)."""

import tomllib
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_config_collect_hn_posts_composes(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_PROXY", "http://proxy.example:8080")
    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config_collect_hn_posts")

    # Required keys present
    assert "collector" in cfg
    assert cfg.collector.tags == ["story"]  # Includes Ask HN and Show HN stories.
    assert "min_points" in cfg.collector
    assert "hn_influence" in cfg
    assert cfg.hn_influence.min_influence == 50
    assert "reference_level" in cfg.hn_influence
    assert "hn_fetcher" in cfg
    assert cfg.hn_fetcher.proxy == "http://proxy.example:8080"
    assert cfg.hn_fetcher.timeout == 30
    assert cfg.hn_fetcher.connection_timeout == 10
    assert "hn_summarizer" in cfg
    assert "retries" in cfg.hn_summarizer
    assert "scorer" in cfg
    assert (
        cfg.scorer._target_
        == "mourat.processors.content_item_scorer.HNContentItemScorer"
    )
    assert cfg.score_filter._target_ == "mourat.filters.HNScoreFilter"
    # writers with enabled guards
    assert cfg.db_writer.platform_id == "hackernews"
    assert cfg.db_writer.influence_metric_id == "hn_points"
    assert cfg.jsonl_writer.output_path.endswith("hn_posts.jsonl")
    # no slop stages anywhere
    assert "slop_classifier" not in cfg
    assert "slop_verdict_filter" not in cfg
    assert "enricher" not in cfg


def test_enricher_and_discovery_configs_compose_proxy(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_PROXY", "http://proxy.example:8080")
    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        posts_cfg = compose(config_name="config_collect_posts")
        papers_cfg = compose(
            config_name="config_collect_influential_papers_from_scratch"
        )
    assert posts_cfg.enricher.proxy == "http://proxy.example:8080"
    assert posts_cfg.enricher.overall_timeout == 600
    assert posts_cfg.enrichment.overall_timeout == 600
    assert papers_cfg.paper_discoverer.proxy == "http://proxy.example:8080"


def test_calibration_config_keys():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/config_calibrate_hn_stats.yaml")
    for key in [
        "sample_limit",
        "min_sample",
        "percentiles",
        "output_path",
        "mode",
        "tags",
    ]:
        assert key in cfg, key


def test_hn_article_contract_dependencies_are_pinned():
    # Spec 24 T8: runtime installation selects the same reviewed contract.
    project_root = Path(__file__).resolve().parent.parent
    with (project_root / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    requirement_dependencies = [
        line.split("#", 1)[0].strip()
        for line in (project_root / "requirements.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    expected_comourat = (
        "comourat @ git+https://github.com/anton-pershin/comourat.git@"
        "f15a0375889464d44f8ca93991fc32dd9b43a2a5"
    )
    for dependencies in (project_dependencies, requirement_dependencies):
        assert expected_comourat in dependencies
        assert "pydantic-ai >= 2.64.0" in dependencies
