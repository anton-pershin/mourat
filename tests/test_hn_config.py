"""Test that the HN collection config composes (spec 19, T19)."""

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_config_collect_hn_posts_composes():
    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config_collect_hn_posts")

    # Required keys present
    assert "collector" in cfg
    assert cfg.collector.tags == ["story", "ask_hn", "show_hn"]
    assert "min_points" in cfg.collector
    assert "hn_influence" in cfg
    assert cfg.hn_influence.min_influence == 50
    assert "reference_level" in cfg.hn_influence
    assert "hn_fetcher" in cfg
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
