"""Tests for configurable post writer orchestration (spec 15)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mourat.data_models import ScoredRedditPostCollection
from mourat.scripts.collect_posts import _write_posts
from mourat.utils.config import read_enabled


def test_post_count_log_reports_in_out_and_filtered(caplog):
    import mourat.scripts.collect_posts as script

    assert script.logger.name == "mourat.scripts.collect_posts"
    with caplog.at_level("INFO", logger=script.logger.name):
        script._log_post_counts("2", "HeuristicSlopFilter", 12, 8)

    assert (
        "step 2 | HeuristicSlopFilter | posts in: 12, posts out: 8, filtered: 4"
        in caplog.text
    )


@pytest.mark.parametrize(
    "db_enabled,jsonl_enabled",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_writer_enable_flags_write_only_to_enabled_outputs(
    tmp_path, monkeypatch, db_enabled, jsonl_enabled
):
    import mourat.scripts.collect_posts as script

    monkeypatch.setattr(script, "read_enabled", read_enabled)
    filtered = ScoredRedditPostCollection(posts=[])
    calls = []

    class FakeConnection:
        def close(self):
            pass

    conn = FakeConnection()
    monkeypatch.setattr(script, "create_connection", lambda path: conn)
    monkeypatch.setattr(script, "read_enabled", read_enabled)

    class FakeWriter:
        def __init__(self, kind):
            self.kind = kind

        def __call__(self, collection, step_id):
            calls.append((self.kind, collection, step_id))
            return collection

    def instantiate(cfg):
        return lambda handler, **kwargs: FakeWriter(
            "db" if "conn" in kwargs else "jsonl"
        )

    monkeypatch.setattr(script.hydra.utils, "instantiate", instantiate)
    cfg = OmegaConf.create(
        {
            "db_writer": {"enabled": db_enabled},
            "jsonl_writer": {"enabled": jsonl_enabled},
        }
    )

    _write_posts(cfg, object(), Path(tmp_path / "posts.db"), filtered, "8")

    assert [(kind, collection, step) for kind, collection, step in calls] == [
        (kind, filtered, "8")
        for kind, enabled in (("db", db_enabled), ("jsonl", jsonl_enabled))
        if enabled
    ]


def test_read_enabled_removes_flag_from_struct_config_and_restores_struct_mode():
    from omegaconf import OmegaConf

    config = OmegaConf.create({"enabled": True, "_target_": "some.Writer"})
    OmegaConf.set_struct(config, True)

    assert read_enabled(config) is True
    assert "enabled" not in config
    assert OmegaConf.is_struct(config) is True


def test_post_writer_configs_compose_disabled_by_default():
    from hydra import compose, initialize_config_dir

    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "config"), version_base="1.3"):
        config = compose(config_name="config_collect_posts")

    assert config.db_writer.enabled is False
    assert config.db_writer._target_ == (
        "mourat.writers.post_writers.PostContentItemDbWriter"
    )
    assert config.jsonl_writer.enabled is False
    assert config.jsonl_writer._target_ == "mourat.writers.post_writers.PostJsonlWriter"
    assert config.jsonl_writer.output_path.endswith("/posts.jsonl")
