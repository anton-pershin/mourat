"""Unit tests for the HN stats calibration script (spec 19)."""

import pytest

from mourat.scripts.calibrate_hn_stats import (
    _append_or_write,
    _compute_stats,
    _collect_scores,
    _render_block,
)

NOW = 1_780_000_000


class _StubClient:
    """Stub Algolia client returning fixed hits for the window query."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_by_date(self, lo, hi, min_points=0, tags=None, page_size=1000):
        self.calls.append({"lo": lo, "hi": hi, "min_points": min_points, "tags": tags})
        return [h for h in self.hits if lo <= h["created_at_i"] < hi]


def test_compute_stats_matches_reference():
    # T8 (R5): stats computed independently over known points.
    scores = list(range(1, 31))
    stats = _compute_stats(scores, min_sample=30)
    assert stats["n"] == 30
    assert stats["median"] == 15.5
    assert stats["mean"] == sum(scores) / 30
    assert stats["p10"] == sorted(scores)[2]
    assert stats["p90"] == sorted(scores)[26]
    assert stats["max"] == 30
    assert "insufficient" not in stats


def test_compute_stats_extra_percentiles():
    # T8 (R5): extra percentiles emitted nearest-rank, both scales accepted.
    scores = list(range(1, 101))
    stats = _compute_stats(scores, min_sample=30, percentiles=[0.7, 70])
    assert stats["p70"] == sorted(scores)[69]
    assert "p7" not in stats  # 70 (0-100 scale) maps to p70, not p7


def test_compute_stats_insufficient():
    # T8 (R5): insufficient sample -> insufficient: true, no median.
    stats = _compute_stats([1, 2, 3], min_sample=30)
    assert stats["insufficient"] is True
    assert stats["n"] == 3
    assert "median" not in stats


def test_collect_scores_window_bounds():
    # T9 (R5): only posts inside [now-48h, now-24h] counted; bounded window.
    hits = [
        {"created_at_i": NOW - 86400, "points": 999},  # too fresh
        {"created_at_i": NOW - 25 * 3600, "points": 10},  # in window
        {"created_at_i": NOW - 40 * 3600, "points": 20},  # in window
        {"created_at_i": NOW - 72 * 3600, "points": 777},  # too old
    ]
    client = _StubClient(hits)
    scores = _collect_scores(client, now=NOW)
    assert scores == [10, 20]
    assert client.calls[0]["lo"] == NOW - 48 * 3600
    assert client.calls[0]["hi"] == NOW - 24 * 3600


def test_render_and_append_or_write(tmp_path):
    # T10 (R5): overwrite/append semantics identical to spec 16.
    out = tmp_path / "stats.yaml"
    block = _render_block({"p90": 19, "n": 500}, "2026-10-15T00:00:00Z")
    _append_or_write(out, block, mode="overwrite")
    first = out.read_text()
    assert "p90" in first
    _append_or_write(out, block.replace("p90", "p91"), mode="append")
    content = out.read_text()
    assert content.count("2026-10-15T00:00:00Z") == 2
    # overwrite replaces entirely
    _append_or_write(out, block, mode="overwrite")
    assert out.read_text().count("p91") == 0


def test_config_composes():
    # T11 (R5): calibration config composes and exposes all keys.
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
    assert cfg.mode == "overwrite"
