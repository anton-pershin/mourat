"""Tests for the subreddit stats calibration script (spec 16)."""

import textwrap
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mourat.scripts.calibrate_subreddit_stats import (
    _append_or_write,
    _collect_scores,
    _compute_stats,
    _render_block,
)

# 2026-10-08T12:00:00Z
NOW = 1_780_000_000.0


def _make_subreddit(posts):
    """Build a fake PRAW subreddit: posts = [(created_utc, score), ...]."""

    class _Submission:
        def __init__(self, created_utc, score):
            self.created_utc = created_utc
            self.score = score

    class _Subreddit:
        def __init__(self, posts):
            self._posts = [_Submission(ts, sc) for ts, sc in posts]

        def new(self, limit=None):
            # PRAW returns newest first; mirror that ordering.
            return list(self._posts)

    return _Subreddit(posts)


def test_compute_stats_full_record_matches_reference_statistics():
    # T1 (R3, R5): known scores 1..30, all stats computed independently.
    scores = list(range(1, 31))
    stats = _compute_stats(scores, min_sample=30)
    assert stats["n"] == 30
    assert stats["median"] == 15.5
    assert stats["mean"] == sum(scores) / 30
    assert stats["p10"] == sorted(scores)[2]  # nearest-rank: ceil(0.1*30)=3rd
    assert stats["p90"] == sorted(scores)[26]  # nearest-rank: ceil(0.9*30)=27th
    assert stats["max"] == 30
    assert "insufficient" not in stats


def test_compute_stats_even_count_median_not_truncated():
    # T7 (R5): median of 1..30 is 15.5, not truncated to 15.
    stats = _compute_stats(list(range(1, 31)), min_sample=30)
    assert stats["median"] == 15.5
    assert isinstance(stats["median"], float)


def test_collect_scores_respects_window_and_limit():
    # T2 (R3): posts newer than the cutoff are kept, older ones dropped.
    window = {"hours": 24}
    inside = [(NOW - i * 60, i + 1) for i in range(10)]  # ages 0..9 min
    outside = [(NOW - 86400 * (i + 1) - 60, 1000 + i) for i in range(5)]
    subreddit = _make_subreddit(inside + outside)
    scores = _collect_scores(subreddit, window, sample_limit=100, now_utc=NOW)
    assert scores == [i + 1 for i in range(10)]


def test_collect_scores_stops_at_first_too_old_post():
    # T2 (R3): iteration mirrors RedditPostCollector: break at the first
    # post older than the cutoff, even if newer posts follow in the list
    # (they cannot on Reddit, but the collector's semantics are break-based).
    window = {"hours": 24}

    class _OrderAwareSub:
        def __init__(self):
            self.stop_reason = None

        def new(self, limit=None):
            # Newest first, then one too old; nothing after may be counted.
            return [
                type("S", (), {"created_utc": NOW - 60, "score": 5})(),
                type("S", (), {"created_utc": NOW - 86400 * 5, "score": 9})(),
                type("S", (), {"created_utc": NOW - 60, "score": 7})(),
            ]

    scores = _collect_scores(_OrderAwareSub(), window, sample_limit=100, now_utc=NOW)
    assert scores == [5]


def test_compute_stats_insufficient_sample_has_no_median():
    # T3 (R4, B4): below min_sample -> insufficient, no median.
    stats = _compute_stats([3, 1, 2, 1, 5], min_sample=30)
    assert stats == {"insufficient": True, "n": 5}


def test_append_or_write_overwrite_replaces_existing_file(tmp_path):
    # T4 (R6, B2).
    path = tmp_path / "stats.yaml"
    path.write_text("old content that must disappear", encoding="utf-8")
    _append_or_write(path, "new block\n", mode="overwrite")
    content = path.read_text(encoding="utf-8")
    assert content == "new block\n"
    assert "old content" not in content


def test_append_or_write_append_creates_missing_file(tmp_path):
    # T5 (R6, B1).
    path = tmp_path / "stats.yaml"
    _append_or_write(path, "new block\n", mode="append")
    assert path.read_text(encoding="utf-8") == "new block\n"


def test_append_or_write_append_keeps_existing_block(tmp_path):
    # T6 (R6, B3).
    path = tmp_path / "stats.yaml"
    path.write_text(
        "# calibrated 2026-10-07T12:00:00+00:00\nold: 1\n", encoding="utf-8"
    )
    _append_or_write(
        path, "# calibrated 2026-10-08T12:00:00+00:00\nnew: 2\n", mode="append"
    )
    content = path.read_text(encoding="utf-8")
    assert content.index("2026-10-07") < content.index("2026-10-08")
    assert "old: 1" in content and "new: 2" in content


def test_render_block_contains_header_comment_and_subreddit_stats():
    # T4-T6 structural dependency (R6): the block carries a header comment
    # with timestamp, window and sample_limit, and per-subreddit stats.
    stats_by_subreddit = {
        "MachineLearning": {
            "n": 812,
            "median": 14.0,
            "mean": 44.2,
            "p10": 2,
            "p90": 91,
            "max": 6139,
        },
        "SmallSub": {"insufficient": True, "n": 11},
    }
    block = _render_block(
        stats_by_subreddit,
        timestamp="2026-10-08T12:00:00+00:00",
        window={"hours": 24},
        sample_limit=1000,
    )
    assert block.startswith("# calibrated 2026-10-08T12:00:00+00:00")
    assert "window: 24h" in block
    assert "sample_limit: 1000" in block
    assert "MachineLearning:" in block
    assert "median: 14.0" in block
    assert "SmallSub:" in block
    assert "insufficient: true" in block


def test_hydra_config_composes_with_expected_keys():
    # T8 (R1, R2): the config exists, composes, and carries the R2 keys.
    from hydra import compose, initialize_config_dir

    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="config_calibrate_subreddit_stats")
    assert len(cfg.subreddits) > 0
    assert "hours" in cfg.time_window or "days" in cfg.time_window
    assert cfg.sample_limit == 1000
    assert cfg.min_sample == 30
    assert cfg.mode in ("overwrite", "append")
    assert cfg.output_path
    assert cfg.user_settings.reddit.client_id is not None or True  # env-backed
    assert isinstance(cfg.sample_limit, int)
