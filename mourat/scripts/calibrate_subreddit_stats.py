"""Calibrate per-subreddit upvote-score statistics.

Measures the recent-post score background of configured subreddits and writes
a YAML reference file. The reference levels (e.g., medians) are meant to be
copied into pipeline config and used as normalization references for Reddit
influence scores.

    python -m mourat.scripts.calibrate_subreddit_stats

No database access, no LLM calls: a measurement tool, not a pipeline step.
"""

import datetime
import logging
import math
import time
from pathlib import Path

import hydra
import praw
import yaml
from omegaconf import DictConfig

from mourat.monitoring import MonitoringHandler
from mourat.utils.common import get_config_path

logger = logging.getLogger("mourat.scripts.calibrate_subreddit_stats")

CONFIG_NAME = "config_calibrate_subreddit_stats"


def _nearest_rank(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of an ascending-sorted list."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _collect_scores(
    subreddit,
    time_window: dict,
    sample_limit: int,
    now_utc: float,
) -> list[int]:
    """Collect scores of posts inside the time window (newest first).

    Mirrors RedditPostCollector's iteration: stop at the first post older
    than the window cutoff. No text filter, no top-K cut.
    """
    cutoff = now_utc - datetime.timedelta(**time_window).total_seconds()
    scores: list[int] = []
    for post in subreddit.new(limit=sample_limit):
        if post.created_utc < cutoff:
            break
        scores.append(post.score)
    return scores


def _compute_stats(scores: list[int], min_sample: int) -> dict:
    """Compute the reference statistics or mark the sample insufficient."""
    n = len(scores)
    if n < min_sample:
        return {"insufficient": True, "n": n}
    ordered = sorted(scores)
    median = float(
        sum(ordered[n // 2 - 1 : n // 2 + 1]) / 2 if n % 2 == 0 else ordered[n // 2]
    )
    return {
        "n": n,
        "median": median,
        "mean": sum(ordered) / n,
        "p10": _nearest_rank(ordered, 0.1),
        "p90": _nearest_rank(ordered, 0.9),
        "max": ordered[-1],
    }


def _format_window(time_window: dict) -> str:
    unit_symbols = {
        "weeks": "w",
        "days": "d",
        "hours": "h",
        "minutes": "m",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
    }
    parts = [
        f"{time_window[unit]}{symbol}"
        for unit, symbol in unit_symbols.items()
        if unit in time_window
    ]
    return "+".join(parts) if parts else str(dict(time_window))


def _render_block(
    stats_by_subreddit: dict,
    timestamp: str,
    window: dict,
    sample_limit: int,
) -> str:
    """Render one self-contained YAML block with a header comment."""
    header = (
        f"# calibrated {timestamp} | window: {_format_window(window)}"
        f" | sample_limit: {sample_limit}\n"
    )
    body = yaml.safe_dump(
        {"subreddits": stats_by_subreddit},
        default_flow_style=False,
        sort_keys=True,
    )
    return header + body


def _append_or_write(path: Path, block: str, mode: str) -> None:
    """Write the block according to the configured mode."""
    if mode == "overwrite":
        path.write_text(block, encoding="utf-8")
    elif mode == "append":
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            path.write_text(existing + block, encoding="utf-8")
        else:
            path.write_text(block, encoding="utf-8")
    else:
        raise ValueError(f"unknown mode: {mode}")


def calibrate_subreddit_stats_main(cfg: DictConfig) -> None:
    """Measure per-subreddit score statistics and write the output YAML."""
    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )

    reddit = praw.Reddit(
        client_id=cfg.user_settings.reddit.client_id,
        client_secret=cfg.user_settings.reddit.client_secret,
        user_agent=cfg.user_settings.reddit.user_agent,
    )
    reddit.read_only = True

    stats_by_subreddit: dict = {}
    for subreddit_name in cfg.subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        scores = _collect_scores(
            subreddit,
            dict(cfg.time_window),
            cfg.sample_limit,
            now_utc=time.time(),
        )
        stats = _compute_stats(scores, cfg.min_sample)
        stats_by_subreddit[subreddit_name] = stats
        if "median" in stats:
            logger.info(
                "r/%s: n=%d, median=%.1f, p10=%d, p90=%d",
                subreddit_name,
                stats["n"],
                stats["median"],
                stats["p10"],
                stats["p90"],
            )
        else:
            logger.info(
                "r/%s: insufficient sample (n=%d < %d)",
                subreddit_name,
                stats["n"],
                cfg.min_sample,
            )

    text_for_monitoring = "\n".join(
        f"- r/{name}: {stats}" for name, stats in stats_by_subreddit.items()
    )
    monitoring_handler("1", text_for_monitoring)

    block = _render_block(
        stats_by_subreddit,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        window=dict(cfg.time_window),
        sample_limit=cfg.sample_limit,
    )
    _append_or_write(Path(cfg.output_path), block, cfg.mode)
    logger.info("stats written to %s (mode=%s)", cfg.output_path, cfg.mode)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(calibrate_subreddit_stats_main)()
