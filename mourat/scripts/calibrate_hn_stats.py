"""HN stats calibration script: measures the points reference level.

Samples stories published in [now-48h, now-24h] (yesterday's cohort) —
HN points saturate within ~24h, so scores in this window are final.
"""

from __future__ import annotations

import datetime
import logging
import statistics
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mourat.clients.hn_algolia import HNAlgoliaClient
from mourat.monitoring import MonitoringHandler
from mourat.utils.common import get_config_path

logger = logging.getLogger("mourat.scripts.calibrate_hn_stats")

CONFIG_NAME = "config_calibrate_hn_stats"


def _compute_stats(
    scores: list[int], min_sample: int, percentiles: list[float] | None = None
) -> dict:
    """Compute summary statistics over story points.

    `percentiles` accepts BOTH scales: fractions in [0, 1] and percentages in
    [0, 100]. Keys are emitted nearest-rank as pXX (e.g. p70, p80).
    """
    if len(scores) < min_sample:
        return {"insufficient": True, "n": len(scores)}

    ordered = sorted(scores)
    n = len(ordered)

    def nearest_rank(p: float) -> int:
        # p arrives as a fraction in [0, 1]
        return ordered[min(n, max(1, int(-(-(n * p) // 1)))) - 1] if n else 0

    stats: dict = {
        "n": n,
        "median": statistics.median(scores),
        "mean": statistics.mean(scores),
        "p10": nearest_rank(0.1),
        "p90": nearest_rank(0.9),
        "max": ordered[-1],
    }
    for value in percentiles or []:
        fraction = value if value <= 1 else value / 100
        key = f"p{round(fraction * 100)}"
        stats[key] = nearest_rank(fraction)
    return stats


def _collect_scores(client: HNAlgoliaClient, now: int | None = None) -> list[int]:
    """Collect story points published in [now-48h, now-24h]."""
    now = (
        now if now is not None else int(datetime.datetime.now(datetime.UTC).timestamp())
    )
    lo = now - 48 * 3600
    hi = now - 24 * 3600
    hits = client.search_by_date(lo=lo, hi=hi, min_points=0, tags=["story"])
    return [hit["points"] for hit in hits]


def _render_block(stats: dict, timestamp: str) -> str:
    """Render one YAML-ish timestamped block."""
    lines = [f"# HN stats collected at {timestamp} (UTC)"]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _append_or_write(path: Path, block: str, mode: str) -> None:
    """Overwrite the file or append the block (creating it if absent)."""
    if mode == "overwrite" or not path.exists():
        path.write_text(block)
        return
    existing = path.read_text()
    path.write_text(
        existing + ("\n" if existing and not existing.endswith("\n") else "") + block
    )


def main(cfg: DictConfig) -> None:
    monitoring_handler: MonitoringHandler = hydra.utils.instantiate(
        cfg.monitoring_handler
    )
    client = HNAlgoliaClient()
    scores = _collect_scores(client)
    stats = _compute_stats(
        scores, min_sample=cfg.min_sample, percentiles=list(cfg.percentiles)
    )
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    block = _render_block(stats, timestamp)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _append_or_write(output_path, block, cfg.mode)
    logger.info("HN calibration: %s", stats)
    monitoring_handler("1", f"HN calibration done: {stats}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
