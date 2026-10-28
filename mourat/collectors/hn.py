"""HN post collector: Function adapter over comourat."""

from __future__ import annotations

from typing import Any

from comourat.clients.hn_algolia import HNAlgoliaClient
from comourat.collectors.hn import HNPostCollectorCore
from comourat.models import HNPostCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler


class HNPostCollector(Function[Any, HNPostCollection]):
    """Collects HN stories from the Algolia API within a time window."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        hn_client: HNAlgoliaClient,
        time_window_lo_hours: int = 48,
        time_window_hi_hours: int = 0,
        min_points: int = 0,
        tags: list[str] | None = None,
        limit: int = 1000,
        text_for_monitoring_template: str = "Collected {count} HN posts",
    ) -> None:
        self.core = HNPostCollectorCore(
            hn_client=hn_client,
            time_window_lo_hours=time_window_lo_hours,
            time_window_hi_hours=time_window_hi_hours,
            min_points=min_points,
            tags=tags,
            limit=limit,
        )
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[HNPostCollection, str]:
        output = self.core.collect()
        text_for_monitoring = self.text_for_monitoring_template.format(
            count=len(output.posts)
        )
        return output, text_for_monitoring
