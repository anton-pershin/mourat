"""HN post collector."""

from __future__ import annotations

import datetime
from typing import Any

from mourat.base import Function
from mourat.clients.hn_algolia import HNAlgoliaClient
from mourat.data_models import HNPostCollection, HNPostInfo
from mourat.monitoring import MonitoringHandler


class HNPostCollector(Function[Any, HNPostCollection]):
    """Collects HN stories from the Algolia API within a time window."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        hn_client: HNAlgoliaClient,
        time_window_lo_hours: int = 48,
        time_window_hi_hours: int = 0,
        min_points: int = 20,
        tags: list[str] | None = None,
        limit: int = 1000,
        text_for_monitoring_template: str = "Collected {count} HN posts",
    ) -> None:
        self.hn_client = hn_client
        self.time_window_lo_hours = time_window_lo_hours
        self.time_window_hi_hours = time_window_hi_hours
        self.min_points = min_points
        self.tags = tags if tags is not None else ["story"]
        self.limit = limit
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data) -> tuple[HNPostCollection, str]:
        now = int(datetime.datetime.now(datetime.UTC).timestamp())
        lo = now - self.time_window_lo_hours * 3600
        hi = now - self.time_window_hi_hours * 3600

        posts: list[HNPostInfo] = []
        page = 0
        while True:
            payload = self.hn_client.search_page(
                lo,
                hi,
                min_points=self.min_points,
                tags=self.tags,
                page=page,
            )
            for hit in payload["hits"]:
                parsed = self.hn_client.parse_hit(hit)
                posts.append(HNPostInfo(**parsed))
                if len(posts) >= self.limit:
                    break
            nb_pages = payload.get("nbPages") or 0
            if len(posts) >= self.limit or page + 1 >= nb_pages:
                break
            page += 1

        output = HNPostCollection(posts=posts)
        text_for_monitoring = self.text_for_monitoring_template.format(count=len(posts))
        return output, text_for_monitoring
