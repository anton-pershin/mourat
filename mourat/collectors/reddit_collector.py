"""Reddit post collector: Function adapter over comourat."""

from __future__ import annotations

from typing import Any

import praw

from comourat.collectors.reddit import RedditPostCollectorCore
from comourat.models import RedditPostCollection
from mourat.base import Function
from mourat.monitoring import MonitoringHandler


class RedditPostCollector(Function[Any, RedditPostCollection]):
    """Collects posts from Reddit subreddits within a time window."""

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        reddit_client: praw.Reddit,
        subreddits: list[str],
        time_window: dict,  # kwargs for timedelta (e.g., {"hours": 24})
        limit_per_subreddit: int = 100,
        max_posts_per_subreddit: int = 10,
        require_text: bool = True,
        text_for_monitoring_template: str = "Collected {count} posts from {subreddits}",
    ) -> None:
        self.core = RedditPostCollectorCore(
            reddit_client=reddit_client,
            subreddits=subreddits,
            time_window=time_window,
            limit_per_subreddit=limit_per_subreddit,
            max_posts_per_subreddit=max_posts_per_subreddit,
            require_text=require_text,
        )
        self.text_for_monitoring_template = text_for_monitoring_template
        self.subreddits = subreddits
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[RedditPostCollection, str]:
        output = self.core.collect()
        text_for_monitoring = self.text_for_monitoring_template.format(
            count=len(output.posts),
            subreddits=", ".join(self.subreddits),
        )
        return output, text_for_monitoring
