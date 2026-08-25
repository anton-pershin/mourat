"""Reddit post collector module."""

from __future__ import annotations

import datetime
from typing import Any

import praw

from mourat.base import Function
from mourat.data_models import RedditPostCollection, RedditPostInfo
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
        self.reddit_client = reddit_client
        self.subreddits = subreddits
        self.time_window = datetime.timedelta(**time_window)
        self.limit_per_subreddit = limit_per_subreddit
        self.max_posts_per_subreddit = max_posts_per_subreddit
        self.require_text = require_text
        self.text_for_monitoring_template = text_for_monitoring_template
        super().__init__(monitoring_handler)

    def _run(self, data: Any) -> tuple[RedditPostCollection, str]:
        cutoff = datetime.datetime.now(datetime.UTC) - self.time_window
        all_posts: list[RedditPostInfo] = []

        for subreddit_name in self.subreddits:
            subreddit = self.reddit_client.subreddit(subreddit_name)
            posts = []

            for post in subreddit.new(limit=self.limit_per_subreddit):
                post_time = datetime.datetime.fromtimestamp(
                    post.created_utc, tz=datetime.UTC
                )
                if post_time < cutoff:
                    break

                if self.require_text and not post.selftext.strip():
                    continue

                posts.append(
                    RedditPostInfo(
                        subreddit=subreddit_name,
                        submission_id=post.id,
                        title=post.title,
                        author=str(post.author),
                        date=post_time.isoformat(),
                        url=post.url,
                        text=post.selftext,
                        score=post.score,
                    )
                )

            posts.sort(key=lambda x: x.score, reverse=True)
            all_posts.extend(posts[: min(len(posts), self.max_posts_per_subreddit)])

        output = RedditPostCollection(posts=all_posts)

        text_for_monitoring = self.text_for_monitoring_template.format(
            count=len(all_posts),
            subreddits=", ".join(self.subreddits),
        )

        return output, text_for_monitoring
