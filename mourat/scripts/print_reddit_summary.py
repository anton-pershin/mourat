import datetime

import praw
import hydra
from omegaconf import DictConfig
from pydantic import BaseModel
from pydantic_ai import Agent
from rich.progress import track
from rich.markdown import Markdown

from mourat.utils.common import get_config_path
from mourat.utils.console import console


CONFIG_NAME = "config_print_reddit_summary"

class RedditPost(BaseModel):
    subreddit: str
    submission_id: str
    title: str
    author: str
    date: datetime.datetime
    url: str
    text: str
    score: int


def summarize_recent_posts(reddit_client: praw.Reddit, agent: Agent, user_prompt_template: str, subreddits: list[str], cutoff_time: datetime.datetime) -> None:
    # Collect posts from each subreddit
    all_posts = {}
    for subreddit_name in track(subreddits, description="Post collection"):
        all_posts[subreddit_name] = []
        subreddit = reddit_client.subreddit(subreddit_name)
        # Get new posts
        for post in subreddit.new(limit=100):
            post_time = datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.UTC)
            if post_time < cutoff_time:
                break

            all_posts[subreddit_name].append(
                RedditPost(
                    subreddit=subreddit_name,
                    submission_id=post.id,
                    title=post.title,
                    author=str(post.author),
                    date=post.created_utc,
                    url=post.url,
                    text=post.selftext,
                    score=post.score,
                )
            )

        all_posts[subreddit_name].sort(key=lambda x: x.score, reverse=True)
        total_posts = len(all_posts[subreddit_name])
        # Only 10 most popular are left
        all_posts[subreddit_name] = all_posts[subreddit_name][:min(total_posts, 10)]

    # Summarize posts
    result = agent.run_sync(
        user_prompt_template.format(
            posts="\n".join([p.model_dump_json(indent=2, ensure_ascii=True) for _, posts in all_posts.items() for p in posts])
        )
    )

    markdown = Markdown(result.output)
    console.print(markdown)
    console.print()



def print_reddit_summary(cfg: DictConfig) -> None:
    # Initialize Reddit API client
    reddit = praw.Reddit(
        client_id=cfg.user_settings.reddit.client_id,
        client_secret=cfg.user_settings.reddit.client_secret,
        user_agent=cfg.user_settings.reddit.user_agent,
    )

    # Calculate the cutoff time
    time_window = datetime.timedelta(**cfg.reddit.time_window)
    cutoff_time = datetime.datetime.now(datetime.UTC) - time_window

    # Create an agent
    model = hydra.utils.instantiate(cfg.slow_llm)
    agent = Agent(model, system_prompt=cfg.system_prompt)
            
    # Summarize non-technical subreddits
    console.rule("[bold]Non-technical subreddits")
    summarize_recent_posts(
        reddit_client=reddit,
        agent=agent,
        user_prompt_template=cfg.user_prompt_template,
        subreddits=cfg.reddit.nontechnical_subreddits,
        cutoff_time=cutoff_time
    )

    # Summarize technical subreddits
    console.rule("[bold]Technical subreddits")
    summarize_recent_posts(
        reddit_client=reddit,
        agent=agent,
        user_prompt_template=cfg.user_prompt_template,
        subreddits=cfg.reddit.technical_subreddits,
        cutoff_time=cutoff_time
    )


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(print_reddit_summary)()

