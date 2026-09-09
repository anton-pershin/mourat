import datetime
from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt


class PaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="Arxiv URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    citation_count: NonNegativeInt | None = Field(
        description="Number of citations", default=None
    )
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(
        description="Date of publication", default=None
    )


class ScoredPaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="Arxiv URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    citation_count: NonNegativeInt | None = Field(
        description="Number of citations", default=None
    )
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(
        description="Date of publication", default=None
    )
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


class PaperInfoCollection(BaseModel):
    papers: list[PaperInfo]


class ScoredPaperInfoCollection(BaseModel):
    papers: list[ScoredPaperInfo]


class AssignedPaperInfo(BaseModel):
    title: str = Field(description="Title of the paper")
    link: str = Field(description="URL of the paper")
    abstract: str = Field(description="Abstract of the paper")
    citation_count: NonNegativeInt | None = Field(
        description="Number of citations", default=None
    )
    authors: list[str] | None = Field(description="Authors of the paper", default=None)
    publication_date: datetime.date | None = Field(
        description="Date of publication", default=None
    )
    assigned_topics: list[str]


class AssignedPaperInfoCollection(BaseModel):
    papers: list[AssignedPaperInfo]


class ListOfTopics(BaseModel):
    topics: list[str] = Field(description="List of topics")


class PaperScoredByAgent(BaseModel):
    title: str = Field(description="Title of the paper")
    score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)
    justification: str = Field(description="Justification for the score")


class QueryInfo(BaseModel):
    general_queries: list[str]
    specific_queries: list[str]


class BusinessProductInfo(BaseModel):
    product_description: str = Field(description="Product description")
    challenges: list[str] = Field(
        description="A list of techinical challenges faced by the product"
    )


class CandidateTopicInfo(BaseModel):
    candidate_topic_name: str = Field(description="Candidate topic name")
    candidate_topic_description: str = Field(
        description="Detailed description of the candidate topic"
    )
    business_product: dict[str, BusinessProductInfo] = Field(
        description=(
            "A dictionary of business product where the key is the business product "
            "and the value is its description"
        )
    )


class CandidateTopicRelevanceInfo(BaseModel):
    product_name: str = Field(description="Product name")
    relevant_challenges: list[str] = Field(
        description="A list of relevant business product challenges"
    )
    relevance_justification: str = Field(
        description=(
            "Justification of why the candidate topic may be relevant "
            "to the listed challenges"
        )
    )
    relevance_score: int = Field(description="Relevance score from 0 to 5", ge=0, le=5)


class CandidateTopicAssessment(BaseModel):
    assessment: list[CandidateTopicRelevanceInfo]


# -- Database entity models --


class BusinessDomain(BaseModel):
    id: str
    name: str
    description: str = ""


class Product(BaseModel):
    id: str
    name: str
    domain_id: str
    description: str = ""


class HighLevelTechnology(BaseModel):
    id: str
    name: str
    product_id: str
    description: str = ""


class TechnicalChallenge(BaseModel):
    id: str
    name: str
    description: str = ""


class Constraint(BaseModel):
    id: str
    name: str
    description: str = ""


class ResearchDomain(BaseModel):
    id: str
    name: str
    description: str = ""


class ResearchDirection(BaseModel):
    id: str
    name: str
    domain_id: str
    description: str = ""


class ResearchObject(BaseModel):
    id: str
    name: str
    direction_id: str
    description: str = ""


class ResearchQuestion(BaseModel):
    id: str
    name: str
    object_id: str
    description: str = ""


class ResearchTopic(BaseModel):
    id: str
    name: str
    description: str = ""


class ContentItem(BaseModel):
    id: str
    name: str
    source_type_id: str
    platform_id: str
    influence_metric_id: str
    description: str = ""
    url: str | None = None
    published_at: str | None = None
    authors: str | None = None
    influence_score: int | None = None


class SourceType(BaseModel):
    id: str
    name: str
    description: str = ""


class Platform(BaseModel):
    id: str
    name: str
    description: str = ""


class InfluenceMetric(BaseModel):
    id: str
    name: str
    description: str = ""


# -- Post collection models --


class RedditPostInfo(BaseModel):
    """A single Reddit post for pipeline processing."""

    subreddit: str
    submission_id: str
    title: str
    author: str
    date: str  # ISO format datetime string
    url: str
    text: str = ""
    score: int = 0


class RedditPostCollection(BaseModel):
    """Collection of Reddit posts."""

    posts: list[RedditPostInfo]


# -- LLM agent output models (used by enrichment and scoring agents) --


class EnrichmentResult(BaseModel):
    """LLM enrichment agent output: additional context points."""

    additional_context: list[str] = Field(
        description=(
            "Short self-contained context points with facts obtained from the web "
            "that are not already evident in the post"
        )
    )


class ScoreEntry(BaseModel):
    """A single relevance score entry."""

    id: str = Field(
        description=(
            "ID of the research question, technical challenge, constraint, or topic"
        )
    )
    type: Literal["rq", "tc", "topic", "constraint"] = Field(
        description="Type: 'rq', 'tc', 'topic', or 'constraint'"
    )
    score: int = Field(description="Relevance score from 0 to 100", ge=0, le=100)
    justification: str = Field(description="Justification for the score")


class ScoringResult(BaseModel):
    """LLM scoring agent output."""

    scores: list[ScoreEntry] = Field(
        description="List of relevance scores for the post"
    )


# -- Post enrichment/scoring pipeline models --


class EnrichedRedditPost(BaseModel):
    """A Reddit post with additional context points gathered from the web."""

    post: RedditPostInfo
    additional_context: list[str] = []


class EnrichedRedditPostCollection(BaseModel):
    """Collection of enriched Reddit posts."""

    posts: list[EnrichedRedditPost]


class ScoredRedditPost(BaseModel):
    """A Reddit post with relevance scores."""

    post: RedditPostInfo
    additional_context: list[str] = []
    relevance_scores: list[ScoreEntry] = []
    filtering_score: float = 0.0


class ScoredRedditPostCollection(BaseModel):
    """Collection of scored Reddit posts."""

    posts: list[ScoredRedditPost]


# -- Paper collection models --


class PaperCandidate(BaseModel):
    """A candidate paper discovered by the LLM agent, as a bibliographic description.

    Carries no identifier fields by design: any DOI or arXiv id the agent
    produces is discarded, not stored (spec decision 4.3).
    """

    title: str = Field(description="Title of the paper as discovered")
    authors: list[str] = Field(description="Authors of the paper, as found")
    description: str = Field(
        description="Short description of the paper drawn from the pages read"
    )
    urls_seen: list[str] = Field(
        description="URLs encountered while discovering the paper, as-encountered"
    )
    provenance: list[str] = Field(
        default_factory=lambda: ["web_search"],
        description=(
            "Generator or agent names that produced this candidate. The "
            "discovery agent leaves the default single web-search entry; "
            "seed expansion records every generator that produced the "
            "candidate (spec 11 §5.3)."
        ),
    )


class PaperCandidateCollection(BaseModel):
    """Collection of discovered paper candidates."""

    papers: list[PaperCandidate]


class ResolvedPaper(BaseModel):
    """A candidate paper resolved to canonical metadata, per spec section 5.3.

    Content-item attributes (what the writers persist) come first; everything
    else is in-flight only — identifiers, influence fields, resolution status
    and the url-absence reason never reach the database.
    """

    # -- content item attributes --
    title: str
    abstract: str = ""
    authors: list[str] = []
    publication_date: str | None = None
    url: str = ""
    # provenance is a content-item attribute: it persists with the paper (FR5)
    provenance: list[str] = Field(default_factory=lambda: ["web_search"])
    # -- in-flight only --
    doi: str | None = None
    arxiv_id: str | None = None
    work_id: str | None = None
    influence_fwci: float | None = None
    influence_cited_by_count: int | None = None
    influence_score: int | None = None
    influence_measure_used: str | None = None
    resolution_status: str = "unresolved"
    url_absent_reason: str | None = None


class ResolvedPaperCollection(BaseModel):
    """Collection of resolved papers."""

    papers: list[ResolvedPaper]


class ScoredPaper(BaseModel):
    """A resolved paper with relevance scores."""

    paper: ResolvedPaper
    relevance_scores: list[ScoreEntry] = []
    filtering_score: float = 0.0


class ScoredPaperCollection(BaseModel):
    """Collection of scored papers."""

    papers: list[ScoredPaper]


class ContentItemCollection(BaseModel):
    """Collection wrapper for stored content items (any source type).

    Spec 11 needs this so `SeedResolver` can take a proper Pydantic
    collection as its input type.
    """

    items: list[ContentItem]


class Seed(BaseModel):
    """A stored content item identified as an expansion seed (spec 11 §5.3).

    `work_id` is the metadata-API identifier assigned by `SeedResolver`
    (in-flight only); `influence_value` is the seed's normalised influence
    used for the floor derivation.
    """

    content_item_id: str
    work_id: str | None = None
    title: str
    influence_value: float | None = None


class SeedCollection(BaseModel):
    """Collection of seeds resolved from stored content items."""

    seeds: list[Seed]


class ContentItemScoringInput(BaseModel):
    """Neutral item shape the generalised scorer consumes."""

    id: str
    title: str
    body_text: str = ""
    context_points: list[str] = []


class ContentItemScoringInputCollection(BaseModel):
    """Collection wrapper for the neutral scoring input."""

    items: list[ContentItemScoringInput]


# -- Slop classification pipeline models --


class SlopVerdict(BaseModel):
    """A binary slop verdict for a single post."""

    submission_id: str = Field(description="Reddit submission id the verdict refers to")
    is_slop: bool = Field(
        description="True if the post bears no meaningful content and should not be enriched"
    )
    justification: str = Field(description="Brief justification for the verdict")


class SlopClassificationResult(BaseModel):
    """LLM slop classification agent output."""

    verdicts: list[SlopVerdict] = Field(description="One verdict per post in the batch")


class ClassifiedRedditPost(BaseModel):
    """A Reddit post with a slop verdict."""

    post: RedditPostInfo
    is_slop: bool = False
    justification: str = ""


class ClassifiedRedditPostCollection(BaseModel):
    """Collection of Reddit posts with slop verdicts."""

    posts: list[ClassifiedRedditPost]
