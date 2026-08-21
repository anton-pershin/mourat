import datetime

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
