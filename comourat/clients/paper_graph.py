"""Common identity and normalized records for paper graph clients."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field

_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(
    r"^(?:arxiv:)?([0-9]{4}\.\d{4,5})(?:v[0-9]+)?$", re.IGNORECASE
)


class PaperIdentity(BaseModel):
    """Canonical paper identity; provider IDs are deliberately absent."""

    arxiv_id: str | None = None
    doi: str | None = None

    @classmethod
    def from_values(
        cls, arxiv_id: str | None = None, doi: str | None = None
    ) -> "PaperIdentity":
        normalized_arxiv = normalize_arxiv_id(arxiv_id)
        normalized_doi = normalize_doi(doi)
        if normalized_doi is not None:
            match = _ARXIV_DOI_RE.match(normalized_doi)
            if match:
                normalized_arxiv = normalized_arxiv or normalize_arxiv_id(
                    match.group(1)
                )
                normalized_doi = None
        return cls(arxiv_id=normalized_arxiv, doi=normalized_doi)


def normalize_arxiv_id(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    match = _ARXIV_ID_RE.match(value)
    return match.group(1) if match else value


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = re.sub(r"^https?://doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:", "", value, flags=re.IGNORECASE)
    return value or None


class PaperRecord(BaseModel):
    """Provider-neutral paper record returned by a graph client."""

    identity: PaperIdentity = Field(default_factory=PaperIdentity)
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    publication_date: date | None = None
    citation_count: int | None = None
    raw_influence: dict[str, float | int] = Field(default_factory=dict)


class PaperPage(BaseModel):
    """A page of records; continuation is opaque to the caller."""

    papers: list[PaperRecord] = Field(default_factory=list)
    continuation: Any = None


class PaperGraphClient(Protocol):
    """Operations required by seed resolution and expansion."""

    def resolve_by_arxiv_id(self, arxiv_id: str) -> PaperRecord | None: ...

    def resolve_by_doi(self, doi: str) -> PaperRecord | None: ...

    def search_papers(self, query: str, continuation: Any = None) -> PaperPage: ...

    def get_citations(
        self, identity: PaperIdentity, continuation: Any = None
    ) -> PaperPage: ...

    def get_references(
        self, identity: PaperIdentity, limit: int | None = None
    ) -> list[PaperRecord]: ...
