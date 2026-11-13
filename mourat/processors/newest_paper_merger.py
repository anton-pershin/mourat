"""Merge newest-paper candidates while preserving canonical source precedence."""

from __future__ import annotations

from collections.abc import Collection

from comourat.models import PaperCandidate as ComouratPaperCandidate
from comourat.models import PaperIdentity
from pydantic import BaseModel, Field

from mourat.data_models import PaperCandidate, PaperCandidateCollection


class NewestPaperMergeResult(BaseModel):
    """Merged candidates and the counts needed for observability and the ledger."""

    collection: PaperCandidateCollection
    arxiv_collected: int = 0
    semantic_scholar_collected: int = 0
    ledger_excluded: int = 0
    missing_provider_id_excluded: int = 0
    missing_doi_excluded: int = 0
    duplicates_excluded: int = 0
    merged_cross_source: int = 0
    new_semantic_scholar_ids: list[str] = Field(default_factory=list)


def _normalized_identity(candidate: ComouratPaperCandidate) -> PaperIdentity:
    identity = PaperIdentity.from_values(
        arxiv_id=candidate.arxiv_id,
        doi=candidate.doi,
    )
    if identity.doi is not None:
        identity.doi = identity.doi.lower()
    return identity


def _aliases(candidate: ComouratPaperCandidate) -> list[tuple[str, str]]:
    identity = _normalized_identity(candidate)
    aliases = []
    if identity.arxiv_id:
        aliases.append(("arxiv", identity.arxiv_id))
    if identity.doi:
        aliases.append(("doi", identity.doi))
    return aliases


def _normalize_candidate(
    candidate: ComouratPaperCandidate,
    source: str,
) -> PaperCandidate:
    normalized = PaperCandidate.model_validate(candidate.model_dump())
    identity = _normalized_identity(normalized)
    normalized.arxiv_id = identity.arxiv_id
    normalized.doi = identity.doi
    normalized.provenance = [source]
    return normalized


class _CandidateIndex:
    """Mutable identity index used only during one merge operation."""

    def __init__(self) -> None:
        self.papers: list[ComouratPaperCandidate] = []
        self._alias_to_index: dict[tuple[str, str], int] = {}

    def find(self, candidate: ComouratPaperCandidate) -> int | None:
        for alias in _aliases(candidate):
            index = self._alias_to_index.get(alias)
            if index is not None:
                return index
        return None

    def add(self, candidate: ComouratPaperCandidate) -> None:
        self.papers.append(candidate)
        self._register(len(self.papers) - 1, candidate)

    def merge(self, index: int, candidate: ComouratPaperCandidate) -> bool:
        """Merge into the primary record and return True for an arXiv-S2 merge."""
        primary = self.papers[index]
        if primary.arxiv_id is None:
            primary.arxiv_id = candidate.arxiv_id
        if primary.doi is None:
            primary.doi = candidate.doi

        cross_source = "arxiv" in primary.provenance and "arxiv" not in (
            candidate.provenance
        )
        if cross_source and "semantic_scholar" not in primary.provenance:
            primary.provenance.append("semantic_scholar")
        self._register(index, primary)
        self._register(index, candidate)
        return cross_source

    def _register(self, index: int, candidate: ComouratPaperCandidate) -> None:
        for alias in _aliases(candidate):
            self._alias_to_index.setdefault(alias, index)


def merge_newest_paper_candidates(
    arxiv: PaperCandidateCollection,
    semantic_scholar: PaperCandidateCollection,
    seen_provider_ids: Collection[str] = (),
) -> NewestPaperMergeResult:
    """Deduplicate candidates by arXiv ID or DOI with arXiv precedence."""
    index = _CandidateIndex()
    seen = set(seen_provider_ids)
    new_provider_ids: list[str] = []
    new_provider_id_set: set[str] = set()
    result = NewestPaperMergeResult(
        collection=PaperCandidateCollection(papers=[]),
        arxiv_collected=len(arxiv.papers),
        semantic_scholar_collected=len(semantic_scholar.papers),
    )

    for raw_candidate in arxiv.papers:
        candidate = _normalize_candidate(raw_candidate, "arxiv")
        matching_index = index.find(candidate)
        if matching_index is None:
            index.add(candidate)
        else:
            index.merge(matching_index, candidate)
            result.duplicates_excluded += 1

    for raw_candidate in semantic_scholar.papers:
        provider_id = getattr(raw_candidate, "semantic_scholar_id", None)
        if not provider_id:
            result.missing_provider_id_excluded += 1
            continue
        if provider_id in seen:
            result.ledger_excluded += 1
            continue
        if provider_id not in new_provider_id_set:
            new_provider_ids.append(provider_id)
            new_provider_id_set.add(provider_id)

        candidate = _normalize_candidate(raw_candidate, "semantic_scholar")
        matching_index = index.find(candidate)
        if matching_index is not None:
            if index.merge(matching_index, candidate):
                result.merged_cross_source += 1
            else:
                result.duplicates_excluded += 1
        elif candidate.doi is None:
            result.missing_doi_excluded += 1
        else:
            index.add(candidate)

    result.collection = PaperCandidateCollection(papers=index.papers)
    result.new_semantic_scholar_ids = new_provider_ids
    return result
