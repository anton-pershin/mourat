"""Canonical paper-identity filtering for combined newest-paper producers."""

from __future__ import annotations

from comourat.models import PaperCandidate, normalize_arxiv_id, normalize_doi

from mourat.base import Function
from mourat.data_models import PaperCandidateCollection
from mourat.monitoring import MonitoringHandler


def _doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    return normalized.lower() if normalized else None


def _is_arxiv_primary(paper: PaperCandidate) -> bool:
    return "arxiv_rss" in paper.provenance


def _identity_keys(paper: PaperCandidate) -> set[tuple[str, str]]:
    keys = set()
    if paper.arxiv_id:
        keys.add(("arxiv", paper.arxiv_id))
    if paper.doi:
        keys.add(("doi", paper.doi))
    return keys


def _merge(primary: PaperCandidate, duplicate: PaperCandidate) -> None:
    """Retain primary metadata and add only missing lineage information."""
    primary.arxiv_id = primary.arxiv_id or duplicate.arxiv_id
    primary.doi = primary.doi or duplicate.doi
    primary.urls_seen.extend(
        url for url in duplicate.urls_seen if url not in primary.urls_seen
    )
    primary.provenance.extend(
        source for source in duplicate.provenance if source not in primary.provenance
    )


class PaperCandidateDeduplicator(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Exclude DOI-less S2-only papers and merge canonical identity matches."""

    def __init__(self, monitoring_handler: MonitoringHandler) -> None:
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        eligible = []
        without_doi = 0
        # Stable sorting gives every arXiv RSS record precedence without
        # requiring the orchestrator to depend on producer call order.
        ordered = sorted(
            data.papers,
            key=lambda paper: 0 if _is_arxiv_primary(paper) else 1,
        )
        for paper in ordered:
            paper.arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            paper.doi = _doi(paper.doi)
            if (
                "semantic_scholar" in paper.provenance
                and not _is_arxiv_primary(paper)
                and not paper.doi
            ):
                without_doi += 1
                continue
            eligible.append(paper)

        # Union identity links so a record carrying two identifiers also joins
        # groups that would otherwise be separate.
        parent = list(range(len(eligible)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        key_owner: dict[tuple[str, str], int] = {}
        for index, paper in enumerate(eligible):
            for key in _identity_keys(paper):
                if key in key_owner:
                    union(index, key_owner[key])
                else:
                    key_owner[key] = index

        groups: dict[int, list[int]] = {}
        for index in range(len(eligible)):
            groups.setdefault(find(index), []).append(index)

        output = []
        duplicate_count = 0
        for indices in sorted(groups.values(), key=lambda indices: indices[0]):
            primary = eligible[indices[0]]
            for index in indices[1:]:
                _merge(primary, eligible[index])
                duplicate_count += 1
            output.append(primary)

        monitoring = (
            f"Papers in: {len(data.papers)}, kept: {len(output)}, "
            f"duplicates merged: {duplicate_count}, "
            f"Semantic Scholar-only without DOI: {without_doi}"
        )
        return PaperCandidateCollection(papers=output), monitoring
