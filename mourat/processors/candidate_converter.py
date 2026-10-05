"""Candidate -> resolved record conversion (spec 14, option B).

The tail stages (PaperContentItemScorer, writers) are typed on the
ResolvedPaper lineage, which the newest-papers path builds without running
PaperResolver. This small deterministic stage maps each assessed candidate
onto a ResolvedPaper:

- description -> abstract (the candidate carries the RSS abstract there)
- first urls_seen element -> url (the arXiv abs link)
- publication_date -> publication_date (already ISO on the candidate)
- influence_score -> influence_score (option (b): one field, measure named
  by the writer's influence_metric_id)
- arxiv_id -> arxiv_id (in-flight; never persisted)
- resolution_status stays at its default: a lineage artefact, meaningless
  on this path (no resolver ran).
"""

from mourat.base import Function
from mourat.data_models import (
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
)
from mourat.monitoring import MonitoringHandler


class CandidateToResolvedConverter(
    Function[PaperCandidateCollection, ResolvedPaperCollection]
):
    def __init__(self, monitoring_handler: MonitoringHandler) -> None:
        super().__init__(monitoring_handler)

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[ResolvedPaperCollection, str]:
        resolved = [
            ResolvedPaper(
                title=p.title,
                abstract=p.description,
                authors=list(p.authors),
                publication_date=p.publication_date,
                url=p.urls_seen[0] if p.urls_seen else "",
                provenance=list(p.provenance),
                arxiv_id=p.arxiv_id,
                influence_score=p.influence_score,
            )
            for p in data.papers
        ]
        lines = [f"Converted {len(resolved)} candidates to resolved records"]
        n_affil = sum(1 for p in data.papers if p.affiliations)
        lines.append(f"Candidates with affiliations: {n_affil}")
        return ResolvedPaperCollection(papers=resolved), "\n".join(lines)
