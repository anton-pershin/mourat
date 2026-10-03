"""Clients for external metadata APIs.

These clients are plain (non-`Function`) helpers: the `Function` invariant
governs pipeline stages, and these are called from inside components.
"""

from mourat.clients.arxiv import ArxivClient
from mourat.clients.openalex import OpenAlexClient
from mourat.clients.paper_graph import (
    PaperGraphClient,
    PaperIdentity,
    PaperPage,
    PaperRecord,
    normalize_arxiv_id,
    normalize_doi,
)
from mourat.clients.semantic_scholar import SemanticScholarClient

__all__ = [
    "ArxivClient",
    "OpenAlexClient",
    "PaperGraphClient",
    "PaperIdentity",
    "PaperPage",
    "PaperRecord",
    "SemanticScholarClient",
    "normalize_arxiv_id",
    "normalize_doi",
]
