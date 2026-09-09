"""Clients for external metadata APIs.

These clients are plain (non-`Function`) helpers: the `Function` invariant
governs pipeline stages, and these are called from inside components.
"""

from mourat.clients.arxiv import ArxivClient
from mourat.clients.openalex import OpenAlexClient

__all__ = ["ArxivClient", "OpenAlexClient"]
