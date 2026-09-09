"""OpenAlex client: title search, citation-graph queries and single-work lookup.

Plain (non-`Function`) client called from within pipeline components. Owns the
request policy per NFR2/NFR4: identifying User-Agent, `select=` field lists,
the `cursor=*` sentinel on the first request of any paged call (without it
OpenAlex silently returns page 1 with no error), and bounded retry with backoff
on 429/5xx.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

OPENALEX_API_BASE_URL = "https://api.openalex.org/works"

# Fields requested on every work lookup/search: everything resolution and
# influence assessment need, nothing more (NFR2: request only needed fields).
DEFAULT_SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "display_name",
    "abstract_inverted_index",
    "authorships",
    "publication_date",
    "publication_year",
    "cited_by_count",
    "fwci",
    "primary_location",
    "locations",
    "best_oa_location",
]


class OpenAlexClient:
    """Client for the unauthenticated OpenAlex works API.

    Args mirror the future Hydra group file keys one-to-one. `http_client` is
    injectable for tests; by default a module-level `requests.Session` with the
    configured identifying User-Agent is used. `api_key` (optional) is sent as
    the `api_key` query parameter on every request for the authenticated,
    higher-rate-limit tier; `None` keeps the unauthenticated public tier.
    """

    def __init__(
        self,
        user_agent: str,
        select_fields: list[str] | None = None,
        max_retries: int = 4,
        regular_delay_seconds: float = 2.0,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        per_page: int = 25,
        api_key: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.select_fields = select_fields or list(DEFAULT_SELECT_FIELDS)
        self.max_retries = max_retries
        self.regular_delay_seconds = regular_delay_seconds
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.per_page = per_page
        self.api_key = api_key
        self._session: requests.Session | None = None
        if http_client is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent})
        else:
            self._http_client = http_client

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        """GET with bounded retry and backoff on 429/5xx (NFR2).

        Transport errors (`Timeout`, `ConnectionError`) are as transient as
        a 5xx and retried the same way, bounded by `max_retries`.
        """
        attempt = 0
        while True:
            if self.api_key:
                params = {**params, "api_key": self.api_key}
            try:
                if self._session is not None:
                    response = self._session.get(
                        url, params=params, timeout=self.timeout_seconds
                    )
                else:
                    response = self._http_client.get(
                        url, params=params, timeout=self.timeout_seconds
                    )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAlex request failed (%s); retry %d/%d in %.1fs: %s",
                    type(exc).__name__,
                    attempt,
                    self.max_retries,
                    delay,
                    url,
                )
                time.sleep(delay)
                continue
            if response.status_code == 200:
                time.sleep(self.regular_delay_seconds)
                return response
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= self.max_retries:
                response.raise_for_status()
                return response
            attempt += 1
            delay = self.backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "OpenAlex request failed with %d; retry %d/%d in %.1fs: %s",
                response.status_code,
                attempt,
                self.max_retries,
                delay,
                url,
            )
            time.sleep(delay)

    def search_works_by_title(
        self, title: str, cursor: str | None = "*"
    ) -> dict[str, Any]:
        """Search works by title text; returns the raw parsed JSON envelope.

        The first call uses the `cursor=*` sentinel (NFR3); callers advance
        paging by passing `meta.next_cursor` back as `cursor`.
        """
        params: dict[str, Any] = {
            "search": title,
            "select": ",".join(self.select_fields),
            "per-page": self.per_page,
            "cursor": cursor,
        }
        response = self._get(OPENALEX_API_BASE_URL, params)
        return response.json()

    def search_works_citing(
        self, work_id: str, cursor: str | None = "*"
    ) -> dict[str, Any]:
        """Return works that cite `work_id` (forward citation-graph expansion).

        Cursor-paged like the title search: the first call uses the `cursor=*`
        sentinel, callers advance via `meta.next_cursor`. Relevance-ranked by
        the API default; no citation-count sort is ever sent (spec 11 FR3).
        """
        bare_id = work_id.rsplit("/", 1)[-1]
        params: dict[str, Any] = {
            "filter": f"cites:{bare_id}",
            "select": ",".join(self.select_fields),
            "per-page": self.per_page,
            "cursor": cursor,
        }
        response = self._get(OPENALEX_API_BASE_URL, params)
        return response.json()

    def get_work_references(self, work_id: str) -> list[str]:
        """Return the OpenAlex ids of works `work_id` cites (backward expansion).

        Reads the record's own `referenced_works` field. An empty or missing
        list is returned as `[]` (normal for preprint-only records), never an
        error.
        """
        record = self.get_work_by_id(work_id)
        references = record.get("referenced_works") or []
        return list(references)

    def get_work_by_id(self, work_id: str) -> dict[str, Any]:
        """Fetch a single work by its OpenAlex id (e.g. `W...`) or URL form.

        OpenAlex records carry ids as `https://openalex.org/W...` — the
        website, not the API. Requesting that host gets a 403, so the bare
        id is always extracted and requested from the API base URL.
        """
        bare_id = work_id.rsplit("/", 1)[-1]
        params: dict[str, Any] = {"select": ",".join(self.select_fields)}
        response = self._get(f"{OPENALEX_API_BASE_URL}/{bare_id}", params)
        return response.json()
