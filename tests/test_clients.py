"""Unit tests for the metadata API clients with mocked HTTP responses."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from mourat.clients.arxiv import ArxivClient
from mourat.clients.openalex import DEFAULT_SELECT_FIELDS, OpenAlexClient


def _make_client(http_client: MagicMock, **kwargs) -> OpenAlexClient:
    defaults: dict = {"user_agent": "mourat-test/0.1 (mailto:test@example.com)"}
    defaults.update(kwargs)
    return OpenAlexClient(**defaults, http_client=http_client)


def _make_arxiv_client(http_client: MagicMock, **kwargs) -> ArxivClient:
    defaults: dict = {"user_agent": "mourat-test/0.1 (mailto:test@example.com)"}
    defaults.update(kwargs)
    return ArxivClient(**defaults, http_client=http_client)


def _json_response(payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    if isinstance(payload, dict):
        response.text = json.dumps(payload)
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return response


class TestSearchWorksByTitle:
    def test_first_request_sends_cursor_sentinel(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_by_title("some title")
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["cursor"] == "*"

    def test_request_carries_select_fields_and_ua(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_by_title("some title")
        _, kwargs = http_client.get.call_args
        requested = set(kwargs["params"]["select"].split(","))
        assert requested == set(DEFAULT_SELECT_FIELDS)
        assert "fwci" in requested
        assert "cited_by_count" in requested

    def test_advancing_paging_passes_next_cursor(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_by_title("t", cursor="WyJzIiwiOiJAZXIiXQ==")
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["cursor"] == "WyJzIiwiOiJAZXIiXQ=="

    def test_results_envelope_returned_unchanged(self):
        payload = {"meta": {"next_cursor": "abc"}, "results": [{"id": "W1"}]}
        http_client = MagicMock()
        http_client.get.return_value = _json_response(payload)
        client = _make_client(http_client)
        assert client.search_works_by_title("t") == payload


class TestApiKey:
    def test_api_key_sent_on_every_request_when_configured(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client, api_key="secret-key")
        client.search_works_by_title("some title")
        client.search_works_citing("W123")
        client.get_work_by_id("W123")
        for call in http_client.get.call_args_list:
            assert call.kwargs["params"]["api_key"] == "secret-key"

    def test_api_key_omitted_when_none(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_by_title("some title")
        assert "api_key" not in http_client.get.call_args.kwargs["params"]

    def test_api_key_does_not_mutate_caller_params(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client, api_key="secret-key")
        params = {"search": "t", "cursor": "*"}
        client._get("https://api.openalex.org/works", params)
        assert "api_key" not in params


class TestSearchWorksCiting:
    def test_filter_cites_with_bare_id_and_cursor_sentinel(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_citing("https://openalex.org/W123")
        args, kwargs = http_client.get.call_args
        assert kwargs["params"]["filter"] == "cites:W123"
        assert kwargs["params"]["cursor"] == "*"

    def test_bare_id_accepted_directly(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_citing("W123")
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["filter"] == "cites:W123"

    def test_advancing_paging_passes_next_cursor(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_citing("W123", cursor="abc123")
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["cursor"] == "abc123"

    def test_request_carries_select_fields(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_citing("W123")
        _, kwargs = http_client.get.call_args
        requested = set(kwargs["params"]["select"].split(","))
        assert requested == set(DEFAULT_SELECT_FIELDS)

    def test_no_citation_count_sort_parameter(self):
        """FR3: search must stay relevance-ranked; no sort param ever sent."""
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"meta": {}, "results": []})
        client = _make_client(http_client)
        client.search_works_citing("W123")
        _, kwargs = http_client.get.call_args
        assert "sort" not in kwargs["params"]
        assert "cited_by_count" not in str(kwargs["params"].get("sort", ""))

    def test_results_envelope_returned_unchanged(self):
        payload = {"meta": {"next_cursor": "n1"}, "results": [{"id": "W9"}]}
        http_client = MagicMock()
        http_client.get.return_value = _json_response(payload)
        client = _make_client(http_client)
        assert client.search_works_citing("W123") == payload


class TestGetWorkReferences:
    def test_returns_referenced_works_list(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response(
            {"id": "https://openalex.org/W1", "referenced_works": ["W2", "W3"]}
        )
        client = _make_client(http_client)
        assert client.get_work_references("W1") == ["W2", "W3"]

    def test_missing_field_returns_empty_list_not_error(self):
        """FR2: preprint-only records have no reference list; normal, not error."""
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"id": "https://openalex.org/W1"})
        client = _make_client(http_client)
        assert client.get_work_references("W1") == []

    def test_null_field_returns_empty_list(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response(
            {"id": "https://openalex.org/W1", "referenced_works": None}
        )
        client = _make_client(http_client)
        assert client.get_work_references("W1") == []


class TestGetWorkById:
    def test_bare_id_hits_works_endpoint(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"id": "https://openalex.org/W1"})
        client = _make_client(http_client)
        client.get_work_by_id("W123")
        args, kwargs = http_client.get.call_args
        assert args[0].endswith("/works/W123")
        assert "select" in kwargs["params"]

    def test_full_url_used_as_given(self):
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"id": "x"})
        client = _make_client(http_client)
        client.get_work_by_id("https://api.openalex.org/works/W9")
        args, _ = http_client.get.call_args
        assert args[0] == "https://api.openalex.org/works/W9"

    def test_website_url_form_rewritten_to_api_host(self):
        """OpenAlex record ids are `https://openalex.org/W...` (the website).

        Requesting that host returns 403; the bare id must always be re-hosted
        on the public API (FR: every request goes to api.openalex.org).
        """
        http_client = MagicMock()
        http_client.get.return_value = _json_response({"id": "x"})
        client = _make_client(http_client)
        client.get_work_by_id("https://openalex.org/W9")
        args, _ = http_client.get.call_args
        assert args[0] == "https://api.openalex.org/works/W9"


class TestRetryPolicy:
    def _flaky_client(self, statuses: list[int]) -> tuple[MagicMock, OpenAlexClient]:
        http_client = MagicMock()
        responses = [_json_response({"error": "e"}, status) for status in statuses]
        http_client.get.side_effect = responses
        return http_client, _make_client(
            http_client, max_retries=2, backoff_seconds=0.0
        )

    def test_retries_on_429_then_succeeds(self):
        http_client, client = self._flaky_client([429, 200])
        result = client.search_works_by_title("t")
        assert result == {"error": "e"}
        assert http_client.get.call_count == 2

    def test_exhausted_retries_raise(self):
        http_client, client = self._flaky_client([500, 500, 500])
        with pytest.raises(requests.HTTPError):
            client.search_works_by_title("t")
        assert http_client.get.call_count == 3  # initial + 2 retries

    def test_4xx_other_than_429_not_retried(self):
        http_client, client = self._flaky_client([404, 200])
        with pytest.raises(requests.HTTPError):
            client.search_works_by_title("t")
        assert http_client.get.call_count == 1


class TestDefaultSession:
    def test_default_session_carries_user_agent_header(self):
        client = OpenAlexClient(user_agent="mourat-test/0.1 (mailto:t@e.com)")
        assert client._session is not None
        assert (
            client._session.headers["User-Agent"] == "mourat-test/0.1 (mailto:t@e.com)"
        )


# -- ArxivClient --


SAMPLE_ARXIV_ENTRY_XML = (
    '<feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry>"
    "<title>  A  Study\n   of\t Whitespace  </title>"
    "<id>http://arxiv.org/abs/2401.00001v1</id>"
    "</entry>"
    "</feed>"
)

EMPTY_ARXIV_FEED_XML = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _arxiv_response(text: str, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.iter_content.return_value = iter([b""])
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code}", response=response
        )
    return response


def _pdf_response(
    status_code: int = 206,
    content_type: str = "application/pdf",
    head: bytes = b"%PDF-",
) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    response.iter_content.return_value = iter([head])
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code}", response=response
        )
    return response


class TestGetTitleById:
    def test_title_lookup_by_id_collapses_whitespace(self):
        http_client = MagicMock()
        http_client.get.return_value = _arxiv_response(SAMPLE_ARXIV_ENTRY_XML)
        client = _make_arxiv_client(http_client)
        assert client.get_title_by_id("2401.00001") == "A Study of Whitespace"
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["id_list"] == "2401.00001"

    def test_empty_feed_returns_none_not_error(self):
        http_client = MagicMock()
        http_client.get.return_value = _arxiv_response(EMPTY_ARXIV_FEED_XML)
        client = _make_arxiv_client(http_client)
        assert client.get_title_by_id("9999.99999") is None

    def test_uses_atom_api_endpoint(self):
        http_client = MagicMock()
        http_client.get.return_value = _arxiv_response(EMPTY_ARXIV_FEED_XML)
        client = _make_arxiv_client(http_client)
        client.get_title_by_id("2401.00001")
        args, _ = http_client.get.call_args
        assert "export.arxiv.org/api/query" in args[0]


class TestArxivDefaultSession:
    def test_default_session_carries_user_agent_header(self):
        client = ArxivClient(user_agent="mourat-test/0.1 (mailto:t@e.com)")
        assert client._session is not None
        assert (
            client._session.headers["User-Agent"] == "mourat-test/0.1 (mailto:t@e.com)"
        )


class TestSearchByTitle:
    @staticmethod
    def _atom_response(entry_id: str | None, title: str | None) -> MagicMock:
        entries = ""
        if entry_id is not None:
            entries = f"<entry><id>{entry_id}</id><title>{title}</title></entry>"
        text = (
            '<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom">' + entries + "</feed>"
        )
        response = MagicMock()
        response.status_code = 200
        response.text = text
        return response

    def test_found_entry_returns_id_and_collapsed_title(self):
        http_client = MagicMock()
        http_client.get.return_value = self._atom_response(
            "http://arxiv.org/abs/2305.13245v3",
            "GQA:  Training  Multi-Query\n Transformer  Models",
        )
        client = _make_arxiv_client(http_client)
        found = client.search_by_title("GQA: Training Multi-Query Transformer Models")
        assert found == ("2305.13245", "GQA: Training Multi-Query Transformer Models")

    def test_empty_feed_returns_none(self):
        http_client = MagicMock()
        http_client.get.return_value = self._atom_response(None, None)
        client = _make_arxiv_client(http_client)
        assert client.search_by_title("No Such Paper") is None

    def test_query_uses_quoted_title_search(self):
        http_client = MagicMock()
        http_client.get.return_value = self._atom_response(None, None)
        client = _make_arxiv_client(http_client)
        client.search_by_title("Some Title Here")
        _, kwargs = http_client.get.call_args
        assert kwargs["params"]["search_query"] == 'ti:"Some Title Here"'
        assert kwargs["params"]["max_results"] == 1

    def test_version_suffix_and_url_prefix_stripped(self):
        http_client = MagicMock()
        http_client.get.return_value = self._atom_response(
            "http://arxiv.org/abs/2401.00001v2", "A Title"
        )
        client = _make_arxiv_client(http_client)
        assert client.search_by_title("A Title") == ("2401.00001", "A Title")


class TestArxivPolitenessDelay:
    def test_regular_delay_sleeps_after_success(self):
        import time as _time

        http_client = MagicMock()
        http_client.get.return_value = self._atom_ok()
        client = _make_arxiv_client(http_client, regular_delay_seconds=1.5)
        t0 = _time.monotonic()
        client.get_title_by_id("1706.03762")
        elapsed = _time.monotonic() - t0
        assert elapsed >= 1.4  # allow tiny scheduling slack

    @staticmethod
    def _atom_ok() -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.text = (
            '<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><id>http://arxiv.org/abs/1706.03762</id>"
            "<title>Attention Is All You Need</title></entry></feed>"
        )
        return response


class TestTransportRetry:
    """A Timeout/ConnectionError is retried like 429/5xx, bounded by max_retries."""

    def test_openalex_readtimeout_retried_then_succeeds(self):
        import requests as _requests

        http_client = MagicMock()
        good = _json_response({"meta": {}, "results": []})
        http_client.get.side_effect = [
            _requests.ConnectionError("read timed out"),
            _requests.Timeout("boom"),
            good,
        ]
        client = _make_client(http_client, max_retries=4, backoff_seconds=0)
        assert client.search_works_by_title("t") == {"meta": {}, "results": []}
        assert http_client.get.call_count == 3

    def test_openalex_timeout_after_max_retries_raises(self):
        import pytest
        import requests as _requests

        http_client = MagicMock()
        http_client.get.side_effect = _requests.Timeout("read timed out")
        client = _make_client(http_client, max_retries=2, backoff_seconds=0)
        with pytest.raises(_requests.Timeout):
            client.search_works_by_title("t")
        assert http_client.get.call_count == 3  # 1 + 2 retries

    def test_arxiv_readtimeout_retried_then_succeeds(self):
        import requests as _requests

        http_client = MagicMock()
        good = MagicMock()
        good.status_code = 200
        good.text = (
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        http_client.get.side_effect = [
            _requests.ConnectionError("reset"),
            _requests.Timeout("read timed out"),
            good,
        ]
        client = _make_arxiv_client(http_client, max_retries=4, backoff_seconds=0)
        assert client.get_title_by_id("1706.03762") is None
        assert http_client.get.call_count == 3
