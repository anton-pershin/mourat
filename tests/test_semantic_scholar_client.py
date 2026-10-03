"""Tests for the Semantic Scholar graph client."""

from unittest.mock import MagicMock

from mourat.clients.paper_graph import PaperIdentity
from mourat.clients.semantic_scholar import SemanticScholarClient


def response(payload, status_code=200):
    result = MagicMock()
    result.status_code = status_code
    result.json.return_value = payload
    result.headers = {}
    return result


def make_client(http_client):
    return SemanticScholarClient(
        api_key="key",
        http_client=http_client,
        regular_delay_seconds=0,
        backoff_seconds=0,
    )


def test_resolve_by_arxiv_id_returns_normalized_record():
    http = MagicMock()
    http.get.return_value = response(
        {
            "title": "A paper",
            "externalIds": {"ArXiv": "1706.03762", "DOI": "10.48550/arXiv.1706.03762"},
            "authors": [{"name": "Author"}],
            "citationCount": 12,
            "publicationDate": "2017-06-02",
        }
    )
    result = make_client(http).resolve_by_arxiv_id("1706.03762")
    assert result is not None
    assert result.identity == PaperIdentity(arxiv_id="1706.03762")
    assert result.citation_count == 12
    assert result.raw_influence == {"citation_count": 12}
    assert result.authors == ["Author"]
    assert http.get.call_args.kwargs["params"]["fields"]


def test_get_citations_uses_identity_lookup_then_citation_endpoint():
    http = MagicMock()
    http.get.side_effect = [
        response(
            {
                "paperId": "internal",
                "title": "A paper",
                "externalIds": {"ArXiv": "1706.03762"},
            }
        ),
        response({"data": [], "next": None}),
    ]
    page = make_client(http).get_citations(PaperIdentity(arxiv_id="1706.03762"))
    assert page.papers == []


def test_api_key_is_sent_as_header_and_doi_path_is_encoded():
    http = MagicMock()
    http.get.return_value = response(
        {"title": "A paper", "externalIds": {"DOI": "10.1234/x/y"}}
    )
    make_client(http).resolve_by_doi("10.1234/x/y")
    call = http.get.call_args
    assert call.kwargs["headers"]["x-api-key"] == "key"
    assert "DOI:10.1234/x/y" in call.args[0]


def test_retry_after_is_used_for_rate_limit(monkeypatch):
    http = MagicMock()
    rate_limited = response({"message": "slow down"}, 429)
    rate_limited.headers = {"Retry-After": "7"}
    http.get.side_effect = [
        rate_limited,
        response({"title": "A paper", "externalIds": {}}),
    ]
    sleeps = []
    monkeypatch.setattr("mourat.clients.semantic_scholar.time.sleep", sleeps.append)
    monkeypatch.setattr("mourat.clients.semantic_scholar.random.uniform", lambda *_: 0)
    make_client(http)._resolve("ARXIV:1706.03762")
    assert sleeps == [7]


def test_nested_endpoint_path_preserves_slash():
    http = MagicMock()
    http.get.return_value = response({"title": "Paper", "externalIds": {}})
    client = make_client(http)
    client.resolve_by_arxiv_id("1706.03762")
    assert http.get.call_args.args[0].endswith("/paper/ARXIV:1706.03762")


def test_get_references_paginates_to_requested_limit():
    http = MagicMock()
    http.get.side_effect = [
        response({"paperId": "internal"}),
        response(
            {
                "data": [{"citedPaper": {"title": "One", "externalIds": {}}}],
                "next": 1,
            }
        ),
        response(
            {
                "data": [{"citedPaper": {"title": "Two", "externalIds": {}}}],
                "next": 2,
            }
        ),
    ]
    client = make_client(http)
    client.page_size = 1
    result = client.get_references(PaperIdentity(arxiv_id="1706.03762"), limit=2)
    assert [paper.title for paper in result] == ["One", "Two"]
    assert http.get.call_count == 3
