"""Unit tests for collector modules with mocked HTTP responses."""

import datetime
import itertools
from unittest.mock import MagicMock, patch

from mourat.clients.openalex import OpenAlexClient
from mourat.clients.paper_graph import PaperIdentity, PaperPage, PaperRecord
from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.seed_expander import SeedExpander
from mourat.collectors.semantic_scholar import SemanticScholarPaperCollector
from mourat.data_models import (
    PaperCandidateCollection,
    PaperInfoCollection,
    Seed,
    SeedCollection,
)
from mourat.monitoring import MonitoringHandler

# --- Helpers ---


def _make_monitoring_handler():
    """Create a MonitoringHandler subclass that does nothing."""

    class DummyHandler(MonitoringHandler):
        def __init__(self):
            pass

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            pass

    return DummyHandler()


SAMPLE_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper One</title>
    <link href="http://arxiv.org/abs/2401.00001"/>
    <description>arXiv:2401.00001v1 [cs.AI] Test Paper One
Abstract: Abstract of paper one</description>
    <author><name>J. Smith</name></author>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
  <entry>
    <title>Test Paper Two</title>
    <link href="http://arxiv.org/abs/2401.00002"/>
    <description>arXiv:2401.00002v1 [cs.AI] Test Paper Two
Abstract: Abstract of paper two</description>
    <author><name>A. Doe</name></author>
    <published>2024-01-16T00:00:00Z</published>
  </entry>
</feed>
"""


class TestArxivPaperCollector:
    """Tests for ArxivPaperCollector newest mode."""

    def test_newest_mode_parses_feed(self):
        mock_handler = _make_monitoring_handler()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_ARXIV_FEED
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        collector = ArxivPaperCollector(
            monitoring_handler=mock_handler,
            http_client=mock_client,
            api_url="http://example.com/feed",
            mode="newest",
        )

        result = collector(None, "test")  # T1

        assert isinstance(result, PaperCandidateCollection)
        assert len(result.papers) == 2
        first = result.papers[0]
        assert first.title == "Test Paper One"
        assert first.description == "Abstract of paper one"  # abstract -> description
        assert first.urls_seen == ["http://arxiv.org/abs/2401.00001"]
        assert first.arxiv_id == "2401.00001"
        assert first.authors == ["Smith"]
        assert first.publication_date == "2024-01-15"  # ISO string (T3b)
        assert result.papers[1].title == "Test Paper Two"

    def test_version_suffix_stripped_from_arxiv_id(self):
        # T2: link carrying a version suffix yields the base id
        feed = SAMPLE_ARXIV_FEED.replace(
            "http://arxiv.org/abs/2401.00001", "http://arxiv.org/abs/2401.00001v2"
        )
        mock_handler = _make_monitoring_handler()
        mock_response = MagicMock()
        mock_response.text = feed
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        collector = ArxivPaperCollector(
            monitoring_handler=mock_handler,
            http_client=mock_client,
            api_url="http://example.com/feed",
            mode="newest",
        )
        result = collector(None, "test")
        assert result.papers[0].arxiv_id == "2401.00001"

    def test_provenance_is_rss(self):
        mock_handler = _make_monitoring_handler()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_ARXIV_FEED
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        collector = ArxivPaperCollector(
            monitoring_handler=mock_handler,
            http_client=mock_client,
            api_url="http://example.com/feed",
            mode="newest",
        )
        result = collector(None, "test")
        assert result.papers[0].provenance == ["arxiv_rss"]

    def test_newest_mode_raises_on_missing_start_date(self):
        mock_handler = _make_monitoring_handler()
        mock_client = MagicMock()

        try:
            ArxivPaperCollector(
                monitoring_handler=mock_handler,
                http_client=mock_client,
                api_url="http://example.com/feed",
                mode="most_relevant",
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


SAMPLE_SS_RESPONSE = {
    "data": [
        {
            "title": "SS Paper One",
            "url": "https://example.com/paper1",
            "abstract": "Abstract one",
            "citationCount": 42,
            "publicationDate": "2024-01-15",
            "authors": [{"name": "J. Smith"}],
        },
        {
            "title": "SS Paper Two",
            "url": "https://example.com/paper2",
            "abstract": "Abstract two",
            "citationCount": 10,
            "publicationDate": "2024-02-20",
            "authors": [{"name": "A. Doe"}],
        },
    ],
    "token": None,
    "next": None,
}


class TestSemanticScholarPaperCollector:
    """Tests for SemanticScholarPaperCollector."""

    def test_bulk_search_parses_response(self):
        mock_handler = _make_monitoring_handler()
        mock_response = MagicMock()
        mock_response.text = '{"data": [], "token": null}'
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        collector = SemanticScholarPaperCollector(
            monitoring_handler=mock_handler,
            http_client=mock_client,
            api_url="https://api.semanticscholar.org/graph/v1",
            mode="newest",
            start_date="2024-01-01",
            end_date="2024-12-31",
            max_results=10,
            strict_keyword_query="test query",
        )

        # Force the mock to return our sample data
        mock_response.text = (
            '{"data": '
            + str(SAMPLE_SS_RESPONSE["data"]).replace("'", '"')
            + ', "token": null}'
        )

        result = collector(None, "test")

        assert isinstance(result, PaperInfoCollection)
        assert len(result.papers) == 2
        assert result.papers[0].title == "SS Paper One"
        assert result.papers[0].citation_count == 42
        assert result.papers[0].publication_date == datetime.date(2024, 1, 15)

    def test_filters_entries_with_missing_mandatory_fields(self):
        mock_handler = _make_monitoring_handler()
        mock_response = MagicMock()
        # First entry missing title, second entry valid
        mock_response.text = (
            '{"data": ['
            '{"title": null, "url": "https://x.com", "abstract": "abs", "citationCount": 0, "publicationDate": null, "authors": []},'
            '{"title": "Valid Paper", "url": "https://x.com/2", "abstract": "abs2", "citationCount": 5, "publicationDate": "2024-03-01", "authors": [{"name": "Test"}]}'
            '], "token": null}'
        )
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        collector = SemanticScholarPaperCollector(
            monitoring_handler=mock_handler,
            http_client=mock_client,
            api_url="https://api.semanticscholar.org/graph/v1",
            mode="newest",
            start_date="2024-01-01",
            end_date="2024-12-31",
            max_results=10,
            strict_keyword_query="test",
        )

        result = collector(None, "test")

        assert len(result.papers) == 1
        assert result.papers[0].title == "Valid Paper"


# --- Provider-neutral SeedExpander ---


def _make_seed(**overrides) -> Seed:
    values = {
        "content_item_id": "ci1",
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "influence_value": 90.0,
    }
    values.update(overrides)
    return Seed(**values)


def _record(doi: str, title: str) -> PaperRecord:
    return PaperRecord(
        identity=PaperIdentity.from_values(doi=doi),
        title=title,
        authors=["A. Author"],
    )


def _make_expander(client=None, **kwargs) -> SeedExpander:
    return SeedExpander(
        monitoring_handler=_make_monitoring_handler(),
        paper_graph_client=client or MagicMock(),
        **kwargs,
    )


class TestSeedExpander:
    def test_openalex_client_exercises_real_reference_call_path(self):
        """AC10: expansion must call OpenAlexClient.get_references, not a stub."""
        seed_work = {
            "id": "https://openalex.org/Wseed",
            "display_name": "Attention Is All You Need",
            "doi": "10.48550/arXiv.1706.03762",
            "locations": [],
            "referenced_works": ["https://openalex.org/Wref"],
        }
        reference_work = {
            "id": "https://openalex.org/Wref",
            "display_name": "Reference Paper",
            "doi": "https://doi.org/10.1000/reference",
            "locations": [],
        }

        class Response:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

            def raise_for_status(self):
                raise AssertionError("unexpected HTTP error")

        class HttpClient:
            def get(self, url, params, timeout):
                if url.endswith("/Wseed"):
                    return Response(seed_work)
                if url.endswith("/Wref"):
                    return Response(reference_work)
                if params.get("filter", "").startswith("doi:"):
                    return Response({"results": [seed_work]})
                if params.get("filter", "").startswith("cites:"):
                    return Response({"meta": {"next_cursor": None}, "results": []})
                if params.get("search"):
                    return Response({"meta": {"next_cursor": None}, "results": []})
                raise AssertionError((url, params))

        client = OpenAlexClient(
            user_agent="test",
            http_client=HttpClient(),
            regular_delay_seconds=0,
        )
        result = SeedExpander(
            monitoring_handler=_make_monitoring_handler(),
            paper_graph_client=client,
            forward_budget=1,
            search_budget=1,
            backward_budget=1,
        )(SeedCollection(seeds=[_make_seed()]), "2")

        assert [paper.title for paper in result.papers] == ["Reference Paper"]
        assert result.papers[0].provenance == ["backward_references"]

    def test_runs_all_generators_and_preserves_provenance(self):
        client = MagicMock()
        client.get_citations.return_value = PaperPage(
            papers=[_record("10.1/forward", "Forward")]
        )
        client.search_papers.return_value = PaperPage(
            papers=[_record("10.1/search", "Search")]
        )
        client.get_references.return_value = [_record("10.1/backward", "Backward")]
        expander = _make_expander(client)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert {paper.title for paper in result.papers} == {
            "Forward",
            "Search",
            "Backward",
        }
        assert {paper.provenance[0] for paper in result.papers} == {
            "forward_citations",
            "relevance_search",
            "backward_references",
        }

    def test_deduplicates_by_canonical_identity(self):
        client = MagicMock()
        shared = _record("10.1/shared", "Shared")
        client.get_citations.return_value = PaperPage(papers=[shared])
        client.search_papers.return_value = PaperPage(papers=[shared])
        client.get_references.return_value = []
        result = _make_expander(client)(SeedCollection(seeds=[_make_seed()]), "2")
        assert len(result.papers) == 1
        assert result.papers[0].provenance == ["forward_citations", "relevance_search"]

    def test_budgets_bound_each_generator(self):
        client = MagicMock()
        client.get_citations.side_effect = lambda identity, continuation: PaperPage(
            papers=[_record(f"10.1/f{continuation or 0}", "Forward")],
            continuation=(continuation or 0) + 1,
        )
        client.search_papers.side_effect = lambda query, continuation: PaperPage(
            papers=[_record(f"10.1/s{continuation or 0}", "Search")],
            continuation=(continuation or 0) + 1,
        )
        client.get_references.return_value = [
            _record(f"10.1/b{i}", "Backward") for i in range(20)
        ]
        result = _make_expander(
            client, forward_budget=3, search_budget=4, backward_budget=5
        )(SeedCollection(seeds=[_make_seed()]), "2")
        assert (
            len([p for p in result.papers if p.provenance == ["forward_citations"]])
            == 3
        )
        assert (
            len([p for p in result.papers if p.provenance == ["relevance_search"]]) == 4
        )
        assert (
            len([p for p in result.papers if p.provenance == ["backward_references"]])
            == 5
        )
