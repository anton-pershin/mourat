"""Unit tests for collector modules with mocked HTTP responses."""

import datetime
import itertools
from unittest.mock import MagicMock, patch

from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.seed_expander import SeedExpander
from mourat.collectors.semantic_scholar import SemanticScholarPaperCollector
from mourat.data_models import PaperInfoCollection, Seed, SeedCollection
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
            start_date="2024-01-01",
            end_date="2024-12-31",
            max_results=100,
        )

        result = collector(None, "test")

        assert isinstance(result, PaperInfoCollection)
        assert len(result.papers) == 2
        assert result.papers[0].title == "Test Paper One"
        assert result.papers[1].title == "Test Paper Two"
        assert result.papers[0].authors == ["Smith"]
        assert result.papers[0].publication_date == datetime.date(2024, 1, 15)

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


# --- SeedExpander (spec 11 task 4) ---


def _make_seed(work_id="https://openalex.org/W123", **overrides) -> Seed:
    defaults: dict = {
        "content_item_id": "ci1",
        "work_id": work_id,
        "title": "Attention Is All You Need",
        "influence_value": 90.0,
    }
    defaults.update(overrides)
    return Seed(**defaults)


def _make_expander(openalex=None, **kwargs) -> SeedExpander:
    if openalex is None:
        openalex = MagicMock()
        # Safe defaults so paging loops terminate; individual tests override.
        openalex.search_works_citing.return_value = {"meta": {}, "results": []}
        openalex.search_works_by_title.return_value = {"meta": {}, "results": []}
        openalex.get_work_references.return_value = []
    return SeedExpander(
        monitoring_handler=_make_monitoring_handler(),
        openalex_client=openalex,
        **kwargs,
    )


def _work(work_id="https://openalex.org/W9", title="Citing Work", **overrides):
    record = {
        "id": work_id,
        "display_name": title,
        "authorships": [{"author": {"display_name": "A. Author"}}],
    }
    record.update(overrides)
    return record


def _envelope(results, next_cursor=None):
    meta = {"next_cursor": next_cursor} if next_cursor else {}
    return {"meta": meta, "results": results}


class TestSeedExpander:
    def test_forward_expansion_returns_citing_works(self):
        openalex = MagicMock()
        openalex.search_works_citing.return_value = _envelope(
            [
                _work(work_id="W10", title="Citer One"),
                _work(work_id="W11", title="Citing Work"),
            ]
        )
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert {p.title for p in result.papers} == {"Citer One", "Citing Work"}
        assert result.papers[0].provenance == ["forward_citations"]

    def test_search_returns_paged_results_from_seed_title(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = _envelope(
            [_work(work_id="W20", title="Searched Work")]
        )
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert [p.title for p in result.papers] == ["Searched Work"]
        assert result.papers[0].provenance == ["relevance_search"]
        openalex.search_works_by_title.assert_called_once_with(
            "Attention Is All You Need", cursor="*"
        )

    def test_backward_expansion_returns_cited_works(self):
        openalex = MagicMock()
        openalex.get_work_references.return_value = ["W30", "W31"]
        openalex.get_work_by_id.side_effect = [
            _work(work_id="W30", title="Referenced One"),
            _work(work_id="W31", title="Referenced Two"),
        ]
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        titles = {p.title for p in result.papers}
        assert titles == {"Referenced One", "Referenced Two"}
        assert all(p.provenance == ["backward_references"] for p in result.papers)

    def test_backward_expansion_empty_references_is_no_error(self):
        """FR2: preprint-only records have no reference list; not an error."""
        openalex = MagicMock()
        openalex.get_work_references.return_value = []
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert result.papers == []

    def test_merge_dedups_by_work_id_with_full_provenance(self):
        """A work from several generators appears once, naming all of them."""
        openalex = MagicMock()
        openalex.search_works_citing.return_value = _envelope(
            [_work(work_id="W9", title="Shared Work")]
        )
        openalex.search_works_by_title.return_value = _envelope(
            [_work(work_id="W9", title="Shared Work")]
        )
        openalex.get_work_references.return_value = []
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert len(result.papers) == 1
        assert result.papers[0].provenance == [
            "forward_citations",
            "relevance_search",
        ]

    def test_forward_budget_bounds_citing_query_paging(self):
        openalex = MagicMock()
        # API offers endless pages with a fresh cursor each time
        counter = itertools.count()
        openalex.search_works_citing.side_effect = lambda work_id, cursor: _envelope(
            [_work(work_id=f"Wf{next(counter)}")], next_cursor=f"c{next(counter)}"
        )
        expander = _make_expander(openalex, forward_budget=3)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert len(result.papers) == 3

    def test_search_budget_bounds_paging(self):
        openalex = MagicMock()
        counter = itertools.count()
        openalex.search_works_by_title.side_effect = lambda title, cursor: _envelope(
            [_work(work_id=f"Ws{next(counter)}")], next_cursor=f"c{next(counter)}"
        )
        expander = _make_expander(openalex, search_budget=5)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert len(result.papers) == 5

    def test_backward_budget_bounds_reference_fetches(self):
        openalex = MagicMock()
        openalex.get_work_references.return_value = [f"Wb{i}" for i in range(50)]
        openalex.get_work_by_id.side_effect = lambda wid: _work(work_id=wid)
        expander = _make_expander(openalex, backward_budget=7)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert len(result.papers) == 7

    def test_records_without_identity_are_skipped(self):
        openalex = MagicMock()
        openalex.search_works_citing.return_value = _envelope(
            [{"display_name": "No Id Here"}, {"id": "", "display_name": "Empty Id"}]
        )
        expander = _make_expander(openalex)
        result = expander(SeedCollection(seeds=[_make_seed()]), "2")
        assert result.papers == []
