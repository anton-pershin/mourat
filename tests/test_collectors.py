"""Unit tests for collector modules with mocked HTTP responses."""

import datetime
from unittest.mock import MagicMock, patch

from mourat.collectors.arxiv import ArxivPaperCollector
from mourat.collectors.semantic_scholar import SemanticScholarPaperCollector
from mourat.data_models import PaperInfoCollection
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
        mock_response.text = '{"data": ' + str(SAMPLE_SS_RESPONSE["data"]).replace("'", '"') + ', "token": null}'

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
