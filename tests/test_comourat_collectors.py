"""comourat collector test: arXiv core produces a neutral collection (AC3).

Mirrors the mocked-feed style of tests/test_collectors.py but exercises the
comourat core standalone, using the neutral PaperCandidate model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from comourat.collectors.arxiv import (
    ArxivPaperCollectorCore,
    candidate_from_entry,
)
from comourat.models import PaperCandidate, PaperCandidateCollection

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
</feed>
"""


class TestCandidateFromEntry:
    def test_neutral_candidate_has_no_provenance_field(self):
        entry = SimpleNamespace(
            title="A paper",
            link="http://arxiv.org/abs/2401.00001",
            description="arXiv:2401.00001v1 [cs.AI] A paper\nAbstract: text",
            authors=[{"name": "J. Smith"}],
        )
        candidate = candidate_from_entry(entry)
        assert isinstance(candidate, PaperCandidate)
        assert candidate.arxiv_id == "2401.00001"
        assert candidate.authors == ["Smith"]
        assert "provenance" not in type(candidate).model_fields
        assert "announce_type" not in type(candidate).model_fields


class TestArxivCoreNewestMode:
    def test_collect_returns_neutral_collection(self):
        response = MagicMock()
        response.text = SAMPLE_ARXIV_FEED
        http_client = MagicMock()
        http_client.get.return_value = response

        collector = ArxivPaperCollectorCore(
            http_client=http_client,
            api_url="http://example.com/feed",
            mode="newest",
        )
        result = collector.collect()
        assert isinstance(result, PaperCandidateCollection)
        assert len(result.papers) == 1
        assert result.papers[0].title == "Test Paper One"
        assert result.papers[0].description == "Abstract of paper one"
        assert result.papers[0].urls_seen == ["http://arxiv.org/abs/2401.00001"]

    def test_filter_new_submissions(self):
        papers = [
            PaperCandidate(
                title="A",
                authors=["X"],
                description="d",
                arxiv_id="2401.00001",
            ),
            PaperCandidate(
                title="B",
                authors=["Y"],
                description="d",
                arxiv_id="2401.00002",
            ),
        ]
        for paper, announce_type in zip(papers, ["new", "replace"]):
            object.__setattr__(
                paper, "__dict__", {**paper.__dict__, "announce_type": announce_type}
            )
        collector = ArxivPaperCollectorCore.__new__(ArxivPaperCollectorCore)
        out, kept, excluded = collector.filter_new_submissions(
            PaperCandidateCollection(papers=papers)
        )
        assert (len(out.papers), kept, excluded) == (1, 1, 1)

    def test_most_relevant_requires_start_date(self):
        with pytest.raises(ValueError):
            ArxivPaperCollectorCore(
                http_client=MagicMock(),
                api_url="http://example.com",
                mode="most_relevant",
            )
