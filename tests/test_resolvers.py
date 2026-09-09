"""Unit tests for PaperResolver and SeedResolver with mocked API clients."""

from unittest.mock import MagicMock

from mourat.data_models import (
    ContentItem,
    ContentItemCollection,
    PaperCandidate,
    PaperCandidateCollection,
    Seed,
)
from mourat.monitoring import MonitoringHandler
from mourat.resolvers.paper_resolver import (
    PaperResolver,
    _abstract_from_inverted_index,
    _extract_arxiv_id,
)
from mourat.resolvers.seed_resolver import SeedResolver


def _make_handler():
    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []

        def __call__(self, step, text_for_monitoring):
            self.calls.append((step, text_for_monitoring))

    return CapturingHandler()


def _make_candidate(**overrides) -> PaperCandidate:
    defaults: dict = {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani"],
        "description": "Introduces the transformer.",
        "urls_seen": ["https://arxiv.org/abs/1706.03762"],
    }
    defaults.update(overrides)
    return PaperCandidate(**defaults)


def _openalex_hit(**overrides) -> dict:
    record = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.5555/3294995",
        "display_name": "Attention Is All You Need",
        "publication_date": "2017-06-12",
        "authorships": [
            {"author": {"display_name": "Ashish Vaswani"}},
            {"author": {"display_name": "Noam Shazeer"}},
        ],
        "abstract_inverted_index": {"Attention": [0], "is": [1], "all": [2]},
        "fwci": 6139.0,
        "cited_by_count": 88404,
    }
    record.update(overrides)
    return record


def _make_resolver(openalex=None, arxiv=None, threshold=0.9) -> PaperResolver:
    return PaperResolver(
        monitoring_handler=_make_handler(),
        openalex_client=openalex or MagicMock(),
        arxiv_client=arxiv or MagicMock(),
        title_similarity_threshold=threshold,
    )


def _run(resolver, candidates):
    """Run the resolver and return (output, monitoring text).

    `Function.__call__` returns only the output; the monitoring text goes to
    the handler, so capture it there.
    """
    handler = resolver.monitoring_handler
    output = resolver(candidates, "2")
    assert handler.calls, "monitoring handler was not called"
    return output, handler.calls[-1][1]


class TestExtractArxivId:
    def test_abs_url(self):
        assert _extract_arxiv_id(["https://arxiv.org/abs/1706.03762"]) == "1706.03762"

    def test_pdf_url(self):
        assert _extract_arxiv_id(["https://arxiv.org/pdf/2401.00001v2"]) == "2401.00001"

    def test_no_arxiv_url_returns_none(self):
        assert _extract_arxiv_id(["https://example.com/paper"]) is None

    def test_version_suffix_stripped(self):
        assert _extract_arxiv_id(["https://arxiv.org/abs/1706.03762v3"]) == "1706.03762"


class TestAbstractFromInvertedIndex:
    def test_reconstructs_word_order(self):
        inv = {"all": [2], "is": [1], "Attention": [0]}
        assert _abstract_from_inverted_index(inv) == "Attention is all"

    def test_missing_index_returns_empty(self):
        assert _abstract_from_inverted_index(None) == ""


class TestResolveByTitle:
    def test_resolved_record_carries_canonical_metadata(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_resolver(openalex=openalex)
        result, monitoring = _run(
            resolver,
            PaperCandidateCollection(
                papers=[_make_candidate(urls_seen=["https://example.com/x"])]
            ),
        )
        assert len(result.papers) == 1
        rp = result.papers[0]
        assert rp.title == "Attention Is All You Need"
        assert rp.abstract == "Attention is all"
        assert rp.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert rp.publication_date == "2017-06-12"
        assert rp.work_id == "https://openalex.org/W123"
        assert rp.doi == "https://doi.org/10.5555/3294995"
        assert rp.url == ""  # verifier's job, stays empty here
        assert rp.resolution_status == "resolved"

    def test_no_api_hit_marks_unresolved_and_drops(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {"meta": {}, "results": []}
        resolver = _make_resolver(openalex=openalex)
        result, monitoring = _run(
            resolver,
            PaperCandidateCollection(
                papers=[_make_candidate(urls_seen=["https://example.com/x"])]
            ),
        )
        assert result.papers == []
        assert "unresolved" in monitoring.lower()
        assert "no_work_found" in monitoring


class TestResolveByArxivId:
    def test_arxiv_id_confirmed_then_openalex_hit_resolves(self):
        arxiv = MagicMock()
        arxiv.get_title_by_id.return_value = "Attention Is All You Need"
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_resolver(openalex=openalex, arxiv=arxiv)
        result, monitoring = _run(
            resolver, PaperCandidateCollection(papers=[_make_candidate()])
        )
        assert len(result.papers) == 1
        rp = result.papers[0]
        assert rp.arxiv_id == "1706.03762"
        assert rp.work_id == "https://openalex.org/W123"
        assert "arxiv=1706.03762" in monitoring

    def test_arxiv_title_mismatch_marks_unresolved(self):
        # A candidate whose arXiv id names a *different* paper: the id is a
        # hint only, the resolved record must carry nothing from it (FR1).
        arxiv = MagicMock()
        arxiv.get_title_by_id.return_value = "A Completely Different Work"
        resolver = _make_resolver(arxiv=arxiv)
        result, monitoring = _run(
            resolver, PaperCandidateCollection(papers=[_make_candidate()])
        )
        assert result.papers == []
        assert "arxiv_title_mismatch" in monitoring

    def test_unknown_arxiv_id_marks_unresolved(self):
        arxiv = MagicMock()
        arxiv.get_title_by_id.return_value = None
        resolver = _make_resolver(arxiv=arxiv)
        result, monitoring = _run(
            resolver, PaperCandidateCollection(papers=[_make_candidate()])
        )
        assert result.papers == []
        assert "1706.03762" in monitoring


class TestTitleCrossCheck:
    def test_openalex_top_hit_beyond_threshold_rejected_wholesale(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [
                _openalex_hit(display_name="Something Else Entirely About Models")
            ],
        }
        resolver = _make_resolver(openalex=openalex)
        result, monitoring = _run(
            resolver,
            PaperCandidateCollection(
                papers=[_make_candidate(urls_seen=["https://example.com/x"])]
            ),
        )
        assert result.papers == []
        assert "openalex_title_mismatch" in monitoring

    def test_supplied_doi_never_reaches_record_on_mismatch(self):
        # The rejected-shape test: a resolver that trusted the API record's
        # doi despite the title mismatch fails this.
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [
                _openalex_hit(
                    display_name="Different Paper",
                    doi="https://doi.org/10.5555/wrong",
                )
            ],
        }
        resolver = _make_resolver(openalex=openalex)
        result, _ = _run(
            resolver,
            PaperCandidateCollection(
                papers=[_make_candidate(urls_seen=["https://example.com/x"])]
            ),
        )
        assert result.papers == []


class TestMonitoring:
    def test_monitoring_leads_with_counts(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_resolver(openalex=openalex)
        _, monitoring = _run(
            resolver,
            PaperCandidateCollection(
                papers=[
                    _make_candidate(urls_seen=["https://example.com/a"]),
                    _make_candidate(
                        title="Unknown Thing", urls_seen=["https://example.com/b"]
                    ),
                ]
            ),
        )
        first_line = monitoring.split("\n")[0]
        assert "in: 2" in first_line
        assert "resolved: 1" in first_line
        assert "unresolved" in first_line

    def test_every_candidate_failing_resolution_completes_and_reports(self):
        """FR5: all-fail run completes, reports counts and every claimed title."""
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {"meta": {}, "results": []}
        resolver = _make_resolver(openalex=openalex)
        result, monitoring = _run(
            resolver,
            PaperCandidateCollection(
                papers=[
                    _make_candidate(
                        title="Ghost Paper One", urls_seen=["https://e.com/1"]
                    ),
                    _make_candidate(
                        title="Ghost Paper Two", urls_seen=["https://e.com/2"]
                    ),
                ]
            ),
        )
        assert result.papers == []  # nothing scored downstream
        assert "resolved: 0" in monitoring
        assert "Ghost Paper One" in monitoring
        assert "Ghost Paper Two" in monitoring


# --- SeedResolver (spec 11 task 3) ---


def _make_content_item(**overrides) -> ContentItem:
    defaults: dict = {
        "id": "ci_attention",
        "name": "Attention Is All You Need",
        "source_type_id": "paper",
        "platform_id": "arxiv",
        "influence_metric_id": "citations",
        "influence_score": 90,
    }
    defaults.update(overrides)
    return ContentItem(**defaults)


def _make_seed_resolver(openalex=None, threshold=0.9) -> SeedResolver:
    return SeedResolver(
        monitoring_handler=_make_handler(),
        openalex_client=openalex or MagicMock(),
        title_similarity_threshold=threshold,
    )


def _run_seeds(resolver, items):
    handler = resolver.monitoring_handler
    output = resolver(items, "2")
    assert handler.calls, "monitoring handler was not called"
    return output, handler.calls[-1][1]


class TestSeedResolver:
    def test_resolves_seed_via_title_search_with_work_id(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_seed_resolver(openalex=openalex)
        result, _ = _run_seeds(
            resolver, ContentItemCollection(items=[_make_content_item()])
        )
        assert len(result.seeds) == 1
        seed = result.seeds[0]
        assert seed.content_item_id == "ci_attention"
        assert seed.work_id == "https://openalex.org/W123"
        assert seed.title == "Attention Is All You Need"

    def test_seed_carries_stored_influence_value(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_seed_resolver(openalex=openalex)
        result, _ = _run_seeds(
            resolver, ContentItemCollection(items=[_make_content_item()])
        )
        assert result.seeds[0].influence_value == 90.0

    def test_title_mismatch_skips_seed_and_reports(self):
        """FR4: a seed failing the cross-check is skipped, not expanded."""
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit(display_name="A Different Work Entirely")],
        }
        resolver = _make_seed_resolver(openalex=openalex)
        result, monitoring = _run_seeds(
            resolver, ContentItemCollection(items=[_make_content_item()])
        )
        assert result.seeds == []
        assert "SEED SKIPPED" in monitoring
        assert "openalex_title_mismatch" in monitoring

    def test_no_work_found_skips_seed(self):
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {"meta": {}, "results": []}
        resolver = _make_seed_resolver(openalex=openalex)
        result, monitoring = _run_seeds(
            resolver,
            ContentItemCollection(items=[_make_content_item(name="Ghost Paper")]),
        )
        assert result.seeds == []
        assert "no_work_found" in monitoring

    def test_seed_without_stored_influence_resolves_with_none(self):
        """A seed with no stored influence still resolves; floor handles it."""
        openalex = MagicMock()
        openalex.search_works_by_title.return_value = {
            "meta": {},
            "results": [_openalex_hit()],
        }
        resolver = _make_seed_resolver(openalex=openalex)
        result, _ = _run_seeds(
            resolver,
            ContentItemCollection(items=[_make_content_item(influence_score=None)]),
        )
        assert result.seeds[0].influence_value is None

    def test_api_error_skips_seed_rather_than_raising(self):
        openalex = MagicMock()
        openalex.search_works_by_title.side_effect = RuntimeError("network down")
        resolver = _make_seed_resolver(openalex=openalex)
        result, monitoring = _run_seeds(
            resolver, ContentItemCollection(items=[_make_content_item()])
        )
        assert result.seeds == []
        assert "api_error" in monitoring

    def test_monitoring_leads_with_counts_and_skips(self):
        openalex = MagicMock()
        openalex.search_works_by_title.side_effect = [
            {"meta": {}, "results": [_openalex_hit()]},
            {"meta": {}, "results": []},
        ]
        resolver = _make_seed_resolver(openalex=openalex)
        _, monitoring = _run_seeds(
            resolver,
            ContentItemCollection(
                items=[
                    _make_content_item(),
                    _make_content_item(id="ci_ghost", name="Ghost Paper"),
                ]
            ),
        )
        first_line = monitoring.split("\n")[0]
        assert "in: 2" in first_line
        assert "resolved: 1" in first_line
        assert "skipped: 1" in first_line
        assert "Ghost Paper" in monitoring
