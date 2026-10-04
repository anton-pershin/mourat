from unittest.mock import MagicMock

from mourat.clients.paper_graph import PaperIdentity, PaperPage, PaperRecord
from mourat.data_models import (
    ContentItem,
    ContentItemCollection,
    PaperCandidate,
    PaperCandidateCollection,
    Seed,
    SeedCollection,
)
from mourat.monitoring import MonitoringHandler
from mourat.collectors.seed_expander import SeedExpander
from mourat.resolvers.paper_resolver import PaperResolver
from mourat.resolvers.seed_resolver import SeedResolver


class Handler(MonitoringHandler):
    def __init__(self):
        self.calls = []

    def __call__(self, step, text_for_monitoring):
        self.calls.append(text_for_monitoring)


def rec(arxiv_id=None, doi=None, title="Attention Is All You Need"):
    return PaperRecord(
        identity=PaperIdentity.from_values(arxiv_id=arxiv_id, doi=doi),
        title=title,
        authors=["Author"],
        abstract="Abstract",
        citation_count=12,
        raw_influence={"fwci": 2.0},
    )


def content_item():
    return ContentItem(
        id="seed-1",
        name="Attention Is All You Need",
        source_type_id="paper",
        platform_id="arxiv",
        influence_metric_id="citations",
        influence_score=90,
    )


def test_seed_resolver_uses_canonical_identity():
    client = MagicMock()
    client.search_papers.return_value = PaperPage(papers=[rec(arxiv_id="1706.03762")])
    result = SeedResolver(Handler(), paper_graph_client=client)(
        ContentItemCollection(items=[content_item()]), "1"
    )
    assert result.seeds[0].arxiv_id == "1706.03762"
    assert result.seeds[0].doi is None


def test_paper_resolver_identity_order():
    client = MagicMock()
    client.resolve_by_arxiv_id.return_value = rec(arxiv_id="1706.03762")
    candidate = PaperCandidate(
        title="Attention Is All You Need",
        authors=[],
        description="",
        urls_seen=[],
        arxiv_id="1706.03762",
    )
    result = PaperResolver(
        Handler(), paper_graph_client=client, arxiv_client=MagicMock()
    )(PaperCandidateCollection(papers=[candidate]), "1")
    assert result.papers[0].arxiv_id == "1706.03762"
    client.resolve_by_arxiv_id.assert_called_once_with("1706.03762")
    client.resolve_by_doi.assert_not_called()


def test_paper_resolver_falls_back_from_arxiv_to_doi():
    client = MagicMock()
    client.resolve_by_arxiv_id.return_value = None
    client.resolve_by_doi.return_value = rec(doi="10.1234/paper")
    candidate = PaperCandidate(
        title="Attention Is All You Need",
        authors=[],
        description="",
        urls_seen=[],
        arxiv_id="1706.03762",
        doi="10.1234/paper",
    )
    result = PaperResolver(
        Handler(), paper_graph_client=client, arxiv_client=MagicMock()
    )(PaperCandidateCollection(papers=[candidate]), "1")
    assert result.papers[0].doi == "10.1234/paper"
    client.resolve_by_doi.assert_called_once_with("10.1234/paper")


def test_seed_expander_uses_all_three_normalized_generators():
    client = MagicMock()
    client.get_citations.return_value = PaperPage(papers=[rec(doi="10.1/forward")])
    client.search_papers.return_value = PaperPage(papers=[rec(doi="10.1/search")])
    client.get_references.return_value = [rec(doi="10.1/backward")]
    seed = Seed(content_item_id="seed-1", arxiv_id="1706.03762", title="Attention")
    result = SeedExpander(
        Handler(),
        paper_graph_client=client,
        forward_budget=1,
        search_budget=1,
        backward_budget=1,
    )(SeedCollection(seeds=[seed]), "1")
    assert {paper.doi for paper in result.papers} == {
        "10.1/forward",
        "10.1/search",
        "10.1/backward",
    }
    client.get_citations.assert_called_once()
    client.get_references.assert_called_once()


def test_semantic_scholar_expansion_with_openalex_resolution():
    semantic_scholar = MagicMock(name="semantic_scholar_expansion")
    semantic_scholar.get_citations.return_value = PaperPage(
        papers=[rec(doi="10.1/discovered")]
    )
    semantic_scholar.search_papers.return_value = PaperPage(papers=[])
    semantic_scholar.get_references.return_value = []
    openalex = MagicMock(name="openalex_resolution")
    openalex.search_papers.return_value = PaperPage(papers=[rec(arxiv_id="1706.03762")])
    openalex.resolve_by_doi.return_value = rec(doi="10.1/discovered")

    seeds = SeedResolver(Handler(), paper_graph_client=openalex)(
        ContentItemCollection(items=[content_item()]), "1"
    )
    candidates = SeedExpander(
        Handler(),
        paper_graph_client=semantic_scholar,
        forward_budget=1,
        search_budget=0,
        backward_budget=0,
    )(seeds, "2")
    resolved = PaperResolver(Handler(), paper_graph_client=openalex)(candidates, "3")

    assert resolved.papers[0].doi == "10.1/discovered"
    openalex.resolve_by_doi.assert_called_once_with("10.1/discovered")
    semantic_scholar.resolve_by_doi.assert_not_called()
    semantic_scholar.resolve_by_arxiv_id.assert_not_called()


def test_default_and_expansion_clients_are_separate_boundaries():
    expansion = MagicMock(name="semantic_scholar")
    resolution = MagicMock(name="openalex")
    expansion.get_citations.return_value = PaperPage(papers=[])
    expansion.search_papers.return_value = PaperPage(papers=[])
    expansion.get_references.return_value = []
    resolution.search_papers.return_value = PaperPage(
        papers=[rec(arxiv_id="1706.03762")]
    )

    seed_result = SeedResolver(Handler(), paper_graph_client=resolution)(
        ContentItemCollection(items=[content_item()]), "1"
    )
    SeedExpander(
        Handler(),
        paper_graph_client=expansion,
        forward_budget=1,
        search_budget=1,
        backward_budget=1,
    )(SeedCollection(seeds=seed_result.seeds), "2")

    assert seed_result.seeds
    resolution.search_papers.assert_called_once()
    expansion.search_papers.assert_called_once()
    expansion.resolve_by_arxiv_id.assert_not_called()
    expansion.resolve_by_doi.assert_not_called()
