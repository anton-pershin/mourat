"""Keep the README script catalogue and architecture notes consistent."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SCRIPT_NAMES = sorted(
    path.stem
    for path in (ROOT / "mourat" / "scripts").glob("*.py")
    if path.stem != "__init__"
)

PIPELINE_STAGES = {
    "collect_newest_papers": (
        "ArxivPaperCollector",
        "SemanticScholarNewestPaperCollector",
        "PaperCandidateDeduplicator",
        "RelevanceTriageClassifier",
        "ArxivHtmlAffiliationFetcher",
        "AuthorityInfluenceAssessor",
        "CandidateToResolvedConverter",
        "PaperContentItemScorer",
        "PaperScoreFilter",
    ),
    "collect_influential_papers_from_scratch": (
        "PaperDiscoverer",
        "PaperResolver",
        "InfluenceAssessor",
        "ArxivPdfVerifier",
        "PaperContentItemScorer",
        "PaperScoreFilter",
    ),
    "collect_influential_papers_from_seeds": (
        "SeedResolver",
        "SeedExpander",
        "PaperResolver",
        "InfluenceAssessor",
        "InfluenceFloorFilter",
        "ArxivPdfVerifier",
        "PaperContentItemScorer",
        "PaperScoreFilter",
    ),
    "collect_posts": (
        "RedditPostCollector",
        "PostInfluenceAssessor",
        "PostInfluenceFilter",
        "HeuristicSlopFilter",
        "PostSlopClassifier",
        "SlopFilter",
        "WebEnricher",
        "PostContentItemScorer",
        "PostScoreFilter",
    ),
    "collect_hn_posts": (
        "HNPostCollector",
        "HNInfluenceAssessor",
        "HNInfluenceFilter",
        "HNContentFetcher",
        "HNSummarizer",
        "HNContentItemScorer",
        "HNScoreFilter",
    ),
}


def _script_section(script_name: str) -> str:
    """Return one script section without the following script sections."""
    marker = f"### `{script_name}.py`"
    assert marker in README, f"Missing README section for {script_name}"
    section = README.split(marker, maxsplit=1)[1]
    next_heading = re.search(r"^###? ", section, flags=re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]
    return section


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_readme_documents_every_script_and_its_config(script_name: str) -> None:
    """Every executable script has a section and a full configuration path."""
    section = _script_section(script_name)
    assert f"`config/config_{script_name}.yaml`" in section


@pytest.mark.parametrize("script_name", sorted(PIPELINE_STAGES))
def test_readme_lists_active_pipeline_stages_in_order(script_name: str) -> None:
    """The pipeline summary follows the current orchestrator order."""
    section = _script_section(script_name)
    positions = []
    for stage in PIPELINE_STAGES[script_name]:
        marker = f"`{stage}`"
        assert marker in section, f"Missing pipeline stage: {stage}"
        positions.append(section.index(marker))
    assert positions == sorted(positions)


def test_newest_pipeline_does_not_use_retired_classifier_description() -> None:
    """The newest-paper section describes relevance triage, not old classifiers."""
    section = _script_section("collect_newest_papers")
    assert "BinaryPaperClassifier" not in section
    assert "ScoreBasedPaperFilter" not in section
    assert "default: 4" not in section


def test_newest_pipeline_documents_semantic_scholar_state_and_identity() -> None:
    section = _script_section("collect_newest_papers")
    for text in (
        "sources=[arxiv]",
        "sources=[semantic_scholar]",
        "publication date",
        "backfill",
        "DOI",
        "after all downstream stages",
    ):
        assert text in section


def test_readme_describes_shared_architecture_and_operation() -> None:
    """The README identifies shared modules and basic operating controls."""
    required_text = (
        "Python 3.12",
        "MOURAT_DB_PATH",
        "db_writer.enabled",
        "jsonl_writer.enabled",
        "`Function`",
        "`comourat`",
        "`mourat/scripts/`",
        "`mourat/base.py`",
        "`mourat/data_models.py`",
        "`mourat/collectors/`",
        "`mourat/resolvers/`",
        "`mourat/processors/`",
        "`mourat/enrichers/`",
        "`mourat/writers/`",
        "`mourat/database/`",
        "`mourat/monitoring.py`",
        "`config/`",
    )
    for text in required_text:
        assert text in README, f"Missing architecture or operation text: {text}"


def test_readme_documents_only_supported_retrieval_query_types() -> None:
    """The retrieval section matches the CLI dispatch in the implementation."""
    section = _script_section("retrieve_content")
    for query_type in (
        "keywords",
        "research_question",
        "technical_challenge",
        "research_topic",
        "influence_score",
    ):
        assert f"`{query_type}`" in section
    assert "query_type=constraint" not in section
