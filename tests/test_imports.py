"""Test that each module imports independently without pulling in unrelated modules."""

import subprocess
import sys


def test_import_mourat():
    """Top-level import should succeed with no circular imports."""
    subprocess.check_call([sys.executable, "-c", "import mourat"])


def test_import_data_models():
    """Data models should be importable without pipeline logic."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.data_models import (PaperInfo, ScoredPaperInfo, "
            "AssignedPaperInfo, PaperInfoCollection, ScoredPaperInfoCollection, "
            "AssignedPaperInfoCollection, QueryInfo, PaperScoredByAgent, "
            "ListOfTopics, BusinessProductInfo, CandidateTopicInfo, "
            "CandidateTopicRelevanceInfo, CandidateTopicAssessment)",
        ]
    )


def test_import_collectors_arxiv():
    """ArxivCollector should be importable without processors."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.collectors.arxiv import ArxivPaperCollector",
        ]
    )


def test_import_collectors_semantic_scholar():
    """SemanticScholarCollector should be importable without processors."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.collectors.semantic_scholar import SemanticScholarPaperCollector",
        ]
    )


def test_import_classifiers():
    """BinaryPaperClassifier should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.classifiers import BinaryPaperClassifier"]
    )


def test_import_scorers():
    """PaperScorer should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.scorers import PaperScorer"]
    )


def test_import_assigners():
    """PaperAssigner should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.assigners import PaperAssigner"]
    )


def test_import_filters():
    """ScoreBasedPaperFilter should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.filters import ScoreBasedPaperFilter"]
    )


def test_import_generators():
    """QueryGeneratorViaLlm should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.generators import QueryGeneratorViaLlm"]
    )


def test_import_assessors():
    """CandidateTopicAssessor should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.assessors import CandidateTopicAssessor"]
    )


def test_import_monitoring():
    """MonitoringHandler and MonitoringViaMarkdownFiles should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.monitoring import MonitoringHandler, MonitoringViaMarkdownFiles",
        ]
    )


def test_import_base():
    """Function base class should be importable independently."""
    subprocess.check_call([sys.executable, "-c", "from mourat.base import Function"])


def test_import_post_slop_classifier():
    """PostSlopClassifier should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.classifiers import PostSlopClassifier"]
    )


def test_import_clients():
    """OpenAlexClient and ArxivClient should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.clients import OpenAlexClient, ArxivClient"]
    )


def test_import_similarity():
    """The shared title-similarity utility should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.utils.similarity import normalize_title, title_similarity, "
            "titles_match",
        ]
    )


def test_import_resolvers():
    """PaperResolver should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.resolvers import PaperResolver"]
    )


def test_import_influence_assessor():
    """InfluenceAssessor should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.processors.influence_assessor import InfluenceAssessor",
        ]
    )


def test_import_arxiv_pdf_verifier():
    """ArxivPdfVerifier should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.processors.arxiv_pdf_verifier import ArxivPdfVerifier",
        ]
    )


def test_import_resolved_paper_models():
    """ResolvedPaper(Collection) should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.data_models import ResolvedPaper, ResolvedPaperCollection",
        ]
    )


def test_import_seed_models():
    """Seed(Collection) and ContentItemCollection should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.data_models import Seed, SeedCollection, ContentItemCollection",
        ]
    )


def test_import_seed_resolver():
    """SeedResolver should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.resolvers import SeedResolver"]
    )


def test_import_seed_expander():
    """SeedExpander should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.collectors.seed_expander import SeedExpander",
        ]
    )


def test_import_influence_floor_filter():
    """InfluenceFloorFilter should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.filters import InfluenceFloorFilter"]
    )


def test_import_seed_expansion_script():
    """The seed-expansion entry-point module should be importable independently."""
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "from mourat.scripts import collect_influential_papers_from_seeds",
        ]
    )
