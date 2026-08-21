"""Test that each module imports independently without pulling in unrelated modules."""

import subprocess
import sys


def test_import_mourat():
    """Top-level import should succeed with no circular imports."""
    subprocess.check_call([sys.executable, "-c", "import mourat"])


def test_import_data_models():
    """Data models should be importable without pipeline logic."""
    subprocess.check_call(
        [sys.executable, "-c",
         "from mourat.data_models import (PaperInfo, ScoredPaperInfo, "
         "AssignedPaperInfo, PaperInfoCollection, ScoredPaperInfoCollection, "
         "AssignedPaperInfoCollection, QueryInfo, PaperScoredByAgent, "
         "ListOfTopics, BusinessProductInfo, CandidateTopicInfo, "
         "CandidateTopicRelevanceInfo, CandidateTopicAssessment)"]
    )


def test_import_collectors_arxiv():
    """ArxivCollector should be importable without processors."""
    subprocess.check_call(
        [sys.executable, "-c",
         "from mourat.collectors.arxiv import ArxivPaperCollector"]
    )


def test_import_collectors_semantic_scholar():
    """SemanticScholarCollector should be importable without processors."""
    subprocess.check_call(
        [sys.executable, "-c",
         "from mourat.collectors.semantic_scholar import SemanticScholarPaperCollector"]
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
        [sys.executable, "-c",
         "from mourat.monitoring import MonitoringHandler, MonitoringViaMarkdownFiles"]
    )


def test_import_base():
    """Function base class should be importable independently."""
    subprocess.check_call(
        [sys.executable, "-c", "from mourat.base import Function"]
    )
