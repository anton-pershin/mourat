"""Tests for database CRUD operations and query engine."""

import os
import tempfile

import pytest

from mourat.database import init_db
from mourat.database import business_domain as bd
from mourat.database import research_domain as rd
from mourat.database import content_item as ci
from mourat.database import query_engine as qe


@pytest.fixture()
def conn():
    """Provide a temporary in-memory database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = init_db(path)
    yield c
    c.close()
    os.unlink(path)


# -- Business domain CRUD --

class TestBusinessDomain:
    def test_create_and_get(self, conn):
        bd.create_business_domain(conn, "ml", "Machine Learning", "ML domain")
        result = bd.get_business_domain(conn, "ml")
        assert result["id"] == "ml"
        assert result["name"] == "Machine Learning"
        assert result["description"] == "ML domain"

    def test_update(self, conn):
        bd.create_business_domain(conn, "ml", "ML", "")
        bd.update_business_domain(conn, "ml", name="Machine Learning")
        result = bd.get_business_domain(conn, "ml")
        assert result["name"] == "Machine Learning"

    def test_delete(self, conn):
        bd.create_business_domain(conn, "ml", "ML", "")
        bd.delete_business_domain(conn, "ml")
        assert bd.get_business_domain(conn, "ml") is None

    def test_list(self, conn):
        bd.create_business_domain(conn, "a", "A")
        bd.create_business_domain(conn, "b", "B")
        result = bd.list_business_domains(conn)
        assert len(result) == 2

    def test_product_crud(self, conn):
        bd.create_business_domain(conn, "ml", "ML")
        bd.create_product(conn, "ml-prod", "ML Product", "ml")
        assert bd.get_product(conn, "ml-prod") is not None
        bd.update_product(conn, "ml-prod", name="Updated Product")
        assert bd.get_product(conn, "ml-prod")["name"] == "Updated Product"
        bd.delete_product(conn, "ml-prod")
        assert bd.get_product(conn, "ml-prod") is None

    def test_technology_challenge_link(self, conn):
        bd.create_business_domain(conn, "ml", "ML")
        bd.create_product(conn, "p1", "P1", "ml")
        bd.create_technology(conn, "t1", "Tech", "p1")
        bd.create_technical_challenge(conn, "c1", "Challenge")
        bd.add_technology_challenge(conn, "t1", "c1")
        challenges = bd.list_technology_challenges(conn, "t1")
        assert len(challenges) == 1
        assert challenges[0]["name"] == "Challenge"


# -- Research domain CRUD --

class TestResearchDomain:
    def test_create_and_get(self, conn):
        rd.create_research_domain(conn, "rd1", "Research Domain", "desc")
        result = rd.get_research_domain(conn, "rd1")
        assert result["id"] == "rd1"
        assert result["name"] == "Research Domain"

    def test_hierarchy(self, conn):
        rd.create_research_domain(conn, "rd", "RD")
        rd.create_research_direction(conn, "dir", "Dir", "rd")
        rd.create_research_object(conn, "obj", "Obj", "dir")
        rd.create_research_question(conn, "q", "Q", "obj")
        rd.create_research_topic(conn, "topic", "Topic")
        rd.add_topic_research_question(conn, "topic", "q")
        questions = rd.list_topic_research_questions(conn, "topic")
        assert len(questions) == 1
        assert questions[0]["id"] == "q"


# -- Content item CRUD --

class TestContentItem:
    def test_create_and_get(self, conn):
        ci.create_source_type(conn, "paper", "Paper")
        ci.create_platform(conn, "arxiv", "Arxiv")
        ci.create_influence_metric(conn, "citations", "Citations")
        ci.create_content_item(
            conn, "item1", "Test Paper",
            source_type_id="paper", platform_id="arxiv",
            influence_metric_id="citations", influence_score=50,
        )
        result = ci.get_content_item(conn, "item1")
        assert result["name"] == "Test Paper"
        assert result["influence_score"] == 50

    def test_relevance_scores(self, conn):
        ci.create_source_type(conn, "paper", "Paper")
        ci.create_platform(conn, "arxiv", "Arxiv")
        ci.create_influence_metric(conn, "citations", "Citations")
        ci.create_content_item(
            conn, "item1", "Test Paper",
            source_type_id="paper", platform_id="arxiv",
            influence_metric_id="citations",
        )
        bd.create_technical_challenge(conn, "tc1", "TC")
        ci.add_item_technical_challenge(conn, "item1", "tc1", "Good match", 80)
        tc_links = ci.list_item_technical_challenges(conn, "item1")
        assert len(tc_links) == 1
        assert tc_links[0]["relevance_score"] == 80
        assert tc_links[0]["justification"] == "Good match"


# -- Query engine --

class TestQueryEngine:
    @pytest.fixture(autouse=True)
    def _seed(self, conn):
        ci.create_source_type(conn, "paper", "Paper")
        ci.create_platform(conn, "arxiv", "Arxiv")
        ci.create_influence_metric(conn, "citations", "Citations")
        ci.create_content_item(
            conn, "item1", "Alpha beta gamma",
            source_type_id="paper", platform_id="arxiv",
            influence_metric_id="citations", influence_score=70,
        )
        ci.create_content_item(
            conn, "item2", "Beta delta epsilon",
            source_type_id="paper", platform_id="arxiv",
            influence_metric_id="citations", influence_score=30,
        )

    def test_search_by_keywords(self, conn):
        results = qe.search_by_keywords(conn, "alpha")
        assert len(results) == 1
        assert results[0]["id"] == "item1"

    def test_search_by_influence_score(self, conn):
        results = qe.search_by_influence_score(conn, 50)
        assert len(results) == 1
        assert results[0]["id"] == "item1"

    def test_search_by_research_question(self, conn):
        ci.create_content_item(
            conn, "item3", "Linked item",
            source_type_id="paper", platform_id="arxiv",
            influence_metric_id="citations", influence_score=90,
        )
        rd.create_research_domain(conn, "rd", "RD")
        rd.create_research_direction(conn, "dir", "Dir", "rd")
        rd.create_research_object(conn, "obj", "Obj", "dir")
        rd.create_research_question(conn, "q1", "Q1", "obj")
        ci.add_item_research_question(conn, "item3", "q1", "relevant", 85)
        results = qe.search_by_research_question(conn, "q1", min_score=80)
        assert len(results) == 1
        assert results[0]["id"] == "item3"

    def test_search_by_technical_challenge(self, conn):
        bd.create_technical_challenge(conn, "tc1", "TC")
        ci.add_item_technical_challenge(conn, "item1", "tc1", "match", 60)
        results = qe.search_by_technical_challenge(conn, "tc1", min_score=50)
        assert len(results) == 1
        assert results[0]["id"] == "item1"

    def test_search_by_research_topic(self, conn):
        rd.create_research_topic(conn, "topic1", "Topic")
        ci.add_item_research_topic(conn, "item2", "topic1", "match", 40)
        results = qe.search_by_research_topic(conn, "topic1", min_score=30)
        assert len(results) == 1
        assert results[0]["id"] == "item2"

    def test_fts5_boolean_query(self, conn):
        """FTS5 should support boolean operators."""
        # Both items have "beta", only item1 has "alpha"
        results = qe.search_by_keywords(conn, "alpha AND beta")
        assert len(results) == 1
        assert results[0]["id"] == "item1"
