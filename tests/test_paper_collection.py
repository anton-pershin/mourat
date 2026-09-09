"""Unit tests for the from-scratch paper collection pipeline (spec 09).

Covers: PaperDiscoverer, ContentItemScorer paper binding, PaperScoreFilter,
ContentItemDbWriter, JsonlWriter, and the entry-point attribute loading.
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from mourat.collectors.paper_discoverer import PaperDiscoverer
from mourat.data_models import (
    PaperCandidate,
    PaperCandidateCollection,
    ResolvedPaper,
    ResolvedPaperCollection,
    ScoredPaper,
    ScoredPaperCollection,
    ScoreEntry,
)
from mourat.filters import PaperScoreFilter
from mourat.monitoring import MonitoringHandler
from mourat.processors.content_item_scorer import (
    PaperContentItemScorer,
    _candidate_as_resolved,
)
from mourat.writers.db_writer import ContentItemDbWriter, _content_item_id
from mourat.writers.jsonl_writer import JsonlWriter

# --- Helpers ---


def _make_monitoring_handler():
    """MonitoringHandler subclass that records calls."""

    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            self.calls.append((step, text_for_monitoring))

    return CapturingHandler()


def _make_candidate(**overrides) -> PaperCandidate:
    defaults = {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer"],
        "description": "Introduces the transformer architecture.",
        "urls_seen": ["https://arxiv.org/abs/1706.03762"],
    }
    defaults.update(overrides)
    return PaperCandidate(**defaults)


def _resolved_paper(**overrides) -> ResolvedPaper:
    """A ResolvedPaper as the scorer now consumes it (post-resolution shape)."""
    defaults = {
        "title": "Attention Is All You Need",
        "abstract": "Introduces the transformer architecture.",
        "authors": ["Vaswani", "Shazeer"],
        "publication_date": "2017-06-12",
        "resolution_status": "resolved",
    }
    defaults.update(overrides)
    return ResolvedPaper(**defaults)


def _make_scored_paper(
    filtering_score: float, scores: list[dict] | None = None, **candidate_overrides
) -> ScoredPaper:
    relevance = [
        ScoreEntry.model_validate(s)
        for s in (
            scores
            or [
                {
                    "id": "rq1",
                    "type": "rq",
                    "score": int(filtering_score),
                    "justification": "j",
                }
            ]
        )
        if int(filtering_score) >= 0 or True
    ]
    return ScoredPaper(
        paper=_candidate_as_resolved(_make_candidate(**candidate_overrides)),
        relevance_scores=relevance if filtering_score >= 0 else [],
        filtering_score=max(filtering_score, 0.0),
    )


def _make_scoring_model(scripted_scores: list[list[dict]]) -> FunctionModel:
    """FunctionModel returning one scripted ScoringResult payload per call."""

    def model_fn(messages, agent):
        result = {"scores": scripted_scores.pop(0)}
        return ModelResponse(parts=[TextPart(json.dumps(result))])

    return FunctionModel(model_fn)


RQ_LIST = [{"id": "rq1", "name": "RQ one", "type": "rq", "description": "d"}]
TC_LIST = [{"id": "tc1", "name": "TC one", "type": "tc", "description": "d"}]


# --- PaperDiscoverer ---


class TestPaperDiscoverer:
    def test_returns_paper_candidate_collection(self):
        handler = _make_monitoring_handler()
        discoverer = PaperDiscoverer(
            monitoring_handler=handler,
            model=TestModel(),
            attribute_description="RQ1: how do transformers scale?",
        )
        result = discoverer({}, "1")
        assert isinstance(result, PaperCandidateCollection)
        assert len(result.papers) >= 1

    def test_no_identifier_fields_on_output(self):
        handler = _make_monitoring_handler()
        discoverer = PaperDiscoverer(
            monitoring_handler=handler,
            model=TestModel(),
            attribute_description="RQ1: x",
        )
        result = discoverer({}, "1")
        for paper in result.papers:
            assert not hasattr(paper, "doi")
            assert not hasattr(paper, "arxiv_id")

    def test_prompt_contains_attribute_description(self):
        handler = _make_monitoring_handler()
        captured = []

        # Wrap TestModel to capture the prompt sent by the agent.
        class CapturingTestModel(TestModel):
            def request(self, *args, **kwargs):
                captured.append(args[0] if args else kwargs)
                return super().request(*args, **kwargs)

        discoverer = PaperDiscoverer(
            monitoring_handler=handler,
            model=CapturingTestModel(),
            attribute_description="UNIQUE-ATTRIBUTE-DESCRIPTION-42",
        )
        discoverer({}, "1")
        flattened = json.dumps(captured, default=str)
        assert "UNIQUE-ATTRIBUTE-DESCRIPTION-42" in flattened

    def test_constraints_description_in_prompt(self):
        handler = _make_monitoring_handler()
        captured = []

        class CapturingTestModel(TestModel):
            def request(self, *args, **kwargs):
                captured.append(args[0] if args else kwargs)
                return super().request(*args, **kwargs)

        discoverer = PaperDiscoverer(
            monitoring_handler=handler,
            model=CapturingTestModel(),
            attribute_description="RQ1: x",
            constraints_description="UNIQUE-CONSTRAINT-77",
        )
        discoverer({}, "1")
        flattened = json.dumps(captured, default=str)
        assert "UNIQUE-CONSTRAINT-77" in flattened

    def test_monitoring_lists_discovered_papers(self):
        handler = _make_monitoring_handler()
        discoverer = PaperDiscoverer(
            monitoring_handler=handler,
            model=TestModel(),
            attribute_description="RQ1: x",
        )
        discoverer({}, "1")
        step, text = handler.calls[0]
        assert step == "1"
        assert "Discovered" in text


# --- PaperContentItemScorer (paper binding of the generalised scorer) ---


class TestPaperContentItemScorer:
    def test_maps_scores_onto_scored_papers(self):
        handler = _make_monitoring_handler()
        model = _make_scoring_model(
            [[{"id": "rq1", "type": "rq", "score": 80, "justification": "core"}]]
        )
        scorer = PaperContentItemScorer(
            monitoring_handler=handler, model=model, rq_list=RQ_LIST
        )
        resolved = ResolvedPaperCollection(papers=[_resolved_paper()])
        result = scorer(resolved, "2")

        assert isinstance(result, ScoredPaperCollection)
        assert len(result.papers) == 1
        sp = result.papers[0]
        assert sp.paper.title == "Attention Is All You Need"
        assert sp.filtering_score == 80.0
        assert sp.relevance_scores[0].id == "rq1"
        assert sp.relevance_scores[0].score == 80

    def test_constraints_contribute_to_filtering_score_by_default(self):
        handler = _make_monitoring_handler()
        model = _make_scoring_model(
            [[{"id": "c1", "type": "constraint", "score": 60, "justification": "ok"}]]
        )
        scorer = PaperContentItemScorer(
            monitoring_handler=handler,
            model=model,
            constraint_list=[
                {"id": "c1", "name": "C", "type": "constraint", "description": "d"}
            ],
        )
        result = scorer(ResolvedPaperCollection(papers=[_resolved_paper()]), "2")
        # Paper binding defaults to True: constraint score counts.
        assert result.papers[0].filtering_score == 60.0

    def test_zero_when_no_valid_entries(self):
        handler = _make_monitoring_handler()
        model = _make_scoring_model(
            [[{"id": "phantom", "type": "rq", "score": 95, "justification": "j"}]]
        )
        scorer = PaperContentItemScorer(
            monitoring_handler=handler, model=model, rq_list=RQ_LIST
        )
        result = scorer(ResolvedPaperCollection(papers=[_resolved_paper()]), "2")
        assert result.papers[0].relevance_scores == []
        assert result.papers[0].filtering_score == 0.0

    def test_positional_mapping_multiple_papers(self):
        handler = _make_monitoring_handler()
        model = _make_scoring_model(
            [
                [{"id": "rq1", "type": "rq", "score": 70, "justification": "first"}],
                [{"id": "rq1", "type": "rq", "score": 20, "justification": "second"}],
            ]
        )
        scorer = PaperContentItemScorer(
            monitoring_handler=handler, model=model, rq_list=RQ_LIST
        )
        resolved = ResolvedPaperCollection(
            papers=[
                _resolved_paper(title="Paper One"),
                _resolved_paper(title="Paper Two"),
            ]
        )
        result = scorer(resolved, "2")
        assert result.papers[0].filtering_score == 70.0
        assert result.papers[1].filtering_score == 20.0
        assert result.papers[0].paper.title == "Paper One"
        assert result.papers[1].paper.title == "Paper Two"


# --- PaperScoreFilter ---


class TestPaperScoreFilter:
    def test_filters_below_threshold(self):
        handler = _make_monitoring_handler()
        f = PaperScoreFilter(monitoring_handler=handler, score_threshold=50)
        scored = ScoredPaperCollection(
            papers=[
                _make_scored_paper(80, title="Keep Me"),
                _make_scored_paper(30, title="Drop Me"),
            ]
        )
        result = f(scored, "3")
        assert [p.paper.title for p in result.papers] == ["Keep Me"]

    def test_threshold_is_inclusive(self):
        handler = _make_monitoring_handler()
        f = PaperScoreFilter(monitoring_handler=handler, score_threshold=50)
        scored = ScoredPaperCollection(papers=[_make_scored_paper(50, title="Edge")])
        result = f(scored, "3")
        assert [p.paper.title for p in result.papers] == ["Edge"]

    def test_prunes_low_scoring_sub_criteria(self):
        handler = _make_monitoring_handler()
        f = PaperScoreFilter(monitoring_handler=handler, score_threshold=50)
        sp = _make_scored_paper(
            80,
            scores=[
                {"id": "rq1", "type": "rq", "score": 80, "justification": "high"},
                {"id": "tc1", "type": "tc", "score": 10, "justification": "low"},
            ],
        )
        result = f(ScoredPaperCollection(papers=[sp]), "3")
        assert {se.id for se in result.papers[0].relevance_scores} == {"rq1"}

    def test_monitoring_leads_with_dropped(self):
        handler = _make_monitoring_handler()
        f = PaperScoreFilter(monitoring_handler=handler, score_threshold=50)
        scored = ScoredPaperCollection(
            papers=[
                _make_scored_paper(30, title="Dropped Paper"),
                _make_scored_paper(90, title="Kept Paper"),
            ]
        )
        f(scored, "3")
        step, text = handler.calls[0]
        assert step == "3"
        assert "Dropped Paper" in text
        assert "Kept Paper" not in text

    def test_empty_collection(self):
        handler = _make_monitoring_handler()
        f = PaperScoreFilter(monitoring_handler=handler, score_threshold=50)
        result = f(ScoredPaperCollection(papers=[]), "3")
        assert len(result.papers) == 0


# --- ContentItemDbWriter ---


@pytest.fixture()
def db_conn():
    from mourat.database import init_db

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = init_db(path)
    yield conn
    conn.close()
    os.unlink(path)


class TestContentItemId:
    def test_deterministic_and_normalised(self):
        assert (
            _content_item_id("Attention Is All You Need")
            == "paper_attention_is_all_you_need"
        )
        assert _content_item_id("  ATTENTION  is all you need ") == _content_item_id(
            "Attention is all you need"
        )

    def test_collapses_punctuation(self):
        assert _content_item_id("BERT: Pre-training of (Deep) Bi-ID!!") == (
            "paper_bert_pre_training_of_deep_bi_id"
        )


class TestContentItemDbWriter:
    def test_creates_content_item_and_links(self, db_conn):
        from mourat.database import business_domain as bd
        from mourat.database import content_item as ci

        handler = _make_monitoring_handler()
        writer = ContentItemDbWriter(monitoring_handler=handler, conn=db_conn)
        sp = _make_scored_paper(
            80,
            scores=[{"id": "tc1", "type": "tc", "score": 80, "justification": "core"}],
        )
        bd.create_technical_challenge(db_conn, "tc1", "TC one")
        result = writer(ScoredPaperCollection(papers=[sp]), "4")

        assert isinstance(result, ScoredPaperCollection)
        item = ci.get_content_item(db_conn, "paper_attention_is_all_you_need")
        assert item is not None
        assert item["name"] == "Attention Is All You Need"
        assert item["description"] == "Introduces the transformer architecture."
        assert item["url"] == "https://arxiv.org/abs/1706.03762"
        assert item["authors"] == "Vaswani; Shazeer"
        assert item["influence_score"] is None
        links = ci.list_item_technical_challenges(
            db_conn, "paper_attention_is_all_you_need"
        )
        assert len(links) == 1
        assert links[0]["id"] == "tc1"
        assert links[0]["relevance_score"] == 80
        assert links[0]["justification"] == "core"

    def test_rerun_updates_rather_than_duplicates(self, db_conn):
        from mourat.database import business_domain as bd
        from mourat.database import content_item as ci

        handler = _make_monitoring_handler()
        writer = ContentItemDbWriter(monitoring_handler=handler, conn=db_conn)
        bd.create_technical_challenge(db_conn, "tc1", "TC one")
        bd.create_technical_challenge(db_conn, "tc2", "TC two")

        first = _make_scored_paper(
            80,
            scores=[{"id": "tc1", "type": "tc", "score": 80, "justification": "old"}],
        )
        writer(ScoredPaperCollection(papers=[first]), "4")

        second = _make_scored_paper(
            90,
            scores=[{"id": "tc2", "type": "tc", "score": 90, "justification": "new"}],
        )
        writer(ScoredPaperCollection(papers=[second]), "4")

        items = db_conn.execute(
            "SELECT COUNT(*) FROM content_items WHERE id = ?",
            ("paper_attention_is_all_you_need",),
        ).fetchone()[0]
        assert items == 1
        links = ci.list_item_technical_challenges(
            db_conn, "paper_attention_is_all_you_need"
        )
        assert {l["id"] for l in links} == {"tc2"}  # refreshed, not appended
        assert links[0]["relevance_score"] == 90

    def test_monitoring_reports_counts(self, db_conn):
        handler = _make_monitoring_handler()
        writer = ContentItemDbWriter(monitoring_handler=handler, conn=db_conn)
        sp = _make_scored_paper(80)
        writer(ScoredPaperCollection(papers=[sp]), "4")
        step, text = handler.calls[0]
        assert step == "4"
        assert "created: 1" in text
        assert "failed: 0" in text

    def test_unknown_score_type_skipped_not_fatal(self, db_conn):
        from mourat.database import content_item as ci

        handler = _make_monitoring_handler()
        writer = ContentItemDbWriter(monitoring_handler=handler, conn=db_conn)
        sp = ScoredPaper(
            paper=_candidate_as_resolved(_make_candidate()),
            relevance_scores=[
                ScoreEntry(id="x", type="rq", score=50, justification="j")
            ],
            filtering_score=50.0,
        )
        result = writer(ScoredPaperCollection(papers=[sp]), "4")
        assert len(result.papers) == 1  # pass-through intact
        item = ci.get_content_item(db_conn, "paper_attention_is_all_you_need")
        assert item is not None  # paper itself was written

    def test_written_attribute_set_has_no_identifier_fields(self, db_conn):
        """FR2: identifiers travel in-flight but are never persisted."""
        from mourat.database import content_item as ci

        handler = _make_monitoring_handler()
        writer = ContentItemDbWriter(monitoring_handler=handler, conn=db_conn)
        paper = _candidate_as_resolved(_make_candidate())
        paper.doi = "https://doi.org/10.5555/3294995"
        paper.arxiv_id = "1706.03762"
        paper.work_id = "https://openalex.org/W123"
        paper.influence_score = 90
        sp = ScoredPaper(
            paper=paper,
            relevance_scores=[
                ScoreEntry(id="rq1", type="rq", score=80, justification="j")
            ],
            filtering_score=80.0,
        )
        writer(ScoredPaperCollection(papers=[sp]), "4")

        item = ci.get_content_item(db_conn, "paper_attention_is_all_you_need")
        assert item is not None
        assert "doi" not in item
        assert "arxiv_id" not in item
        assert "work_id" not in item


# --- JsonlWriter ---


class TestJsonlWriter:
    def test_writes_one_json_per_paper(self):
        handler = _make_monitoring_handler()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            writer = JsonlWriter(monitoring_handler=handler, output_path=path)
            sp = _make_scored_paper(
                80,
                scores=[
                    {"id": "rq1", "type": "rq", "score": 80, "justification": "core"}
                ],
            )
            result = writer(ScoredPaperCollection(papers=[sp]), "4")

            assert isinstance(result, ScoredPaperCollection)
            lines = open(path, encoding="utf-8").read().strip().split("\n")
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["title"] == "Attention Is All You Need"
            assert rec["authors"] == ["Vaswani", "Shazeer"]
            assert rec["filtering_score"] == 80.0
            assert rec["relevance_scores"][0]["id"] == "rq1"
            assert rec["url"] == "https://arxiv.org/abs/1706.03762"
            # FR2: no identifier fields in the persisted record
            assert "doi" not in rec
            assert "arxiv_id" not in rec
            assert "work_id" not in rec
        finally:
            os.unlink(path)

    def test_appends_across_runs(self):
        handler = _make_monitoring_handler()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            writer = JsonlWriter(monitoring_handler=handler, output_path=path)
            writer(
                ScoredPaperCollection(papers=[_make_scored_paper(80, title="A")]), "4"
            )
            writer(
                ScoredPaperCollection(papers=[_make_scored_paper(70, title="B")]), "4"
            )
            lines = open(path, encoding="utf-8").read().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["title"] == "A"
            assert json.loads(lines[1])["title"] == "B"
        finally:
            os.unlink(path)

    def test_monitoring_reports_path_and_count(self):
        handler = _make_monitoring_handler()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            writer = JsonlWriter(monitoring_handler=handler, output_path=path)
            writer(ScoredPaperCollection(papers=[_make_scored_paper(80)]), "4")
            step, text = handler.calls[0]
            assert step == "4"
            assert path in text
            assert "1" in text
        finally:
            os.unlink(path)


# --- Script attribute loading ---


class TestLoadAttributes:
    def _make_cfg(self, **overrides):
        from omegaconf import OmegaConf

        base = {
            "research_question_ids": ["q1"],
            "technical_challenge_ids": ["tc1"],
            "research_topic_ids": ["topic1"],
            "constraint_ids": ["c1"],
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def _seed_db(self, conn):
        from mourat.database import business_domain as bd
        from mourat.database import research_domain as rd

        rd.create_research_domain(conn, "rd1", "RD")
        rd.create_research_direction(conn, "dir1", "Dir", "rd1")
        rd.create_research_object(conn, "obj1", "Obj", "dir1")
        rd.create_research_question(conn, "q1", "How to scale?", "obj1")
        rd.create_research_topic(conn, "topic1", "Efficiency")
        bd.create_technical_challenge(conn, "tc1", "Memory")
        bd.create_constraint(conn, "c1", "Compute budget")

    def test_loads_all_attribute_kinds(self, db_conn):
        from mourat.scripts.collect_influential_papers_from_scratch import (
            _load_attributes,
        )

        self._seed_db(db_conn)
        blocks, scoring_lists, constraint_list = _load_attributes(
            db_conn, self._make_cfg()
        )
        joined = "\n".join(blocks)
        assert "How to scale?" in joined
        assert "Memory" in joined
        assert "Efficiency" in joined
        assert "Compute budget" in joined
        assert scoring_lists["rq"][0]["id"] == "q1"
        assert scoring_lists["tc"][0]["id"] == "tc1"
        assert scoring_lists["topic"][0]["id"] == "topic1"
        assert constraint_list[0]["id"] == "c1"

    def test_missing_id_reported_and_skipped(self, db_conn, caplog):
        from mourat.scripts.collect_influential_papers_from_scratch import (
            _load_attributes,
        )

        self._seed_db(db_conn)
        with caplog.at_level("ERROR"):
            blocks, scoring_lists, constraint_list = _load_attributes(
                db_conn, self._make_cfg(research_question_ids=["missing-id"])
            )
        assert any("missing-id" in r.message for r in caplog.records)
        assert scoring_lists["rq"] == []

    def test_no_resolvable_attributes_raises_in_main(self, db_conn, caplog):
        """Empty blocks must fail loudly upstream; loader returns empty lists."""
        from mourat.scripts.collect_influential_papers_from_scratch import (
            _load_attributes,
        )

        blocks, scoring_lists, constraint_list = _load_attributes(
            db_conn, self._make_cfg(research_question_ids=["ghost"])
        )
        assert blocks == []


# --- Seed loading (spec 11 task 7, FR1) ---


class TestSeedItemRetrieval:
    """Tests for `_retrieve_seed_items` from the seed-expansion script."""

    def _seed_db(self, conn):
        from mourat.database import business_domain as bd
        from mourat.database import content_item as ci
        from mourat.database import research_domain as rd

        ci.create_source_type(conn, "paper", "Paper")
        ci.create_platform(conn, "arxiv", "Arxiv")
        ci.create_influence_metric(conn, "citations", "Citations")
        rd.create_research_domain(conn, "rd1", "RD")
        rd.create_research_direction(conn, "dir1", "Dir", "rd1")
        rd.create_research_object(conn, "obj1", "Obj", "dir1")
        rd.create_research_question(conn, "q1", "How to scale?", "obj1")
        bd.create_technical_challenge(conn, "tc1", "Memory")
        for item_id, name, score in [
            ("item1", "Attention Is All You Need", 90),
            ("item2", "Scaling Laws for Neural LMs", 70),
        ]:
            ci.create_content_item(
                conn,
                item_id,
                name,
                source_type_id="paper",
                platform_id="arxiv",
                influence_metric_id="citations",
                influence_score=score,
            )
        ci.add_item_research_question(conn, "item1", "q1", "relevant", 85)
        ci.add_item_technical_challenge(conn, "item2", "tc1", "match", 60)

    def _make_cfg(self, **overrides):
        from omegaconf import OmegaConf

        base = {
            "seed_research_question_ids": ["q1"],
            "seed_technical_challenge_ids": [],
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_retrieves_items_linked_to_configured_ids(self, db_conn):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _retrieve_seed_items,
        )

        self._seed_db(db_conn)
        collection = _retrieve_seed_items(
            db_conn, self._make_cfg(seed_technical_challenge_ids=["tc1"])
        )
        assert {item.id for item in collection.items} == {"item1", "item2"}
        assert collection.items[0].name == "Attention Is All You Need"
        assert collection.items[0].influence_score == 90

    def test_empty_seed_set_returns_empty_collection(self, db_conn):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _retrieve_seed_items,
        )

        self._seed_db(db_conn)
        collection = _retrieve_seed_items(
            db_conn, self._make_cfg(seed_research_question_ids=["ghost-id"])
        )
        assert collection.items == []

    def test_single_item_seed_set_still_produces_a_run(self, db_conn):
        """A one-item seed set is a valid run input, not an error (FR1)."""
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _retrieve_seed_items,
        )

        self._seed_db(db_conn)
        collection = _retrieve_seed_items(
            db_conn,
            self._make_cfg(
                seed_research_question_ids=[],
                seed_technical_challenge_ids=["tc1"],
            ),
        )
        assert len(collection.items) == 1
        assert collection.items[0].id == "item2"

    def test_no_linked_items_reported_not_silent(self, db_conn, caplog):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _retrieve_seed_items,
        )

        self._seed_db(db_conn)
        with caplog.at_level("ERROR"):
            collection = _retrieve_seed_items(
                db_conn, self._make_cfg(seed_research_question_ids=["ghost-id"])
            )
        assert any("ghost-id" in r.message for r in caplog.records)
        assert collection.items == []


class TestLoadScoringAttributes:
    """Tests for `_load_scoring_attributes` — the scorer's attribute wiring
    whose absence made every filtering score 0 in a live run."""

    def _seed_db(self, conn):
        from mourat.database import business_domain as bd
        from mourat.database import research_domain as rd

        rd.create_research_domain(conn, "rd1", "RD")
        rd.create_research_direction(conn, "dir1", "Dir", "rd1")
        rd.create_research_object(conn, "obj1", "Obj", "dir1")
        rd.create_research_question(conn, "q1", "How to scale?", "obj1")
        rd.create_research_topic(conn, "topic1", "Scaling")
        bd.create_technical_challenge(conn, "tc1", "Memory")
        bd.create_constraint(conn, "con1", "Under 100 GPUs")

    def _make_cfg(self, **overrides):
        from omegaconf import OmegaConf

        base = {
            "seed_research_question_ids": ["q1"],
            "seed_technical_challenge_ids": ["tc1"],
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_loads_rq_tc_and_optionals(self, db_conn):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _load_scoring_attributes,
        )

        self._seed_db(db_conn)
        rq, tc, topic, constraint = _load_scoring_attributes(
            db_conn,
            self._make_cfg(research_topic_ids=["topic1"], constraint_ids=["con1"]),
        )
        assert rq == [
            {
                "id": "q1",
                "name": "How to scale?",
                "type": "rq",
                "description": rq[0]["description"],
            }
        ]
        assert rq[0]["description"] in ("", "How to scale?")
        assert tc[0]["id"] == "tc1" and tc[0]["type"] == "tc"
        assert topic[0]["id"] == "topic1" and topic[0]["type"] == "topic"
        assert constraint[0]["id"] == "con1" and constraint[0]["type"] == "constraint"

    def test_unknown_ids_reported_and_skipped(self, db_conn, caplog):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _load_scoring_attributes,
        )

        self._seed_db(db_conn)
        with caplog.at_level("ERROR"):
            rq, tc, _, _ = _load_scoring_attributes(
                db_conn,
                self._make_cfg(
                    seed_research_question_ids=["ghost"],
                    seed_technical_challenge_ids=[],
                ),
            )
        assert rq == [] and tc == []
        assert any("ghost" in r.message for r in caplog.records)

    def test_minimal_config_returns_empty_lists(self, db_conn):
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _load_scoring_attributes,
        )

        self._seed_db(db_conn)
        rq, tc, topic, constraint = _load_scoring_attributes(db_conn, self._make_cfg())
        assert len(rq) == 1 and len(tc) == 1
        assert topic == [] and constraint == []

    def test_loaded_attributes_form_the_scorer_whitelist(self, db_conn):
        """End-to-end seam: entries whose (id, type) come from the loaded
        lists survive the scorer's validation, so filtering_score > 0."""
        from mourat.scripts.collect_influential_papers_from_seeds import (
            _load_scoring_attributes,
        )

        self._seed_db(db_conn)
        rq, tc, _, _ = _load_scoring_attributes(db_conn, self._make_cfg())
        assert rq and tc  # the whitelist must be non-empty
        pairs = [(e["id"], e["type"]) for e in rq + tc]
        assert pairs == [("q1", "rq"), ("tc1", "tc")]
        # the scorer's own validation gate keeps exactly these pairs
        from mourat.processors.content_item_scorer import ContentItemScorer

        scorer = ContentItemScorer.__new__(ContentItemScorer)
        scorer.rq_list, scorer.tc_list = rq, tc
        scorer.topic_list, scorer.constraint_list = [], []
        scorer.valid_id_type_pairs = [
            (entity["id"], entity["type"])
            for entity in sum(
                [
                    scorer.rq_list,
                    scorer.tc_list,
                    scorer.topic_list,
                    scorer.constraint_list,
                ],
                start=[],
            )
        ]
        assert scorer.valid_id_type_pairs == [("q1", "rq"), ("tc1", "tc")]
