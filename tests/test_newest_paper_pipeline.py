"""Tests for the newest-papers pipeline (spec 14): triage classifier,
affiliation fetcher, authority assessor, converter, script wiring.

Conventions (same as the rest of the suite): no test performs a real
network request — HTTP clients are mocked; LLM responses are scripted via
FunctionModel returning ModelResponse(parts=[TextPart(json)]); retry
timing is asserted with mocked time.sleep.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from mourat.collectors.arxiv_html_affiliations import (
    ArxivHtmlAffiliationFetcher,
    _extract_affiliations,
)
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler
from mourat.processors.authority_influence import AuthorityInfluenceAssessor
from mourat.processors.candidate_converter import CandidateToResolvedConverter
from mourat.processors.relevance_triage import RelevanceTriageClassifier

# --- Helpers ---


def _make_monitoring_handler():
    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []

        def __call__(self, step: str, text_for_monitoring: str) -> None:
            self.calls.append((step, text_for_monitoring))

    return CapturingHandler()


def _make_collection(n=3, with_ids=True) -> PaperCandidateCollection:
    papers = []
    for i in range(n):
        papers.append(
            PaperCandidate(
                title=f"Paper {chr(65 + i)}",
                authors=[f"Author {chr(65 + i)}"],
                description=f"Abstract of paper {chr(65 + i)}",
                urls_seen=[f"https://arxiv.org/abs/2401.0000{i}"] if with_ids else [],
                arxiv_id=f"2401.0000{i}" if with_ids else None,
                publication_date="2024-01-01",
            )
        )
    return PaperCandidateCollection(papers=papers)


def _ids_in_prompt(prompt: str) -> list[str]:
    """Paper ids the scripted model fn sees, extracted from the prompt."""
    import re

    return re.findall(r'"id": "(paper_\d+)"', prompt)


def _verdicts_json(ids: list[str], score=None, relevant=None, skip=None):
    """Build a TriageResult/AuthorityResult JSON for the given ids."""
    verdicts = []
    for pid in ids:
        if skip is not None and pid in skip:
            continue
        if score is not None:
            verdicts.append(
                {"id": pid, "score": score(pid) if callable(score) else score}
            )
        else:
            verdicts.append(
                {
                    "id": pid,
                    "relevant": relevant(pid) if callable(relevant) else relevant,
                }
            )
    return json.dumps({"verdicts": verdicts})


# --- ArxivHtmlAffiliationFetcher (R3: T4, T5) ---

AFFIL_HTML = """
<span class="ltx_personname">Alice Smith</span>
<span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">Affiliation: </span>University of Somewhere</span>
<span class="ltx_personname">Bob Jones</span>
<span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">Affiliation: </span>Famous Lab</span>
<span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">Affiliation: </span>Other Institute</span>
"""

NO_AFFIL_HTML = "<html><body>No affiliation markup here</body></html>"


class TestExtractAffiliations:
    def test_maps_authors_to_affiliations(self):
        # T4: each author gets their affiliation; Bob's two spans accumulate
        affs = _extract_affiliations(AFFIL_HTML)
        assert affs["Alice Smith"] == ["University of Somewhere"]
        assert affs["Bob Jones"] == ["Famous Lab", "Other Institute"]

    def test_no_nodes_gives_empty(self):
        assert _extract_affiliations(NO_AFFIL_HTML) == {}


class TestArxivHtmlAffiliationFetcher:
    def _fetcher(self, responses: dict[str, object]):
        client = MagicMock()

        def get(url):
            r = MagicMock()
            result = responses.get(url, Exception("404"))
            if isinstance(result, Exception):
                raise result
            r.text = result
            r.raise_for_status = lambda: None
            return r

        client.get.side_effect = get
        return ArxivHtmlAffiliationFetcher(
            monitoring_handler=_make_monitoring_handler(),
            http_client=client,
            request_delay_seconds=0.0,
        )

    def test_attaches_affiliations(self):
        # T4
        url = "https://arxiv.org/html/2401.00000v1"
        fetcher = self._fetcher({url: AFFIL_HTML})
        coll = _make_collection(1)
        result = fetcher(coll, "t")
        assert result.papers[0].affiliations == {
            "Alice Smith": ["University of Somewhere"],
            "Bob Jones": ["Famous Lab", "Other Institute"],
        }

    def test_404_gives_unknown_not_exception(self):
        # T5a
        fetcher = self._fetcher({})
        coll = _make_collection(1)
        result = fetcher(coll, "t")
        assert result.papers[0].affiliations is None

    def test_render_without_affiliation_nodes_gives_unknown(self):
        # T5b
        url = "https://arxiv.org/html/2401.00000v1"
        fetcher = self._fetcher({url: NO_AFFIL_HTML})
        coll = _make_collection(1)
        result = fetcher(coll, "t")
        assert result.papers[0].affiliations is None

    def test_candidate_without_arxiv_id_skipped(self):
        fetcher = self._fetcher({})
        coll = _make_collection(1, with_ids=False)
        result = fetcher(coll, "t")
        assert result.papers[0].affiliations is None


# --- RelevanceTriageClassifier (R9: T16-T21) ---


ATTRS = {
    "rq_list": [
        {"id": "rq1", "name": "RQ one", "type": "rq", "description": "kv cache"}
    ],
    "tc_list": [],
    "topic_list": [],
}


class TestRelevanceTriageClassifier:
    def _classifier(self, model_fn, **kwargs):
        return RelevanceTriageClassifier(
            monitoring_handler=_make_monitoring_handler(),
            model=FunctionModel(model_fn),
            **{**ATTRS, **kwargs},
        )

    def test_verdicts_matched_and_batch_splitting(self):
        # T16: verdicts matched by id; batch size forces 2 calls
        calls = []

        def model_fn(messages, agent):
            prompt = messages[-1].parts[-1].content
            ids = _ids_in_prompt(prompt)
            calls.append(list(ids))
            return ModelResponse(parts=[TextPart(_verdicts_json(ids, relevant=True))])

        clf = self._classifier(model_fn, batch_size=2)
        coll = _make_collection(3)
        result = clf(coll, "t")
        assert len(calls) == 2  # batch splitting
        assert all(p.title in {f"Paper {c}" for c in "ABC"} for p in result.papers)
        assert len(result.papers) == 3  # all true -> all kept

    def test_false_drops_paper(self):
        def model_fn(messages, agent):
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            relevant = lambda pid: pid != "paper_1"  # noqa: E731
            return ModelResponse(
                parts=[TextPart(_verdicts_json(ids, relevant=relevant))]
            )

        handler = _make_monitoring_handler()
        clf = RelevanceTriageClassifier(
            monitoring_handler=handler,
            model=FunctionModel(model_fn),
            **ATTRS,
        )
        result = clf(_make_collection(3), "t")
        monitoring = handler.calls[-1][1]
        assert [p.title for p in result.papers] == ["Paper A", "Paper C"]
        assert "Paper B" in monitoring  # dropped titles listed

    @patch("mourat.processors.relevance_triage.time.sleep")
    def test_transport_failure_keeps_everything(self, mock_sleep):
        # T17: model raises on every attempt -> all kept, warning logged
        def model_fn(messages, agent):
            raise RuntimeError("caila down")

        clf = self._classifier(model_fn, request_retries=2)
        with patch("mourat.processors.relevance_triage.logger.warning") as warn:
            result = clf(_make_collection(3), "t")
        assert len(result.papers) == 3  # fail-open
        assert mock_sleep.call_count == 2  # bounded retries
        assert warn.called

    @patch("mourat.processors.relevance_triage.time.sleep")
    def test_malformed_json_keeps_everything(self, mock_sleep):
        # T18: malformed JSON on every attempt -> keep-everything behaviour
        def model_fn(messages, agent):
            return ModelResponse(parts=[TextPart("not json at all {")])

        clf = self._classifier(model_fn, request_retries=1)
        result = clf(_make_collection(2), "t")
        assert len(result.papers) == 2

    def test_missing_verdict_keeps_paper(self):
        # T19: one verdict omitted -> that paper kept, others matched
        def model_fn(messages, agent):
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            skipped = ids[-1:]
            return ModelResponse(
                parts=[TextPart(_verdicts_json(ids, relevant=True, skip=skipped))]
            )

        clf = self._classifier(model_fn)
        result = clf(_make_collection(2), "t")
        assert len(result.papers) == 2  # fail-open

    def test_hallucinated_id_discarded(self):
        # T20: verdict for an id not in the batch -> discarded
        def model_fn(messages, agent):
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            extra = [{"id": "paper_999", "relevant": False}]
            verdicts = [{"id": pid, "relevant": True} for pid in ids] + extra
            return ModelResponse(parts=[TextPart(json.dumps({"verdicts": verdicts}))])

        clf = self._classifier(model_fn)
        result = clf(_make_collection(2), "t")
        assert len(result.papers) == 2  # bogus verdict affected nobody

    def test_constraints_excluded_from_prompt(self):
        # T21: constraint text never reaches the classifier prompt
        captured = []

        def model_fn(messages, agent):
            captured.append(messages[-1].parts[-1].content)
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            return ModelResponse(parts=[TextPart(_verdicts_json(ids, relevant=True))])

        clf = RelevanceTriageClassifier(
            monitoring_handler=_make_monitoring_handler(),
            model=FunctionModel(model_fn),
            rq_list=ATTRS["rq_list"],
            tc_list=[],
            topic_list=[],
        )
        # constraints are not even a constructor parameter — assert the
        # prompt built carries only the configured attribute kinds
        clf(_make_collection(1), "t")
        prompt = captured[0]
        assert "Research questions" in prompt
        assert "constraint" not in prompt.lower().replace("constraints excluded", "")

    def test_no_attributes_configured_keeps_all(self):
        clf = RelevanceTriageClassifier(
            monitoring_handler=_make_monitoring_handler(),
            model=FunctionModel(
                lambda m: (_ for _ in ()).throw(AssertionError("no LLM call expected"))
            ),
            rq_list=[],
            tc_list=[],
            topic_list=[],
        )
        result = clf(_make_collection(2), "t")
        assert len(result.papers) == 2


# --- AuthorityInfluenceAssessor (R4: T6-T12) ---


class TestAuthorityInfluenceAssessor:
    def _assessor(self, model_fn, **kwargs):
        return AuthorityInfluenceAssessor(
            monitoring_handler=_make_monitoring_handler(),
            model=FunctionModel(model_fn),
            **kwargs,
        )

    def test_scores_matched_by_id_and_batch_splitting(self):
        # T6
        calls = []

        def model_fn(messages, agent):
            prompt = messages[-1].parts[-1].content
            ids = _ids_in_prompt(prompt)
            calls.append(list(ids))
            return ModelResponse(parts=[TextPart(_verdicts_json(ids, score=42))])

        assessor = self._assessor(model_fn, batch_size=2)
        coll = _make_collection(3)
        result = assessor(coll, "t")
        assert len(calls) == 2
        assert all(p.influence_score == 42 for p in result.papers)

    @patch("mourat.processors.authority_influence.time.sleep")
    def test_transport_failure_zero_scores(self, mock_sleep):
        # T7: exhausted retries -> every paper score 0, warnings logged
        def model_fn(messages, agent):
            raise RuntimeError("caila down")

        assessor = self._assessor(model_fn, request_retries=2)
        with patch("mourat.processors.authority_influence.logger.warning") as warn:
            result = assessor(_make_collection(3), "t")
        assert all(p.influence_score == 0 for p in result.papers)
        assert mock_sleep.call_count == 2
        assert warn.called

    def test_malformed_json_zero_scores(self):
        # T8
        def model_fn(messages, agent):
            return ModelResponse(parts=[TextPart("{broken json")])

        handler = _make_monitoring_handler()
        assessor = AuthorityInfluenceAssessor(
            monitoring_handler=handler,
            model=FunctionModel(model_fn),
            request_retries=1,
        )
        result = assessor(_make_collection(2), "t")
        assert all(p.influence_score == 0 for p in result.papers)
        assert "Zero-scored papers" in handler.calls[-1][1]

    def test_missing_verdict_triggers_one_follow_up(self):
        # T9: one omitted verdict -> exactly one follow-up call with that paper
        calls = []

        def model_fn(messages, agent):
            prompt = messages[-1].parts[-1].content
            ids = _ids_in_prompt(prompt)
            calls.append(list(ids))
            if len(ids) == 3 and calls.count(ids) == 1:
                # first batch call: omit the last verdict
                return ModelResponse(
                    parts=[TextPart(_verdicts_json(ids, score=50, skip=[ids[-1]]))]
                )
            return ModelResponse(parts=[TextPart(_verdicts_json(ids, score=77))])

        assessor = self._assessor(model_fn, batch_size=25)
        result = assessor(_make_collection(3), "t")
        assert len(calls) == 2  # main batch + one follow-up
        assert len(calls[1]) == 1  # follow-up contains only the missing paper
        assert result.papers[0].influence_score == 50
        assert result.papers[1].influence_score == 50
        assert result.papers[2].influence_score == 77

    @patch("mourat.processors.authority_influence.time.sleep")
    def test_failed_follow_up_zero_score(self, mock_sleep):
        # T10: follow-up also fails -> that paper 0, others unaffected
        calls = []

        def model_fn(messages, agent):
            prompt = messages[-1].parts[-1].content
            ids = _ids_in_prompt(prompt)
            calls.append(list(ids))
            if len(ids) == 3:
                return ModelResponse(
                    parts=[TextPart(_verdicts_json(ids, score=50, skip=[ids[-1]]))]
                )
            raise RuntimeError("follow-up fails")

        assessor = self._assessor(model_fn)
        result = assessor(_make_collection(3), "t")
        assert result.papers[0].influence_score == 50
        assert result.papers[1].influence_score == 50
        assert result.papers[2].influence_score == 0

    def test_hallucinated_id_discarded(self):
        # T11
        def model_fn(messages, agent):
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            verdicts = [{"id": pid, "score": 30} for pid in ids]
            verdicts.append({"id": "paper_999", "score": 99})
            return ModelResponse(parts=[TextPart(json.dumps({"verdicts": verdicts}))])

        assessor = self._assessor(model_fn)
        result = assessor(_make_collection(2), "t")
        assert all(p.influence_score == 30 for p in result.papers)

    def test_unmeasurable_scores_zero(self):
        # T12: "LLM recognises nobody" -> 0, not None
        def model_fn(messages, agent):
            ids = _ids_in_prompt(messages[-1].parts[-1].content)
            return ModelResponse(parts=[TextPart(_verdicts_json(ids, score=0))])

        assessor = self._assessor(model_fn)
        result = assessor(_make_collection(2), "t")
        for p in result.papers:
            assert p.influence_score == 0
            assert p.influence_score is not None


# --- CandidateToResolvedConverter (option B) ---


class TestCandidateToResolvedConverter:
    def test_maps_fields(self):
        handler = _make_monitoring_handler()
        conv = CandidateToResolvedConverter(handler)
        coll = _make_collection(2)
        result = conv(coll, "t")
        assert len(result.papers) == 2
        first = result.papers[0]
        assert first.title == "Paper A"
        assert first.abstract == "Abstract of paper A"  # description -> abstract
        assert first.url == "https://arxiv.org/abs/2401.00000"
        assert first.publication_date == "2024-01-01"
        assert first.arxiv_id == "2401.00000"  # in-flight, never persisted
        assert first.influence_score is None  # assessor fills it before this stage
        assert "Converted 2" in handler.calls[-1][1]


# --- Script wiring (T13, T15) ---
