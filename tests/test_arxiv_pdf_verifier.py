"""Unit tests for ArxivPdfVerifier (FR4): arXiv title-search verification."""

from unittest.mock import MagicMock

from mourat.data_models import ResolvedPaper, ResolvedPaperCollection
from mourat.monitoring import MonitoringHandler
from mourat.processors.arxiv_pdf_verifier import ArxivPdfVerifier


def _make_handler():
    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []

        def __call__(self, step, text_for_monitoring):
            self.calls.append((step, text_for_monitoring))

    return CapturingHandler()


def _make_verifier(arxiv=None) -> ArxivPdfVerifier:
    if arxiv is None:
        arxiv = MagicMock()
        arxiv.search_by_title.return_value = None
    return ArxivPdfVerifier(
        monitoring_handler=_make_handler(),
        arxiv_client=arxiv,
        title_similarity_threshold=0.9,
    )


def _make_paper(**overrides) -> ResolvedPaper:
    defaults = {
        "title": "Attention Is All You Need",
    }
    defaults.update(overrides)
    return ResolvedPaper(**defaults)


def _verify_one_paper(verifier: ArxivPdfVerifier, paper: ResolvedPaper):
    result, monitoring = verifier._run(ResolvedPaperCollection(papers=[paper]))
    return result.papers[0], monitoring


class TestUrlRecorded:
    def test_title_search_hit_verifies(self):
        arxiv = MagicMock()
        arxiv.search_by_title.return_value = (
            "1706.03762",
            "Attention Is All You Need",
        )
        paper, _ = _verify_one_paper(_make_verifier(arxiv=arxiv), _make_paper())
        assert paper.url == "https://arxiv.org/pdf/1706.03762"
        assert paper.url_absent_reason is None
        arxiv.search_by_title.assert_called_once_with("Attention Is All You Need")

    def test_whitespace_mangled_arxiv_title_verifies(self):
        # Atom titles arrive whitespace-collapsed from the client; a
        # verbatim match must verify.
        arxiv = MagicMock()
        arxiv.search_by_title.return_value = (
            "1706.03762",
            "Attention Is All You Need",
        )
        paper, _ = _verify_one_paper(
            _make_verifier(arxiv=arxiv),
            _make_paper(title="Attention Is All You Need"),
        )
        assert paper.url == "https://arxiv.org/pdf/1706.03762"


class TestUrlLeftEmpty:
    def test_not_on_arxiv_leaves_url_empty(self):
        # Empty search feed: the paper has no arXiv preprint.
        arxiv = MagicMock()
        arxiv.search_by_title.return_value = None
        paper, monitoring = _verify_one_paper(
            _make_verifier(arxiv=arxiv), _make_paper()
        )
        assert paper.url == ""
        assert paper.url_absent_reason == "not_on_arxiv"
        assert "not_on_arxiv" in monitoring

    def test_title_mismatch_leaves_url_empty(self):
        # arXiv's top hit for the quoted title names a different paper.
        arxiv = MagicMock()
        arxiv.search_by_title.return_value = (
            "2401.99999",
            "A Completely Different Work",
        )
        paper, monitoring = _verify_one_paper(
            _make_verifier(arxiv=arxiv), _make_paper()
        )
        assert paper.url == ""
        assert paper.url_absent_reason == "arxiv_title_mismatch"
        assert "arxiv_title_mismatch" in monitoring


class TestApiErrorContainment:
    def test_request_exception_becomes_api_error_not_a_crash(self):
        # The regression from the live run: a ReadTimeout after retries
        # must record api_error for that paper, never abort the step.
        arxiv = MagicMock()
        arxiv.search_by_title.side_effect = TimeoutError("read timed out")
        papers = [_make_paper(title="One"), _make_paper(title="Two")]
        verifier = _make_verifier(arxiv=arxiv)
        result, monitoring = verifier._run(ResolvedPaperCollection(papers=papers))
        assert len(result.papers) == 2
        assert all(p.url == "" for p in result.papers)
        assert all(p.url_absent_reason == "api_error" for p in result.papers)
        assert "api_error" in monitoring

    def test_one_failing_paper_does_not_block_the_rest(self):
        arxiv = MagicMock()
        arxiv.search_by_title.side_effect = [
            TimeoutError("read timed out"),
            ("1706.03762", "Attention Is All You Need"),
        ]
        verifier = _make_verifier(arxiv=arxiv)
        result, monitoring = verifier._run(
            ResolvedPaperCollection(papers=[_make_paper(title="Doomed"), _make_paper()])
        )
        assert result.papers[0].url_absent_reason == "api_error"
        assert result.papers[1].url == "https://arxiv.org/pdf/1706.03762"
        assert "verified: 1" in monitoring


class TestRunBehaviour:
    def test_mixed_batch_reports_counts_and_reasons(self):
        arxiv = MagicMock()
        arxiv.search_by_title.side_effect = [
            ("1706.03762", "Attention Is All You Need"),
            None,  # no preprint
            ("2401.99999", "A Completely Different Work"),  # mismatch
        ]
        verifier = _make_verifier(arxiv=arxiv)
        data = ResolvedPaperCollection(
            papers=[
                _make_paper(),
                _make_paper(title="Journal Only Paper"),
                _make_paper(title="Near Miss Title"),
            ]
        )
        result, monitoring = verifier._run(data)
        assert result.papers[0].url.startswith("https://arxiv.org/pdf/")
        assert result.papers[1].url_absent_reason == "not_on_arxiv"
        assert result.papers[2].url_absent_reason == "arxiv_title_mismatch"
        first_line = monitoring.split("\n")[0]
        assert "Papers in: 3, verified: 1, url absent: 2" in first_line
        assert "'not_on_arxiv': 1" in first_line
        assert "'arxiv_title_mismatch': 1" in first_line

    def test_every_paper_failing_verification_completes(self):
        verifier = _make_verifier()
        data = ResolvedPaperCollection(
            papers=[
                _make_paper(title="One"),
                _make_paper(title="Two"),
            ]
        )
        result, monitoring = verifier._run(data)
        assert len(result.papers) == 2
        assert all(p.url == "" for p in result.papers)
        assert "verified: 0" in monitoring
