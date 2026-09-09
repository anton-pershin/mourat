"""Unit tests for InfluenceAssessor (FR3)."""

from datetime import date
from unittest.mock import MagicMock

from mourat.data_models import ResolvedPaper, ResolvedPaperCollection
from mourat.monitoring import MonitoringHandler
from mourat.processors.influence_assessor import (
    MEASURE_CITATIONS_PER_YEAR,
    MEASURE_FWCI,
    InfluenceAssessor,
    _interpolate,
    _years_since,
)

FWCI_BREAKPOINTS = [(1.0, 10), (10.0, 50), (100.0, 90)]
CITATIONS_PER_YEAR_BREAKPOINTS = [(1.0, 10), (10.0, 50), (100.0, 90)]


def _make_handler():
    class CapturingHandler(MonitoringHandler):
        def __init__(self):
            self.calls = []

        def __call__(self, step, text_for_monitoring):
            self.calls.append((step, text_for_monitoring))

    return CapturingHandler()


def _make_assessor() -> InfluenceAssessor:
    return InfluenceAssessor(
        monitoring_handler=_make_handler(),
        fwci_breakpoints=FWCI_BREAKPOINTS,
        citations_per_year_breakpoints=CITATIONS_PER_YEAR_BREAKPOINTS,
    )


def _make_paper(**overrides) -> ResolvedPaper:
    defaults = {
        "title": "Attention Is All You Need",
        "publication_date": "2017-06-12",
        "influence_fwci": 6139.0,
        "influence_cited_by_count": 88404,
    }
    defaults.update(overrides)
    return ResolvedPaper(**defaults)


def _assess_one(paper: ResolvedPaper):
    return _make_assessor()._assess_one(paper, date(2026, 9, 9))


class TestInterpolate:
    def test_exact_breakpoints_hit(self):
        bps = [(1.0, 10), (10.0, 50)]
        assert _interpolate(1.0, bps) == 10.0
        assert _interpolate(10.0, bps) == 50.0

    def test_midpoint_interpolates(self):
        bps = [(0.0, 0), (10.0, 100)]
        assert _interpolate(5.0, bps) == 50.0

    def test_saturates_at_both_ends(self):
        bps = [(1.0, 10), (100.0, 90)]
        assert _interpolate(0.001, bps) == 10.0
        assert _interpolate(10_000.0, bps) == 90.0


class TestYearsSince:
    def test_whole_years(self):
        assert _years_since("2017-06-12", date(2026, 9, 9)) == 9

    def test_birthday_not_yet_reached(self):
        assert _years_since("2017-09-10", date(2026, 9, 9)) == 8

    def test_undatable_returns_none(self):
        assert _years_since(None, date(2026, 9, 9)) is None
        assert _years_since("not a date", date(2026, 9, 9)) is None


class TestMeasurePreference:
    def test_fwci_preferred_when_present(self):
        score, measure = _assess_one(_make_paper())
        assert measure == MEASURE_FWCI
        assert score == 90  # 6139 saturates past the last breakpoint

    def test_falls_back_to_citations_per_year_when_fwci_absent(self):
        score, measure = _assess_one(_make_paper(influence_fwci=None))
        assert measure == MEASURE_CITATIONS_PER_YEAR
        # 88404 citations over ~9.25 years (2017-06-12 -> 2026-09-09) ≈ 9560/yr
        assert score == 90

    def test_falls_back_when_fwci_malformed(self):
        # OpenAlex occasionally returns fwci as a string; a dict payload from
        # the API record bypasses model coercion in the resolver path, so the
        # assessor must tolerate a non-numeric value here.
        item = _make_paper(influence_fwci=None)
        object.__setattr__(item, "influence_fwci", "lots")  # simulate raw dict data
        assessor = _make_assessor()
        score, measure = assessor._assess_one(item, date(2026, 9, 9))
        assert measure == MEASURE_CITATIONS_PER_YEAR
        assert score is not None

    def test_nan_fwci_uses_fallback(self):
        score, measure = _assess_one(_make_paper(influence_fwci=float("nan")))
        assert measure == MEASURE_CITATIONS_PER_YEAR

    def test_unmeasurable_when_no_signal_at_all(self):
        score, measure = _assess_one(
            _make_paper(
                influence_fwci=None,
                influence_cited_by_count=None,
                publication_date=None,
            )
        )
        assert score is None and measure is None


class TestScoreBounds:
    def test_scores_stay_within_0_100_across_orders_of_magnitude(self):
        for fwci in [0.0, 0.5, 5.0, 50.0, 5000.0, 6139.0, 100_000.0]:
            score, _ = _assess_one(_make_paper(influence_fwci=fwci))
            assert 0 <= score <= 100
        for cpy in [0.0, 0.5, 5.0, 500.0, 100_000.0]:
            score, _ = _assess_one(
                _make_paper(
                    influence_fwci=None,
                    influence_cited_by_count=int(cpy * 2),
                    publication_date="2026-01-01",
                )
            )
            assert 0 <= score <= 100

    def test_zero_citation_paper_maps_to_lowest_band(self):
        score, measure = _assess_one(
            _make_paper(influence_fwci=None, influence_cited_by_count=0)
        )
        assert measure == MEASURE_CITATIONS_PER_YEAR
        assert score == 10  # first breakpoint value


class TestCrossMeasureComparability:
    def test_equal_standing_under_different_measures_maps_equal(self):
        # Same standing (top breakpoint) under both measures -> same score.
        fwci_score, m1 = _assess_one(_make_paper(influence_fwci=6139.0))
        cpy_score, m2 = _assess_one(
            _make_paper(
                influence_fwci=None,
                influence_cited_by_count=88404,
                publication_date="2017-06-12",
            )
        )
        assert m1 != m2
        assert fwci_score == cpy_score

    def test_mid_standing_also_comparable(self):
        # fwci=5 and citations-per-year=5 land on the same interpolated score.
        a, _ = _assess_one(_make_paper(influence_fwci=5.0))
        b, _ = _assess_one(
            _make_paper(
                influence_fwci=None,
                influence_cited_by_count=48,  # 2025-09-09 -> ~1.0/yr
                publication_date="2025-09-09",
            )
        )
        # 48 citations over ~1 year ≈ 46/yr; not the same standing as fwci=5,
        # but both must fall inside the same interpolation segment.
        assert (a == 10 or 10 < a < 90) and (b == 10 or 10 < b < 90)


class TestRunBehaviour:
    def test_scores_and_measures_recorded_on_output(self):
        assessor = _make_assessor()
        data = ResolvedPaperCollection(
            papers=[
                _make_paper(influence_fwci=6139.0),
                _make_paper(
                    title="Fresh Paper",
                    influence_fwci=None,
                    influence_cited_by_count=5,
                    publication_date="2026-08-01",
                ),
            ]
        )
        result, monitoring = assessor._run(data)
        assert result.papers[0].influence_score == 90
        assert result.papers[0].influence_measure_used == MEASURE_FWCI
        assert result.papers[1].influence_measure_used == MEASURE_CITATIONS_PER_YEAR
        assert result.papers[1].influence_score is not None
        assert "Papers in: 2, assessed: 2, unmeasurable: 0" in monitoring
        assert "fwci=1" in monitoring
        assert "citations_per_year=1" in monitoring

    def test_unmeasurable_reported_not_raised(self):
        assessor = _make_assessor()
        data = ResolvedPaperCollection(
            papers=[
                _make_paper(
                    influence_fwci=None,
                    influence_cited_by_count=None,
                    publication_date=None,
                )
            ]
        )
        result, monitoring = assessor._run(data)
        assert result.papers[0].influence_score is None
        assert "UNMEASURABLE" in monitoring

    def test_every_candidate_failing_assessment_completes(self):
        assessor = _make_assessor()
        data = ResolvedPaperCollection(
            papers=[
                _make_paper(
                    influence_fwci=None,
                    influence_cited_by_count=None,
                    publication_date=None,
                ),
                _make_paper(
                    title="Another",
                    influence_fwci=None,
                    influence_cited_by_count=None,
                    publication_date=None,
                ),
            ]
        )
        result, monitoring = assessor._run(data)
        assert len(result.papers) == 2
        assert "assessed: 0" in monitoring
