"""Tests for the shared title-similarity utility."""

from mourat.utils.similarity import normalize_title, title_similarity, titles_match


class TestNormalizeTitle:
    def test_case_and_whitespace_are_neutralised(self):
        assert normalize_title("  A   Study\n of\tThings ") == "a study of things"

    def test_empty_string_stays_empty(self):
        assert normalize_title("") == ""


class TestTitleSimilarity:
    def test_identical_titles_score_one(self):
        assert (
            title_similarity("Attention Is All You Need", "attention is all you need")
            == 1.0
        )

    def test_completely_different_titles_score_low(self):
        score = title_similarity(
            "Attention Is All You Need", "Fitting Linear Mixed-Effects Models"
        )
        assert score < 0.3

    def test_score_is_bounded_and_symmetric(self):
        a, b = "A Study of Whitespace", "A Study of Whitespace Patterns"
        score = title_similarity(a, b)
        assert 0.0 <= score <= 1.0
        assert score == title_similarity(b, a)

    def test_trailing_version_markers_reduce_but_not_eliminate_similarity(self):
        # e.g. "Title" vs "Title [arXiv preprint]" style variants.
        score = title_similarity(
            "Attention Is All You Need", "Attention Is All You Need."
        )
        assert score > 0.9


class TestTitlesMatch:
    def test_identical_titles_match_at_default_threshold(self):
        assert titles_match("Some Paper Title", "  some  paper title ", threshold=0.9)

    def test_different_titles_do_not_match(self):
        assert not titles_match(
            "Attention Is All You Need",
            "A Milestone Paper About Something Else Entirely",
            threshold=0.9,
        )

    def test_threshold_drives_the_verdict(self):
        score = title_similarity(
            "A Study of Whitespace", "A Study of Whitespace Patterns"
        )
        assert titles_match(
            "A Study of Whitespace", "A Study of Whitespace Patterns", threshold=score
        )
        assert not titles_match(
            "A Study of Whitespace",
            "A Study of Whitespace Patterns",
            threshold=score + 0.01,
        )
