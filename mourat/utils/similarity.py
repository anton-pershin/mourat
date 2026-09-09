"""Normalised-title comparison shared by resolution components."""

import difflib
import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Case-fold and collapse whitespace runs to a single space."""
    return _WHITESPACE_RUN.sub(" ", title.strip()).casefold()


def title_similarity(claimed: str, resolved: str) -> float:
    """Similarity of two titles in [0, 1], after normalisation."""
    return difflib.SequenceMatcher(
        None, normalize_title(claimed), normalize_title(resolved)
    ).ratio()


def titles_match(claimed: str, resolved: str, threshold: float) -> bool:
    """True when similarity reaches the configured threshold."""
    return title_similarity(claimed, resolved) >= threshold
