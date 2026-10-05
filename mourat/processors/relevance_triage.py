"""Batched boolean relevance triage for large paper feeds (spec 14, R9).

The first stage after RSS collection: drops papers that are clearly
irrelevant to every configured research attribute before the expensive
per-item stages (HTML fetch, authority LLM call, relevance scoring).

This is a TRUE classifier: given a batch of papers (title + abstract) and
the attribute list (research questions, technical challenges, research
topics — constraints excluded by design), it returns only per-paper
true/false — "plausibly relevant to at least one attribute" — nothing else.

Fail-open rule: a paper with no valid verdict (transport error, malformed
response, or a verdict missing from an otherwise-valid response after the
bounded retries) is KEPT, never dropped. This filter is the one place where
a false positive destroys good data before any other stage sees it.
"""

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a relevance triage classifier for a research paper feed. "
    "You decide whether each paper is plausibly relevant to at least one of "
    "the listed research attributes. Be conservative: when genuinely unsure, "
    "answer true — a false 'true' only costs one extra scoring call, while a "
    "false 'false' permanently discards a potentially relevant paper."
)


class TriageVerdict(BaseModel):
    """One paper's boolean verdict, matched back by id."""

    id: str = Field(description="The paper id exactly as given in the input")
    relevant: bool = Field(
        description=(
            "True when the paper is plausibly relevant to at least one listed "
            "research attribute, false otherwise"
        )
    )


class TriageResult(BaseModel):
    verdicts: list[TriageVerdict]


def _render_attributes(
    rq_list: list[dict],
    tc_list: list[dict],
    topic_list: list[dict],
) -> str:
    """Attribute blocks for the prompt. Constraints are never included."""
    blocks = []
    for kind, label, entities in (
        ("rq", "Research questions", rq_list),
        ("tc", "Technical challenges", tc_list),
        ("topic", "Research topics", topic_list),
    ):
        if not entities:
            continue
        lines = [f"{label}:"]
        for e in entities:
            lines.append(f"- id={e['id']} name={e['name']}: {e.get('description', '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_batch_prompt(papers: list[tuple[str, str, str]], attributes: str) -> str:
    """Papers are (id, title, description) triples."""
    paper_lines = []
    for pid, title, description in papers:
        paper_lines.append(
            json.dumps(
                {"id": pid, "title": title, "abstract": description},
                ensure_ascii=False,
            )
        )
    return (
        "### Research attributes\n\n"
        f"{attributes}\n\n"
        "### Papers\n\n"
        + "\n".join(paper_lines)
        + "\n\nFor each paper, decide whether it is plausibly relevant to at "
        "least one research attribute listed above. Return one verdict per "
        "paper: its id (exactly as given) and a boolean 'relevant' field. "
        "Return nothing else."
    )


class RelevanceTriageClassifier(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Keeps candidates plausibly relevant to at least one attribute (R9).

    Pass-through by count: the output collection contains only the kept
    papers, in input order. Fail-open: any paper without a valid 'false'
    verdict is kept.
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        rq_list: list[dict] | None = None,
        tc_list: list[dict] | None = None,
        topic_list: list[dict] | None = None,
        batch_size: int = 25,
        system_prompt: str = SYSTEM_PROMPT,
        request_retries: int = 2,
        request_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=TriageResult,
            system_prompt=system_prompt,
        )
        self.rq_list = rq_list or []
        self.tc_list = tc_list or []
        self.topic_list = topic_list or []
        self.batch_size = batch_size
        self.request_retries = request_retries
        self.request_retry_delay_seconds = request_retry_delay_seconds
        super().__init__(monitoring_handler)

    def _run_batch(
        self, papers: list[tuple[str, str, str]], attributes: str
    ) -> dict[str, bool]:
        """One batch call with the bounded retry loop; None on exhaustion."""
        prompt = _build_batch_prompt(papers, attributes)
        last_error: Exception | None = None
        result: TriageResult | None = None
        for attempt in range(self.request_retries + 1):
            try:
                result = self.agent.run_sync(prompt).output
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.request_retries:
                    logger.warning(
                        "triage batch of %d papers failed after %d attempts: %s",
                        len(papers),
                        attempt + 1,
                        exc,
                    )
                    return {}
                logger.warning(
                    "triage batch attempt %d/%d failed: %s",
                    attempt + 1,
                    self.request_retries + 1,
                    exc,
                )
                time.sleep(self.request_retry_delay_seconds)
        if result is None:  # defensive: loop always sets result or returns
            return {}

        valid_ids = {pid for pid, _, _ in papers}
        verdicts: dict[str, bool] = {}
        for v in result.verdicts:
            if v.id in valid_ids:  # hallucinated ids are silently discarded
                verdicts[v.id] = bool(v.relevant)
        return verdicts

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        attributes = _render_attributes(self.rq_list, self.tc_list, self.topic_list)
        if not attributes:
            # No attributes configured: nothing to be relevant to. Fail open.
            logger.warning(
                "No research attributes configured for triage; keeping all "
                "%d papers",
                len(data.papers),
            )
            return data, (
                f"Triage skipped: no research attributes configured; "
                f"kept {len(data.papers)} papers"
            )

        triples = [
            (f"paper_{i}", p.title, p.description) for i, p in enumerate(data.papers)
        ]

        kept: list[PaperCandidate] = []
        dropped_lines: list[str] = []
        id_to_paper = {f"paper_{i}": p for i, p in enumerate(data.papers)}

        for start in range(0, len(triples), self.batch_size):
            batch = triples[start : start + self.batch_size]
            verdicts = self._run_batch(batch, attributes)
            for pid, _, _ in batch:
                paper = id_to_paper[pid]
                verdict = verdicts.get(pid)
                if verdict is False:
                    dropped_lines.append(
                        f"### {paper.title}\nDropped by relevance triage."
                    )
                else:
                    # None (no verdict / failed batch) fails open: kept.
                    kept.append(paper)

        total = len(data.papers)
        dropped_count = total - len(kept)
        pct = (100.0 * dropped_count / total) if total else 0.0
        lines = [
            f"Papers in: {total}, kept: {len(kept)}, dropped: {dropped_count} "
            f"({pct:.1f}% dropped)"
        ]
        lines.extend(dropped_lines)
        return PaperCandidateCollection(papers=kept), "\n".join(lines)
