"""Authority influence assessment (spec 14, R4).

Maps (authors, affiliations) -> a single 0-100 authority influence score
per paper via a batched LLM call, judged against prompt-configured anchors.

Batched classification per the established pattern: each paper in the
prompt carries a stable id; verdicts are matched back by id; hallucinated
ids are silently discarded. Fail-open with score 0: this score is not a
filter anywhere on the path, so a paper never receives a score that removes
it from the pipeline.

Verdict failure handling (R4):
- (1) transport error / (2) malformed response on the whole batch: bounded
  retry loop (same batch re-sent up to `request_retries` attempts with
  `request_retry_delay_seconds` between attempts); after exhaustion every
  paper in the batch continues with score 0 and a logged warning.
- (3) verdict missing for a paper inside a valid response: ONE follow-up
  call containing only the missing papers (same retry loop); if that also
  fails, those papers continue with score 0 and a warning.
- (4) verdict for an id not in the batch: silently discarded.
- "LLM recognises nobody": score 0 (user ruling 2026-10-04), not None.
"""

import logging
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from mourat.base import Function
from mourat.data_models import PaperCandidate, PaperCandidateCollection
from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You assess the authority influence of research papers. For each paper "
    "you judge the reputation of its authors and their institutional "
    "affiliations and output one integer score 0-100.\n"
    "Anchor points:\n"
    "- 100: the paper has a field-leading figure (a name any researcher in "
    "the field would recognise) or authors from a top-tier institution "
    "(e.g. a top-10 CS university, a famous industrial lab).\n"
    "- 50: active researchers from a reputable but not famous group.\n"
    "- 0: none of the authors or institutions is recognisable to you.\n"
    "Judge the authors and affiliations only; the paper's topic must not "
    "affect the score."
)


class AuthorityVerdict(BaseModel):
    id: str = Field(description="The paper id exactly as given in the input")
    score: int = Field(description="Authority influence score 0-100", ge=0, le=100)


class AuthorityResult(BaseModel):
    verdicts: list[AuthorityVerdict]


def _render_papers(papers: list[tuple[str, PaperCandidate]]) -> str:
    lines = []
    for pid, paper in papers:
        if paper.affiliations:
            affs = "; ".join(
                f"{a}: {', '.join(insts)}" for a, insts in paper.affiliations.items()
            )
        else:
            affs = "unknown"
        lines.append(
            f'{{"id": "{pid}", "title": {paper.title!r}, '
            f'"authors": {paper.authors!r}, "affiliations": "{affs}"}}'
        )
    return "\n".join(lines)


class AuthorityInfluenceAssessor(
    Function[PaperCandidateCollection, PaperCandidateCollection]
):
    """Writes the path's influence_score onto candidates (R4, option (b)).

    Pass-through by count: every candidate is returned with
    `influence_score` filled (0 when unmeasurable or unassessable). Which
    measure produced the score is decided per path via the DB writer's
    `influence_metric_id` ("authority" on this path).
    """

    def __init__(
        self,
        monitoring_handler: MonitoringHandler,
        model: Model,
        batch_size: int = 25,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        request_retries: int = 2,
        request_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=AuthorityResult,
            system_prompt=system_prompt,
        )
        self.batch_size = batch_size
        self.request_retries = request_retries
        self.request_retry_delay_seconds = request_retry_delay_seconds
        super().__init__(monitoring_handler)

    def _call_with_retries(self, prompt: str, n_papers: int) -> AuthorityResult | None:
        """Bounded retry loop over one batch prompt; None on exhaustion."""
        for attempt in range(self.request_retries + 1):
            try:
                result: AuthorityResult = self.agent.run_sync(prompt).output
                return result
            except Exception as exc:
                if attempt >= self.request_retries:
                    logger.warning(
                        "authority batch of %d papers failed after %d attempts: %s",
                        n_papers,
                        attempt + 1,
                        exc,
                    )
                    return None
                logger.warning(
                    "authority batch attempt %d/%d failed: %s",
                    attempt + 1,
                    self.request_retries + 1,
                    exc,
                )
                time.sleep(self.request_retry_delay_seconds)
        return None

    def _run_batch(self, papers: list[tuple[str, PaperCandidate]]) -> dict[str, int]:
        """One batch; {} when the whole batch failed after retries (R4-1/2)."""
        prompt = (
            "Rate each paper's authority influence on a 0-100 scale.\n\n"
            + _render_papers(papers)
            + "\n\nReturn one verdict per paper: its id (exactly as given) "
            "and a 'score' 0-100. Return nothing else."
        )
        result = self._call_with_retries(prompt, len(papers))
        if result is None:
            return {}
        valid_ids = {pid for pid, _ in papers}
        return {v.id: v.score for v in result.verdicts if v.id in valid_ids}

    def _run(
        self, data: PaperCandidateCollection
    ) -> tuple[PaperCandidateCollection, str]:
        t_start = time.monotonic()
        zero_scored: list[str] = []
        total = len(data.papers)
        triples = [(f"paper_{i}", p) for i, p in enumerate(data.papers)]
        id_to_paper = dict(triples)

        for start in range(0, len(triples), self.batch_size):
            batch = triples[start : start + self.batch_size]
            scores = self._run_batch(batch)

            missing = [(pid, p) for pid, p in batch if pid not in scores]
            if missing and scores:
                # R4-3: ONE follow-up call with only the missing papers.
                logger.info(
                    "authority follow-up call for %d missing verdicts",
                    len(missing),
                )
                follow_up = self._run_batch(missing)
                scores.update(follow_up)

            for pid, paper in batch:
                score = scores.get(pid)
                if score is None:
                    zero_scored.append(paper.title)
                    paper.influence_score = 0
                else:
                    paper.influence_score = int(score)

        lines = [
            f"Papers: {total}, scored: {total - len(zero_scored)}, "
            f"zero-scored (unassessable or failed): {len(zero_scored)}"
        ]
        if zero_scored:
            lines.append("Zero-scored papers:")
            lines.extend(f"- {t}" for t in zero_scored)
        lines.append(f"(stage took {time.monotonic() - t_start:.1f}s)")
        return data, "\n".join(lines)
