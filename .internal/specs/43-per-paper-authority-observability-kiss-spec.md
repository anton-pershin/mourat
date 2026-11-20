## Per-paper authority observability

### 1. Requirement analysis

This spec implements approved GitHub issue #43. It uses the optional verdict rationale and structured assessment result supplied by `anton-pershin/comourat#30` and merged in `anton-pershin/comourat#31`.

**R1 — Author and affiliation details.** For every paper in the `AuthorityInfluenceAssessor` monitoring report, retain the existing identifier, title, and score or unassessable reason, and report the author and affiliation input exactly as the authority-assessment prompt receives it. Add an `Authors:` line containing the candidate's author-list representation and an `Affiliations:` line containing the affiliation mapping. Use the prompt's existing rendering rules: preserve every mapping entry and its order, render each as `author: affiliation 1, affiliation 2`, and use `unknown` when the mapping is absent or empty. Do not filter values, reconcile provider and candidate names, reorder entries, or otherwise interpret the input.

**R2 — Score explanations.** For every successfully scored paper, including a paper with score 0, add the optional `rationale` from the `AuthorityVerdict` returned by `AuthorityInfluenceAssessorCore.assess()`. Match the verdict to the candidate through the existing `paper_<index>` ID. If the validated verdict has no rationale or has a blank rationale, report `Rationale: not provided`; do not change the score or make the paper unassessable.

**R3 — Existing failure behavior.** For an unassessable paper, retain its cause-specific reason and add its author and affiliation details. Do not add a score-explanation line because no validated verdict exists. Preserve fail-open scores, retries, follow-up behavior, batching, and all aggregate summary lines.

**R4 — Monitoring ownership.** Keep the change in the `mourat` Function adapter. Do not add monitoring behavior to `comourat`, persist rationales on paper candidates, or change pipeline outputs other than the authority-assessment monitoring text.

Expected behavior variants:

- **B1 — Explained score.** A scored paper has authors, known affiliations, and a rationale; all appear next to its existing assessment line.
- **B2 — Valid zero score.** A validated score-0 verdict receives the same author, affiliation, and rationale details as any other scored paper.
- **B3 — Missing affiliations or authors.** Monitoring shows the prompt's empty author list or `unknown` affiliation value unchanged and does not affect assessment.
- **B4 — Missing optional rationale.** A scored verdict without a rationale remains scored and reports `Rationale: not provided`.
- **B5 — Unassessable paper.** A missing or failed verdict keeps its existing reason, includes the prompt's author and affiliation inputs, and has no rationale line.
- **B6 — Provider name differs.** The affiliation mapping contains provider-only display names with known affiliations alongside candidate names with `unknown` values. Monitoring retains all entries in their existing order, exactly as the prompt receives them.

### 2. Tests

- **T1 — Detailed scored papers.** Covers R1, R2, R4, B1, and B2. Assess two papers with different rationales. Give one paper several authors and affiliations and a score of 0; give the other paper no affiliation data. Assert the exact adjacent assessment, author-list, affiliation-mapping, and rationale lines using the prompt's rendering rules.
- **T2 — Optional rationale and empty authors.** Covers R1, R2, B3, and B4. Assess papers with no authors and valid verdicts with an omitted or blank rationale. Assert `Authors: []`, `Affiliations: unknown`, `Rationale: not provided`, and retention of the scores.
- **T3 — Unassessable details.** Covers R1, R3, B3, and B5. Omit a paper's verdict. Assert its existing unassessable reason, its prompt author list and `Affiliations: unknown`, no rationale line, and the unchanged aggregate summary.
- **T4 — Provider-only names retain affiliation evidence.** Covers R1, R2, B1, B3, and B6. Reproduce the reviewed DOI example: candidate names differ from OpenAlex display names, and the real affiliation-mapping behavior retains both groups. Assert that the monitoring author list and complete affiliation string match the prompt input, including candidate `unknown` values, provider institutions, order, and the rationale.
- **T5 — Regression coverage.** Covers R1 through R4 and all variants. Run the authority-assessor tests and the complete test suite. Confirm existing score matching, batch splitting, retries, stable identifiers, zero-score handling, and summary retention remain valid. Run formatting, import-order, lint, and type checks for the touched code.

### 3. Implementation plan

#### 3.1 Implementation repos

- `anton-pershin/mourat` — specification, adapter implementation, and tests.
- `anton-pershin/comourat` — completed external dependency only; no change in this spec.

#### 3.2 High-level design

The first diagram copies the newest-paper pipeline from the constitution. Orange marks the modified authority-influence component. Other stages and their relationships remain unchanged.

```mermaid
flowchart LR
  DB[(Database)]
  N1["Producer<br/>Collect and merge newest papers"] --> N2["Sieve<br/>Relevance triage"] --> N3["Map<br/>Fetch affiliations"] --> N4["Map<br/>Assess authority influence"] --> N5["Map<br/>Resolve candidates"] --> N6["Map<br/>Score research relevance"] --> N7["Sieve<br/>Filter by score"]
  N7 --> NDB["Consumer<br/>Database writer"]
  N7 --> NJ["Consumer<br/>JSONL writer"]
  DB -. "research context and collection ledger" .-> N1
  DB -. "research context" .-> N2
  DB -. "research context" .-> N6
  N1 -. "record after success" .-> DB
  NDB -->|"store on request"| DB
  NJ --> NFILE[(JSONL output)]
  style N4 fill:#fff0dc,stroke:#e67e22,stroke-width:4px
```

The component diagram shows the extended adapter data flow. The core supplies the existing summary lines and validated verdicts. The adapter adds candidate-owned author details and verdict-owned rationales to the monitoring text.

```mermaid
flowchart LR
  C["Paper candidates<br/>authors and affiliations"] --> A["Mourat AuthorityInfluenceAssessor"]
  A -->|"bridge affiliations by paper index"| Core["Comourat AuthorityInfluenceAssessorCore"]
  Core --> O["Assessed candidates"]
  Core --> L["Existing per-paper and aggregate lines"]
  Core --> V["Validated verdicts by paper ID<br/>score and optional rationale"]
  C --> D["Format author and affiliation details"]
  L --> E["Insert per-paper detail lines"]
  V --> E
  D --> E
  E --> M["MonitoringHandler"]
  O --> P["Next pipeline stage"]
  style A fill:#fff0dc,stroke:#e67e22,stroke-width:4px
  style D fill:#fff0dc,stroke:#e67e22,stroke-width:4px
  style E fill:#fff0dc,stroke:#e67e22,stroke-width:4px
```

The adapter consumes the three-value `assess()` result. It inserts indented `Authors:` and `Affiliations:` lines after every existing per-paper assessment line. These lines reproduce the candidate author list and affiliation string used in the prompt without filtering or name reconciliation. For a paper with a validated verdict, the adapter also inserts an indented `Rationale:` line. It leaves the aggregate lines and all assessment decisions unchanged.

#### 3.3 Todo list

1. [x] Add the specification and focused monitoring tests.
2. [x] Run the focused tests before the adapter change and confirm failure.
3. [x] Consume the structured verdict result and add per-paper author, affiliation, and rationale lines.
4. [x] Run focused tests and fix failures.
5. [x] Run the complete test suite and repository quality checks.
6. [x] Push the assigned branch and open a draft pull request linked to issue #43.

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `.internal/specs/43-per-paper-authority-observability-kiss-spec.md` | New: requirements, tests, and component design for detailed authority monitoring. |
| `mourat/processors/authority_influence.py` | Modified: consume validated verdicts and enrich per-paper monitoring lines. |
| `tests/test_newest_paper_pipeline.py` | Modified: cover authors, affiliations, rationales, missing optional data, and unassessable papers. |
