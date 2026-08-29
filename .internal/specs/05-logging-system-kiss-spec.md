## Logging system

### 1. Requirement analysis

#### 1.1 Motivation

Diagnostic output in `mourat` is currently a mix of bare `print()` calls in library code
and `rich` console output in scripts. This has three consequences:

- **Nothing is persisted.** Hydra already configures a root logger with a file handler at
  `${hydra.runtime.output_dir}/${hydra.job.name}.log` (`config/hydra/job_logging/base.yaml`),
  but no module in `mourat/` ever calls a logger, so the log files are effectively empty.
  Every diagnostic goes to stdout and is lost when the terminal scrolls.
- **There is no timing information.** The pipeline is dominated by LLM calls and web
  fetches, and `WebEnricher` is the most expensive stage, but there is no record of how
  long any stage or item took. Obtaining per-stage timing is a primary motivation for this
  change.
- **Verbosity is not controllable.** Diagnostics are either always printed or not printed
  at all. There is no way to enable detailed tracing for one run, or for one component,
  without editing code.

Three output channels are currently conflated and must be kept distinct: **diagnostics**
(what becomes logging), **monitoring artifacts** (the `MonitoringHandler` markdown files,
unchanged by this spec), and **human-facing CLI UI** (the `rich` tables, markdown and
interactive prompt in scripts, also unchanged).

#### 1.2 Functional requirements

- **FR1.** Every module under `mourat/` that emits diagnostics obtains a logger via
  `logging.getLogger(__name__)` declared at module level. No central `get_logger()` wrapper
  and no `mourat/utils/logging.py` module is introduced.
- **FR2.** No module under `mourat/` configures logging. `logging.basicConfig` is not
  called anywhere in the package. Logging configuration lives exclusively in
  `config/hydra/job_logging/base.yaml` and is applied by Hydra at job start.
- **FR3.** All existing bare `print()` calls carrying diagnostics are replaced by logger
  calls at the level given in 1.4:
  `mourat/classifiers.py`, `mourat/assigners.py`, `mourat/scorers.py` (two sites),
  `mourat/enrichers/web_enricher.py`, `mourat/collectors/semantic_scholar.py`,
  `mourat/scripts/collect_posts.py`.
- **FR4.** Diagnostics emitted inside an `except` block that reports a failure use
  `logger.exception` (or `logger.error(..., exc_info=True)`) so the traceback is recorded.
- **FR5.** `Function.__call__` in `mourat/base.py` logs, for every pipeline step: a start
  record, and a completion record carrying the elapsed time of `_run`, the elapsed time of
  the monitoring handler call, and their total. Records identify the step by `step_id` and
  by the component class name.
- **FR6.** If `_run` raises, `Function.__call__` logs the failure with the elapsed time and
  the traceback, then re-raises. The exception is not swallowed.
- **FR7.** The completion record uses a fixed field layout
  (`step <id> | <Component> | done | run=<s> monitoring=<s> total=<s>`) so that a full
  pipeline profile can be extracted from a log file with a single `grep`.
- **FR8.** Components that iterate over items or batches log one `DEBUG` record per
  iteration containing the position (`i/n`), an identifier of the item, and the elapsed
  time of that iteration. This applies to `WebEnricher`, `PostScorer`,
  `BinaryPaperClassifier`, `PostSlopClassifier` and `PaperScorer`.
- **FR9.** `WebEnricher` logs the LLM call duration separately from the total per-post
  duration, so that time spent in the model can be distinguished from time spent fetching
  pages.
- **FR10.** Fail-open paths in `PostSlopClassifier` that are currently silent — a batch
  whose model call raised, a post for which no verdict was returned, a verdict carrying an
  unknown `submission_id` — emit a `WARNING`.
- **FR11.** `rich.progress.track` is removed from all library code:
  `mourat/classifiers.py` (two sites), `mourat/scorers.py`,
  `mourat/processors/post_scorer.py`, together with the now-unused imports. The progress
  signal it provided is replaced by the FR8 `DEBUG` records.
- **FR12.** `rich.progress.track` is retained in `mourat/scripts/print_reddit_summary.py`,
  which is an interactive, human-facing script that persists nothing.
- **FR13.** The logging configuration defines two handlers, `console` and `file`, sharing
  the same formatter, so that console output is a literal mirror of the file. Removing the
  `console` handler later must not change the file output in any way.
- **FR14.** The `mourat` logger is set to `INFO`; the root logger is set to `WARNING`; the
  noisy third-party loggers `httpx`, `httpcore`, `openai`, `praw`, `prawcore`, `urllib3`
  and `trafilatura` are set to `WARNING`. All levels are declared explicitly in the config
  rather than relying on library defaults.
- **FR15.** Verbosity is changeable per run from the command line without editing code,
  both globally (`hydra.job_logging.loggers.mourat.level=DEBUG`) and for a single module
  (`+hydra.job_logging.loggers.mourat.enrichers.web_enricher.level=DEBUG`).
- **FR16.** The `rich` output of `mourat/scripts/retrieve_content.py` is split by purpose:
  its error paths (unknown query type, `db_path` not set, database file missing) become
  logged errors, while the results `Table` and the "no content items found" message remain
  `rich` console output. The module-local `Console()` at line 20 is replaced by the shared
  singleton from `mourat/utils/console.py`.

#### 1.3 Non-functional requirements

- **NFR1.** Only the standard library `logging` module is used. No third-party logging
  package and no `rich.logging.RichHandler` is introduced.
- **NFR2.** Log messages use `%`-style lazy formatting (`logger.debug("x %s", v)`), not
  f-strings, so that arguments of suppressed records are not formatted.
- **NFR3.** No secret is ever logged. In particular, `DEBUG` records may contain prompts
  but must never contain API keys, LLM client configuration, or the contents of
  `user_settings.yaml`.
- **NFR4.** Elapsed times are measured with `time.monotonic()`, not `time.time()`.
- **NFR5.** The change is behaviour-preserving for the pipeline: no component's inputs,
  outputs, `text_for_monitoring` content, or control flow is altered by this spec.
- **NFR6.** The `MonitoringHandler` mechanism is untouched. Monitoring markdown files
  remain the artifact channel and are not duplicated into the log.

#### 1.4 Expected behavioural variants

| Situation | Expected behaviour |
|---|---|
| A pipeline step starts | `INFO` record with step id and component class name |
| A pipeline step completes | `INFO` record with step id, component name, `_run` duration, monitoring duration, total |
| `_run` raises | `ERROR` record with step id, component name, elapsed time and traceback; exception re-raised |
| The monitoring handler call is slow | Its duration appears as a separate field, not folded into the `_run` duration |
| LLM answer fails validation in `BinaryPaperClassifier` / `PaperAssigner` (paper dropped) | `ERROR` with traceback, naming the dropped paper |
| LLM answer fails validation in `PaperScorer` (batch skipped) | `ERROR` with traceback, naming the number of papers skipped |
| `PaperScorer` receives a verdict whose title matches no input paper | `WARNING` naming the unmatched title |
| `WebEnricher` exceeds the usage limit for a post | `ERROR` with traceback, naming the skipped post |
| `PostSlopClassifier` batch call raises | `WARNING` naming the batch; all its posts retained (fail-open) |
| `PostSlopClassifier` returns no verdict for a post | `WARNING` naming the post; post retained |
| `PostSlopClassifier` returns a verdict for an unknown `submission_id` | `WARNING` naming the id; verdict ignored |
| Semantic Scholar returns an API error code and the collector retries | `WARNING` with the error code, message and retry delay |
| `collect_posts` finishes | `INFO` with the per-stage counts (collected / heuristic / slop / enriched / scored / saved) |
| Per-item iteration in an expensive stage | `DEBUG` with `i/n`, item id and elapsed time; suppressed at the default level |
| `WebEnricher` processes a post | `DEBUG` distinguishing LLM call time from total per-post time |
| Default configuration | `mourat` records at `INFO` and above reach both handlers; third-party records below `WARNING` are suppressed |
| `hydra.job_logging.loggers.mourat.level=DEBUG` passed on the command line | Per-item `DEBUG` records appear, without any code change |
| A single module's level is overridden | Only that module's `DEBUG` records appear |
| The `console` handler is removed from the config | File output is byte-identical to before; nothing is printed to stdout |
| A long stage runs with `track` removed | Console shows the start record and then nothing until completion; progress is recoverable from the log at `DEBUG` |
| `retrieve_content` runs with `db_path` unset or the database missing | `ERROR` logged; script returns cleanly without creating the database |
| `retrieve_content` runs successfully | The results `Table` is printed to the console via `rich`, not through logging |

### 2. Tests

Manual verification, no automated tests.

### 3. Implementation plan

#### 3.1 Implementation repos

`mourat` is both the management repo and the sole implementation repo. All changes are
made there.

#### 3.2. Solution design

##### Logging configuration (`config/hydra/job_logging/base.yaml`)

The file already exists and already defines the `simple` formatter and the `console` /
`file` handlers. It is extended with an explicit `loggers` block and an explicit root
level; the formatter and handler definitions are unchanged, so both handlers keep emitting
the identical record format (FR13):

```yaml
loggers:
  mourat:
    level: INFO
  httpx: {level: WARNING}
  httpcore: {level: WARNING}
  openai: {level: WARNING}
  praw: {level: WARNING}
  prawcore: {level: WARNING}
  urllib3: {level: WARNING}
  trafilatura: {level: WARNING}
root:
  level: WARNING
  handlers: [file, console]
```

Because levels live in the config, both the global and the per-module verbosity overrides
of FR15 work as plain Hydra overrides with no code support. Retiring the console later is
the single edit `handlers: [file]`.

##### Stage timing (`mourat/base.py`)

`Function.__call__` becomes the single instrumentation point for every component in the
project — collectors, enrichers, classifiers, scorers and filters all inherit it, so one
edit satisfies FR5–FR7 for all of them and no component needs its own stage logging.

```python
logger = logging.getLogger(__name__)

def __call__(self, data: InputT, step_id: str) -> OutputT:
    name = type(self).__name__
    logger.info("step %s | %s | start", step_id, name)

    t0 = time.monotonic()
    try:
        output, text_for_monitoring = self._run(data)
    except Exception:
        logger.exception(
            "step %s | %s | FAILED after %.2fs", step_id, name, time.monotonic() - t0
        )
        raise
    run_s = time.monotonic() - t0

    t1 = time.monotonic()
    self.monitoring_handler(step=step_id, text_for_monitoring=text_for_monitoring)
    monitoring_s = time.monotonic() - t1

    logger.info(
        "step %s | %s | done | run=%.2fs monitoring=%.2fs total=%.2fs",
        step_id, name, run_s, monitoring_s, run_s + monitoring_s,
    )
    return output
```

`_run` time and monitoring-write time are measured separately so that a slow monitoring
handler cannot silently inflate a component's cost. The `except` clause exists so that a
crash mid-pipeline still records which stage died and how long it had been running, then
re-raises unchanged (FR6, NFR5).

##### Per-item timing

In each expensive loop the iteration is wrapped with a `time.monotonic()` reading and a
single `DEBUG` record carrying `i/n`, an item identifier and the elapsed time (FR8). The
identifier is `submission_id` for posts and the title for papers. The loop counter also
supplies the progress signal that `track` used to give:

| Component | Loop | Identifier |
|---|---|---|
| `WebEnricher._run` | per post | `post_info.submission_id` |
| `PostScorer._run` | per post | `ep.post.submission_id` |
| `PostSlopClassifier._run` | per batch | batch index and size |
| `BinaryPaperClassifier._run` | per paper | `p.title` |
| `PaperScorer._run` | per group | group index and size |

##### LLM vs fetch time in `WebEnricher` (FR9)

`WebEnricher` runs an agent with two tools, so its per-post wall clock is the sum of model
time and web-fetch time, and `agent.run_sync` does not expose the split. The two tool
functions `extract_url` and `web_search` in `_create_enrichment_agent` are therefore
instrumented individually: each logs its own `DEBUG` duration and accumulates into a
per-post tool-time total. The per-post record then reports `total`, `tools` and
`llm=total-tools`, which is the split that determines whether a slow post is the model's
fault or the network's.

##### `print` → logger conversion

Seven sites, each keeping its existing message text and control flow (NFR5) and gaining
`%`-style lazy arguments (NFR2):

| Site | New level | Reason |
|---|---|---|
| `classifiers.py` `BinaryPaperClassifier` validation failure | `logger.exception` | paper is dropped |
| `assigners.py` validation failure | `logger.exception` | paper is dropped |
| `scorers.py` batch validation failure | `logger.exception` | a whole batch is skipped |
| `scorers.py` unmatched verdict title | `logger.warning` | one verdict discarded, no traceback available |
| `enrichers/web_enricher.py` usage limit exceeded | `logger.exception` | post is skipped |
| `collectors/semantic_scholar.py` API error code | `logger.warning` | transient, the collector retries |
| `scripts/collect_posts.py` final summary | `logger.info` | normal completion |

##### Silent fail-open paths (FR10)

`PostSlopClassifier._run` currently has three paths that discard information with a bare
`continue` and no output at all: the `except UnexpectedModelBehavior` around the batch
call, the `verdict is None` branch, and — implicitly — verdicts whose `submission_id` is
absent from `verdict_by_id`'s source batch. Each gains a `WARNING`. The unknown-id case
needs one added check, since verdicts are currently looked up by post rather than iterated,
so an unknown id is dropped without any code path noticing it. Behaviour is unchanged in
all three cases: posts are still retained, verdicts still ignored.

##### `track` removal (FR11, FR12)

`for x in track(seq, description=...)` becomes `for i, x in enumerate(seq, 1)` in
`classifiers.py` (both sites), `scorers.py` and `processors/post_scorer.py`, and the
`from rich.progress import track` import is dropped from all three files.
`scripts/print_reddit_summary.py` keeps its `track` and its import.

Note that `BinaryPaperClassifier.__init__` and `PaperScorer.__init__` accept a
`progress_title` parameter that is stored but never used even today. It is left in place —
removing it would change the constructor signature and the Hydra configs that may pass it,
which is outside this spec.

##### `retrieve_content.py` (FR16)

The module-local `Console()` is removed in favour of `from mourat.utils.console import
console`. The three error paths (`unknown query type`, `db_path` unset, database file
missing) become `logger.error` calls and keep returning cleanly without creating the
database. The results `Table` and the `No content items found.` message stay on `console`,
because they are the script's output rather than diagnostics.

#### 3.3 Todo list

1. [ ] Extend `config/hydra/job_logging/base.yaml` with the explicit `loggers` block and
   root level
2. [ ] Add stage start / completion / failure timing to `Function.__call__` in
   `mourat/base.py`
3. [ ] Convert the seven `print` sites to logger calls at the levels listed in 3.2
4. [ ] Add `WARNING` records to the three silent fail-open paths in `PostSlopClassifier`
5. [ ] Remove `track` from `classifiers.py`, `scorers.py` and
   `processors/post_scorer.py`, and drop the unused imports
6. [ ] Add per-item `DEBUG` timing to the five loops listed in 3.2
7. [ ] Instrument `extract_url` and `web_search` and report the LLM / tool split per post
   in `WebEnricher`
8. [ ] Switch `retrieve_content.py` to the shared console and convert its error paths to
   `logger.error`
9. [ ] Confirm the static checks: no `print(` left in `mourat/` except the interactive
   prompt in `utils/console.py`, no `basicConfig`, `track` only in
   `print_reddit_summary.py`, no f-string log messages
10. [ ] Run `python -m black`, `isort`, `pylint`, `mypy` on the modified files with the
    project venv and fix what they report
11. [ ] Hand over to the user for manual verification

#### 3.4 Modification summary

| File | Action |
|------|--------|
| `config/hydra/job_logging/base.yaml` | Modified: add explicit `loggers` block (`mourat` at INFO, seven third-party loggers at WARNING) and explicit root level |
| `mourat/base.py` | Modified: add `logging`/`time` imports and module logger; add start, completion and failure records with `_run` / monitoring / total timings to `Function.__call__` |
| `mourat/classifiers.py` | Modified: add module logger; remove `track` and its import (2 sites); convert the validation-failure `print` to `logger.exception`; add WARNING to the three fail-open paths in `PostSlopClassifier`; add per-paper and per-batch DEBUG timing |
| `mourat/assigners.py` | Modified: add module logger; convert the validation-failure `print` to `logger.exception` |
| `mourat/scorers.py` | Modified: add module logger; remove `track` and its import; convert the batch-failure `print` to `logger.exception` and the unmatched-title `print` to `logger.warning`; add per-group DEBUG timing |
| `mourat/enrichers/web_enricher.py` | Modified: add module logger; convert the usage-limit `print` to `logger.exception`; instrument `extract_url` and `web_search`; add per-post DEBUG timing with the LLM / tool split |
| `mourat/collectors/semantic_scholar.py` | Modified: add module logger; convert the API-error `print` to `logger.warning` |
| `mourat/processors/post_scorer.py` | Modified: add module logger; remove `track` and its import; add per-post DEBUG timing |
| `mourat/scripts/collect_posts.py` | Modified: add module logger; convert the final summary `print` to `logger.info` |
| `mourat/scripts/retrieve_content.py` | Modified: add module logger; replace the local `Console()` with the shared singleton; convert the three error paths to `logger.error` |
| `mourat/utils/console.py` | Unchanged: remains the `rich` singleton and the interactive prompt |
| `mourat/scripts/print_reddit_summary.py` | Unchanged: keeps `track` and its `rich` output (FR12) |
| `mourat/monitoring.py` | Unchanged (NFR6) |
