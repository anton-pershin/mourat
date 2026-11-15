"""T13/T15: script wiring tests for the newest-papers pipeline.

Runs collect_newest_papers_main with a fixture DB and fully mocked
components (mocked HTTP, FunctionModel LLMs) — no network anywhere.
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from mourat.data_models import PaperCandidate, PaperCandidateCollection

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)


class _TestMonitoringHandler:
    """Monitoring handler that records calls in memory (no file I/O)."""

    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


def _init_db(tmp_path):
    import mourat.database.business_domain as bd
    import mourat.database.research_domain as rd
    from mourat.database import init_db

    fd, dbfile = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
    os.close(fd)
    conn = init_db(dbfile)
    rd.create_research_topic(conn, "topic1", "KV cache optimization")
    rd.update_research_topic(
        conn, "topic1", description="KV cache optimization methods"
    )
    return conn, dbfile


def _hydra_cfg(tmp_path, dbfile, overrides=()):
    from hydra import compose, initialize_config_dir

    config_dir = Path(REPO) / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config_collect_newest_papers",
            overrides=[
                f"++db_path={dbfile}",
                f"++jsonl_output_path={tmp_path}/out.jsonl",
                "research_topic_ids=[topic1]",
                "monitoring_handler=dummy_test",
                "++sources=[arxiv]",
                *overrides,
            ],
        )
    return cfg


class _DummyMonitoring:
    def __init__(self):
        self.calls = []

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        self.calls.append((step, text_for_monitoring))


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <arxiv:announce_type>new</arxiv:announce_type>
    <title>KV Cache Trick</title>
    <link href="http://arxiv.org/abs/2401.00001"/>
    <description>arXiv:2401.00001v1 [cs.LG] KV Cache Trick
Abstract: A method for KV cache optimization</description>
    <author><name>J. Smith</name></author>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
  <entry>
    <arxiv:announce_type>new</arxiv:announce_type>
    <title>Unrelated Cooking Study</title>
    <link href="http://arxiv.org/abs/2401.00002"/>
    <description>arXiv:2401.00002v1 [cs.LG] Unrelated Cooking Study
Abstract: Recipes and kitchen arrangements</description>
    <author><name>A. Chef</name></author>
    <published>2024-01-16T00:00:00Z</published>
  </entry>
</feed>
"""

AFFIL_HTML = """
<span class="ltx_personname">J. Smith</span>
<span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">Affiliation: </span>Famous Lab</span>
"""


def _llm_factory(scripted, prompts=None):
    """FunctionModel factory that pops the next scripted JSON per call."""

    def model_fn(messages, agent):
        if prompts is not None:
            prompts.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart(scripted.pop(0))])

    return FunctionModel(model_fn)


class TestScriptWiring:
    def test_full_chain_with_mocks(self, tmp_path, monkeypatch):
        conn, dbfile = _init_db(tmp_path)
        conn.close()

        cfg = _hydra_cfg(
            tmp_path,
            dbfile,
            overrides=[
                "db_writer.enabled=true",
                "feeds.cs_lg.api_url=http://fake/rss",
            ],
        )

        scripted = [
            # triage: paper_0 true, paper_1 false
            json.dumps(
                {
                    "verdicts": [
                        {"id": "paper_0", "relevant": True},
                        {"id": "paper_1", "relevant": False},
                    ]
                }
            ),
            # authority: paper_0 -> 88
            json.dumps({"verdicts": [{"id": "paper_0", "score": 88}]}),
            # scorer: one topic score above threshold
            json.dumps(
                {
                    "scores": [
                        {
                            "id": "topic1",
                            "type": "topic",
                            "score": 90,
                            "justification": "on topic",
                        }
                    ]
                }
            ),
        ]

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.create_connection",
            lambda path: (
                _init_db(tmp_path)[0] if path != dbfile else sqlite3.connect(dbfile)
            ),
        )

        from mourat.database import init_db as _init

        # reopen a persistent connection for the writers
        conn2 = _init(dbfile)

        def fake_create_connection(path):
            # the script closes each connection it opens; hand out fresh ones
            return _init(dbfile)

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.create_connection",
            fake_create_connection,
        )

        fake_client = MagicMock()

        def get_side_effect(url, **kwargs):
            return _FakeResponse(
                SAMPLE_FEED if url == "http://fake/rss" else AFFIL_HTML
            )

        fake_client.get.side_effect = get_side_effect

        from types import SimpleNamespace

        import httpx as real_httpx

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx",
            SimpleNamespace(
                Client=lambda **kw: fake_client, Timeout=real_httpx.Timeout
            ),
        )
        (
            monkeypatch.setattr(
                "mourat.scripts.collect_newest_papers._load_attributes", None
            )
            if False
            else None
        )

        with patch(
            "mourat.scripts.collect_newest_papers.hydra.utils.instantiate",
            side_effect=None,
        ):
            pass  # instantiation happens for real; LLMs are swapped below

        # Swap the two LLM instantiation calls to our FunctionModel
        real_instantiate = None
        import hydra as _hydra

        real_instantiate = _hydra.utils.instantiate

        def smart_instantiate(cfg_node, *args, **kwargs):
            target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
            if target.endswith(("OpenAIChatModel", "OpenAIResponsesModel")):
                return _llm_factory(scripted)
            return real_instantiate(cfg_node, *args, **kwargs)

        monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)

        from mourat.scripts.collect_newest_papers import collect_newest_papers_main

        collect_newest_papers_main(cfg)

        # JSONL written (jsonl_writer enabled by default)
        out = Path(cfg.jsonl_output_path)
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1  # only the relevant paper survived
        rec = json.loads(lines[0])
        assert rec["title"] == "KV Cache Trick"

        # DB row written with the authority metric
        import mourat.database.content_item as ci
        from mourat.database import init_db as _init

        item = ci.get_content_item(_init(dbfile), "paper_arxiv-2401.00001")
        assert item is not None
        assert item["influence_score"] == 88
        assert item["influence_metric_id"] == "authority"

    def test_both_writers_off_writes_nothing(self, tmp_path, monkeypatch):
        # T15: both off = both off; nothing written, no error
        conn, dbfile = _init_db(tmp_path)
        conn.close()

        cfg = _hydra_cfg(
            tmp_path,
            dbfile,
            overrides=[
                "db_writer.enabled=false",
                "jsonl_writer.enabled=false",
                "feeds.cs_lg.api_url=http://fake/rss",
            ],
        )

        scripted = [
            json.dumps(
                {
                    "verdicts": [
                        {"id": "paper_0", "relevant": True},
                        {"id": "paper_1", "relevant": False},
                    ]
                }
            ),
            json.dumps({"verdicts": [{"id": "paper_0", "score": 88}]}),
            json.dumps(
                {
                    "scores": [
                        {
                            "id": "topic1",
                            "type": "topic",
                            "score": 90,
                            "justification": "on topic",
                        }
                    ]
                }
            ),
        ]

        from mourat.database import init_db as _init

        conn2 = _init(dbfile)
        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.create_connection",
            lambda path: conn2,
        )

        fake_client = MagicMock()

        def get_side_effect(url, **kwargs):
            return _FakeResponse(
                SAMPLE_FEED if url == "http://fake/rss" else AFFIL_HTML
            )

        fake_client.get.side_effect = get_side_effect
        from types import SimpleNamespace

        import httpx as real_httpx

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx",
            SimpleNamespace(
                Client=lambda **kw: fake_client, Timeout=real_httpx.Timeout
            ),
        )

        import hydra as _hydra

        real_instantiate = _hydra.utils.instantiate

        def smart_instantiate(cfg_node, *args, **kwargs):
            target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
            if target.endswith(("OpenAIChatModel", "OpenAIResponsesModel")):
                return _llm_factory(scripted)
            return real_instantiate(cfg_node, *args, **kwargs)

        monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)

        from mourat.scripts.collect_newest_papers import collect_newest_papers_main

        collect_newest_papers_main(cfg)

        out = Path(cfg.jsonl_output_path)
        assert not out.exists()
        import mourat.database.content_item as ci
        from mourat.database import init_db as _init

        assert ci.get_content_item(_init(dbfile), "paper_arxiv-2401.00001") is None

    def test_jsonl_only_when_db_off(self, tmp_path, monkeypatch):
        # T15 second variant: DB off, JSONL on -> file produced
        conn, dbfile = _init_db(tmp_path)
        conn.close()

        cfg = _hydra_cfg(
            tmp_path,
            dbfile,
            overrides=[
                "db_writer.enabled=false",
                "feeds.cs_lg.api_url=http://fake/rss",
            ],
        )
        scripted = [
            json.dumps(
                {
                    "verdicts": [
                        {"id": "paper_0", "relevant": True},
                        {"id": "paper_1", "relevant": False},
                    ]
                }
            ),
            json.dumps({"verdicts": [{"id": "paper_0", "score": 88}]}),
            json.dumps(
                {
                    "scores": [
                        {
                            "id": "topic1",
                            "type": "topic",
                            "score": 90,
                            "justification": "on topic",
                        }
                    ]
                }
            ),
        ]

        from mourat.database import init_db as _init

        conn2 = _init(dbfile)
        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.create_connection", lambda path: conn2
        )
        fake_client = MagicMock()

        def get_side_effect(url, **kwargs):
            return _FakeResponse(
                SAMPLE_FEED if url == "http://fake/rss" else AFFIL_HTML
            )

        fake_client.get.side_effect = get_side_effect
        from types import SimpleNamespace

        import httpx as real_httpx

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx",
            SimpleNamespace(
                Client=lambda **kw: fake_client, Timeout=real_httpx.Timeout
            ),
        )

        import hydra as _hydra

        real_instantiate = _hydra.utils.instantiate

        def smart_instantiate(cfg_node, *args, **kwargs):
            target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
            if target.endswith(("OpenAIChatModel", "OpenAIResponsesModel")):
                return _llm_factory(scripted)
            return real_instantiate(cfg_node, *args, **kwargs)

        monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)

        from mourat.scripts.collect_newest_papers import collect_newest_papers_main

        collect_newest_papers_main(cfg)
        out = Path(cfg.jsonl_output_path)
        assert out.exists()
        assert len(out.read_text().strip().splitlines()) == 1


class _SemanticScholarResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def _run_semantic_scholar_script(
    tmp_path,
    monkeypatch,
    output_path=None,
    fail_db_write=False,
    author_affiliations=("Famous Lab",),
    openalex_authorships=None,
    prompts=None,
):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_PROXY", raising=False)
    conn, dbfile = _init_db(tmp_path)
    conn.close()
    output_path = output_path or tmp_path / "out.jsonl"
    cfg = _hydra_cfg(
        tmp_path,
        dbfile,
        overrides=[
            "++sources=[semantic_scholar]",
            "db_writer.enabled=true",
            f"++jsonl_output_path={output_path}",
            "semantic_scholar_newest.api_url=http://fake/ss",
            "semantic_scholar_newest.reference_date=2024-02-01",
            "semantic_scholar_newest.regular_delay_seconds=0",
            "semantic_scholar_newest.backoff_seconds=0",
            "semantic_scholar_newest.jitter_seconds=0",
            "++arxiv_html_affiliations.openalex_regular_delay_seconds=0",
            "++arxiv_html_affiliations.openalex_backoff_seconds=0",
        ],
    )
    scripted = [
        json.dumps({"verdicts": [{"id": "paper_0", "relevant": True}]}),
        json.dumps({"verdicts": [{"id": "paper_0", "score": 88}]}),
        json.dumps(
            {
                "scores": [
                    {
                        "id": "topic1",
                        "type": "topic",
                        "score": 90,
                        "justification": "on topic",
                    }
                ]
            }
        ),
    ]

    from mourat.database import init_db

    monkeypatch.setattr(
        "mourat.scripts.collect_newest_papers.create_connection",
        lambda path: init_db(dbfile),
    )
    payload = {
        "data": [
            {
                "paperId": "ss-journal",
                "title": "KV Cache Journal Paper",
                "url": "https://example.com/journal",
                "abstract": "A journal method for KV cache optimization",
                "citationCount": 0,
                "publicationDate": "2024-01-02",
                "authors": [{"authorId": "author-j-smith", "name": "J. Smith"}],
                "externalIds": {"DOI": "10.1234/JOURNAL"},
            }
        ],
        "token": None,
    }
    openalex_results = []
    if openalex_authorships is not None:
        openalex_results = [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1234/JOURNAL",
                "display_name": "KV Cache Journal Paper",
                "locations": [],
                "authorships": openalex_authorships,
            }
        ]
    fake_client = MagicMock()

    def get_side_effect(url, **kwargs):
        if url == "http://fake/ss/paper/search/bulk":
            return _SemanticScholarResponse(payload)
        if url == "https://api.openalex.org/works":
            return _SemanticScholarResponse({"meta": {}, "results": openalex_results})
        raise AssertionError(f"unexpected URL: {url}")

    fake_client.get.side_effect = get_side_effect
    fake_client.post.side_effect = lambda url, **kwargs: (
        _SemanticScholarResponse(
            [
                {
                    "authorId": "author-j-smith",
                    "name": "J. Smith",
                    "affiliations": list(author_affiliations),
                }
            ]
        )
        if url == "http://fake/ss/author/batch"
        else (_ for _ in ()).throw(AssertionError(f"unexpected URL: {url}"))
    )

    from comourat.clients.semantic_scholar import SemanticScholarClient

    def make_semantic_scholar_client(**kwargs):
        return SemanticScholarClient(http_client=fake_client, **kwargs)

    monkeypatch.setattr(
        "mourat.collectors.arxiv_html_affiliations.SemanticScholarClient",
        make_semantic_scholar_client,
        raising=False,
    )

    from comourat.clients.openalex import OpenAlexClient

    def make_openalex_client(**kwargs):
        return OpenAlexClient(http_client=fake_client, **kwargs)

    monkeypatch.setattr(
        "mourat.collectors.arxiv_html_affiliations.OpenAlexClient",
        make_openalex_client,
        raising=False,
    )

    from types import SimpleNamespace

    import httpx as real_httpx

    monkeypatch.setattr(
        "mourat.scripts.collect_newest_papers.httpx",
        SimpleNamespace(
            Client=lambda **kw: fake_client,
            Timeout=real_httpx.Timeout,
        ),
    )

    import hydra as _hydra

    real_instantiate = _hydra.utils.instantiate

    def smart_instantiate(cfg_node, *args, **kwargs):
        target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
        if target.endswith(("OpenAIChatModel", "OpenAIResponsesModel")):
            return _llm_factory(scripted, prompts)
        return real_instantiate(cfg_node, *args, **kwargs)

    monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)
    if fail_db_write:

        def fail_create(*args, **kwargs):
            raise sqlite3.DatabaseError("injected database write failure")

        monkeypatch.setattr(
            "mourat.writers.db_writer.ci.create_content_item", fail_create
        )

    from mourat.scripts.collect_newest_papers import collect_newest_papers_main

    collect_newest_papers_main(cfg)
    return dbfile, output_path, fake_client


def test_semantic_scholar_only_reaches_shared_tail_and_records_ledger(
    tmp_path, monkeypatch
):
    prompts = []
    dbfile, output_path, fake_client = _run_semantic_scholar_script(
        tmp_path, monkeypatch, prompts=prompts
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["title"] for record in records] == ["KV Cache Journal Paper"]
    assert records[0]["provenance"] == ["semantic_scholar"]
    assert all(
        call.args[0] == "http://fake/ss/paper/search/bulk"
        for call in fake_client.get.call_args_list
    )
    fake_client.post.assert_called_once_with(
        "http://fake/ss/author/batch",
        params={"fields": "name,affiliations"},
        headers={},
        timeout=30.0,
        json={"ids": ["author-j-smith"]},
    )
    authority_prompt = prompts[1]
    assert "\"authors\": ['J. Smith']" in authority_prompt
    assert "J. Smith: Famous Lab" in authority_prompt
    assert '"affiliations": "unknown"' not in authority_prompt

    from mourat.database import content_item as ci
    from mourat.database import init_db
    from mourat.database.paper_identity import list_semantic_scholar_ledger_ids

    conn = init_db(dbfile)
    item = ci.get_content_item(conn, "paper_doi-10.1234/journal")
    assert item is not None
    assert item["influence_score"] == 88
    assert list_semantic_scholar_ledger_ids(conn) == {"ss-journal"}
    conn.close()


def test_semantic_scholar_without_affiliations_keeps_unknown_authority_input(
    tmp_path, monkeypatch
):
    prompts = []

    _run_semantic_scholar_script(
        tmp_path,
        monkeypatch,
        author_affiliations=(),
        prompts=prompts,
    )

    assert "\"authors\": ['J. Smith']" in prompts[1]
    assert '"affiliations": "unknown"' in prompts[1]


def test_openalex_affiliations_reach_authority_when_semantic_scholar_is_empty(
    tmp_path, monkeypatch
):
    prompts = []

    _run_semantic_scholar_script(
        tmp_path,
        monkeypatch,
        author_affiliations=(),
        openalex_authorships=[
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "J. Smith",
                },
                "institutions": [{"display_name": "OpenAlex University"}],
                "raw_affiliation_strings": ["OpenAlex Raw Lab"],
            }
        ],
        prompts=prompts,
    )

    authority_prompt = prompts[1]
    assert "\"authors\": ['J. Smith']" in authority_prompt
    assert "J. Smith: OpenAlex University, OpenAlex Raw Lab" in authority_prompt
    assert '"affiliations": "unknown"' not in authority_prompt


def test_both_sources_write_combined_output_with_arxiv_precedence(
    tmp_path, monkeypatch
):
    conn, dbfile = _init_db(tmp_path)
    conn.close()
    cfg = _hydra_cfg(
        tmp_path,
        dbfile,
        overrides=[
            "++sources=[arxiv,semantic_scholar]",
            "db_writer.enabled=true",
            "feeds.cs_lg.api_url=http://fake/rss",
            "semantic_scholar_newest.api_url=http://fake/ss",
            "semantic_scholar_newest.queries=[query]",
            "semantic_scholar_newest.reference_date=2024-02-01",
            "semantic_scholar_newest.regular_delay_seconds=0",
            "semantic_scholar_newest.backoff_seconds=0",
            "semantic_scholar_newest.jitter_seconds=0",
            "monitoring_handler=via_md_files",
            f"result_dir={tmp_path}",
        ],
    )
    scripted = [
        json.dumps(
            {
                "verdicts": [
                    {"id": "paper_0", "relevant": True},
                    {"id": "paper_1", "relevant": False},
                    {"id": "paper_2", "relevant": True},
                ]
            }
        ),
        json.dumps(
            {
                "verdicts": [
                    {"id": "paper_0", "score": 88},
                    {"id": "paper_1", "score": 77},
                ]
            }
        ),
        json.dumps(
            {
                "scores": [
                    {
                        "id": "topic1",
                        "type": "topic",
                        "score": 90,
                        "justification": "on topic",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "scores": [
                    {
                        "id": "topic1",
                        "type": "topic",
                        "score": 80,
                        "justification": "on topic",
                    }
                ]
            }
        ),
    ]
    payload = {
        "data": [
            {
                "paperId": "ss-shared",
                "title": "Semantic Scholar secondary title",
                "url": "https://example.com/shared",
                "abstract": "Secondary abstract",
                "publicationDate": "2024-01-02",
                "authors": [{"name": "J. Smith"}],
                "externalIds": {
                    "ArXiv": "2401.00001",
                    "DOI": "10.1234/shared",
                },
            },
            {
                "paperId": "ss-journal",
                "title": "Semantic Scholar journal",
                "url": "https://example.com/journal",
                "abstract": "Journal abstract",
                "publicationDate": "2024-01-02",
                "authors": [{"name": "A. Author"}],
                "externalIds": {"DOI": "10.1234/journal"},
            },
        ],
        "token": None,
    }
    empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
    fake_client = MagicMock()

    def get_side_effect(url, **kwargs):
        if url == "http://fake/rss":
            return _FakeResponse(SAMPLE_FEED)
        if url == "http://fake/ss/paper/search/bulk":
            return _SemanticScholarResponse(payload)
        if url == "https://arxiv.org/html/2401.00001v1":
            return _FakeResponse(AFFIL_HTML)
        return _FakeResponse(empty_feed)

    fake_client.get.side_effect = get_side_effect
    from mourat.database import init_db

    monkeypatch.setattr(
        "mourat.scripts.collect_newest_papers.create_connection",
        lambda path: init_db(dbfile),
    )
    from types import SimpleNamespace

    import httpx as real_httpx

    monkeypatch.setattr(
        "mourat.scripts.collect_newest_papers.httpx",
        SimpleNamespace(
            Client=lambda **kw: fake_client,
            Timeout=real_httpx.Timeout,
        ),
    )
    openalex = MagicMock()
    openalex.resolve_by_doi.return_value = None
    monkeypatch.setattr(
        "mourat.collectors.arxiv_html_affiliations.OpenAlexClient",
        lambda **kwargs: openalex,
        raising=False,
    )
    import hydra as _hydra

    real_instantiate = _hydra.utils.instantiate

    def smart_instantiate(cfg_node, *args, **kwargs):
        target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
        if target.endswith(("OpenAIChatModel", "OpenAIResponsesModel")):
            return _llm_factory(scripted)
        return real_instantiate(cfg_node, *args, **kwargs)

    monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)
    from mourat.scripts.collect_newest_papers import collect_newest_papers_main

    collect_newest_papers_main(cfg)

    arxiv_monitor = (tmp_path / "step_1-arxiv-cs_lg.md").read_text()
    semantic_monitor = (tmp_path / "step_1-semantic_scholar.md").read_text()
    merge_monitor = (tmp_path / "step_1-merge.md").read_text()
    assert "http://fake/rss" in arxiv_monitor
    assert "# Publication window" in semantic_monitor
    assert "2024-01-02 to 2024-01-02" in semantic_monitor
    assert "# Source merge" in merge_monitor

    records = [
        json.loads(line)
        for line in Path(cfg.jsonl_output_path).read_text().splitlines()
    ]
    assert [record["title"] for record in records] == [
        "KV Cache Trick",
        "Semantic Scholar journal",
    ]
    assert records[0]["provenance"] == ["arxiv", "semantic_scholar"]
    assert records[1]["provenance"] == ["semantic_scholar"]

    from mourat.database import content_item as ci
    from mourat.database.paper_identity import (
        find_paper_content_item_ids,
        list_semantic_scholar_ledger_ids,
    )

    conn = init_db(dbfile)
    assert ci.get_content_item(conn, "paper_arxiv-2401.00001") is not None
    assert ci.get_content_item(conn, "paper_doi-10.1234/journal") is not None
    assert conn.execute("SELECT COUNT(*) FROM content_items").fetchone()[0] == 2
    assert find_paper_content_item_ids(conn, doi="10.1234/shared") == [
        "paper_arxiv-2401.00001"
    ]
    assert list_semantic_scholar_ledger_ids(conn) == {"ss-shared", "ss-journal"}
    conn.close()


def test_failure_before_writers_finish_leaves_ledger_retryable(tmp_path, monkeypatch):
    from mourat.database import init_db
    from mourat.database.paper_identity import list_semantic_scholar_ledger_ids

    with pytest.raises(FileNotFoundError):
        _run_semantic_scholar_script(
            tmp_path,
            monkeypatch,
            output_path=tmp_path / "missing" / "out.jsonl",
        )

    dbfile = next(tmp_path.glob("*.db"))
    conn = init_db(dbfile)
    assert list_semantic_scholar_ledger_ids(conn) == set()
    conn.close()


def test_database_writer_failure_does_not_record_ledger(tmp_path, monkeypatch):
    from mourat.database import init_db
    from mourat.database.paper_identity import list_semantic_scholar_ledger_ids

    with pytest.raises(sqlite3.DatabaseError):
        _run_semantic_scholar_script(tmp_path, monkeypatch, fail_db_write=True)

    dbfile = next(tmp_path.glob("*.db"))
    conn = init_db(dbfile)
    assert list_semantic_scholar_ledger_ids(conn) == set()
    conn.close()


@pytest.mark.parametrize(
    "sources,expected",
    [
        (["arxiv"], ("arxiv",)),
        (["semantic_scholar"], ("semantic_scholar",)),
        (["semantic_scholar", "arxiv"], ("arxiv", "semantic_scholar")),
    ],
)
def test_source_selection_supports_each_enabled_combination(sources, expected):
    from omegaconf import OmegaConf

    from mourat.scripts.collect_newest_papers import _selected_sources

    cfg = OmegaConf.create({"sources": sources})
    assert _selected_sources(cfg) == expected


@pytest.mark.parametrize("sources", [[], ["unknown"], ["arxiv", "arxiv"]])
def test_source_selection_rejects_invalid_values(sources):
    from omegaconf import OmegaConf

    from mourat.scripts.collect_newest_papers import _selected_sources

    with pytest.raises(ValueError):
        _selected_sources(OmegaConf.create({"sources": sources}))


def test_affiliation_configuration_forwards_both_provider_transports(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-test-key")
    conn, dbfile = _init_db(tmp_path)
    conn.close()

    cfg = _hydra_cfg(
        tmp_path,
        dbfile,
        overrides=[
            "semantic_scholar_newest.api_url=http://fake/ss",
            "semantic_scholar_newest.timeout_seconds=12.5",
            "semantic_scholar_newest.max_retries=7",
            "semantic_scholar_newest.backoff_seconds=3.0",
            "semantic_scholar_newest.regular_delay_seconds=0.0",
            "semantic_scholar_newest.jitter_seconds=0.25",
            "++arxiv_html_affiliations.openalex_enabled=false",
            "++arxiv_html_affiliations.openalex_user_agent=mourat-test/0.1",
            "++arxiv_html_affiliations.openalex_timeout_seconds=14.5",
            "++arxiv_html_affiliations.openalex_max_retries=6",
            "++arxiv_html_affiliations.openalex_backoff_seconds=4.0",
            "++arxiv_html_affiliations.openalex_regular_delay_seconds=1.5",
        ],
    )

    affiliation = cfg.arxiv_html_affiliations
    assert affiliation.semantic_scholar_api_url == "http://fake/ss"
    assert affiliation.semantic_scholar_api_key == "test-key"
    assert affiliation.semantic_scholar_proxy == "http://proxy.example:8080"
    assert affiliation.semantic_scholar_timeout_seconds == 12.5
    assert affiliation.semantic_scholar_max_retries == 7
    assert affiliation.semantic_scholar_backoff_seconds == 3.0
    assert affiliation.semantic_scholar_regular_delay_seconds == 0.0
    assert affiliation.semantic_scholar_jitter_seconds == 0.25
    assert affiliation.openalex_enabled is False
    assert affiliation.openalex_user_agent == "mourat-test/0.1"
    assert affiliation.openalex_api_key == "openalex-test-key"
    assert affiliation.openalex_timeout_seconds == 14.5
    assert affiliation.openalex_max_retries == 6
    assert affiliation.openalex_backoff_seconds == 4.0
    assert affiliation.openalex_regular_delay_seconds == 1.5


def test_default_configuration_selects_both_sources(tmp_path):
    from hydra import compose, initialize_config_dir

    conn, dbfile = _init_db(tmp_path)
    conn.close()
    config_dir = Path(REPO) / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config_collect_newest_papers",
            overrides=[
                f"++db_path={dbfile}",
                "research_topic_ids=[topic1]",
                "monitoring_handler=dummy_test",
            ],
        )
    assert list(cfg.sources) == ["arxiv", "semantic_scholar"]
    assert cfg.semantic_scholar_newest.lag_days == 30
    assert cfg.semantic_scholar_newest.stride_days == 1
    assert cfg.semantic_scholar_newest.backfill_days == 0
    affiliation = cfg.arxiv_html_affiliations
    assert affiliation.openalex_enabled is True
    assert affiliation.openalex_user_agent == "mourat/0.1 (mailto:contact@example.com)"
    assert affiliation.openalex_timeout_seconds == 30.0
    assert affiliation.openalex_max_retries == 4
    assert affiliation.openalex_backoff_seconds == 2.0
    assert affiliation.openalex_regular_delay_seconds == 2.0
    assert cfg.semantic_scholar_newest.max_results_per_query is None
