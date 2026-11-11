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
    source_defaults = (
        []
        if any(item.startswith("sources=") for item in overrides)
        else ["sources=[arxiv]"]
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config_collect_newest_papers",
            overrides=[
                f"++db_path={dbfile}",
                f"++jsonl_output_path={tmp_path}/out.jsonl",
                "research_topic_ids=[topic1]",
                "monitoring_handler=dummy_test",
                *source_defaults,
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
    def __init__(self, text, status_code=200, headers=None):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)

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


def _llm_factory(scripted):
    """FunctionModel factory that pops the next scripted JSON per call."""

    def model_fn(messages, agent):
        return ModelResponse(parts=[TextPart(scripted.pop(0))])

    return FunctionModel(model_fn)


def _run_newest_sources(
    tmp_path, monkeypatch, fail_writer=False, semantic_scholar_only=False
):
    """Run the real pipeline with mocked providers and scripted LLMs."""
    conn, dbfile = _init_db(tmp_path)
    conn.close()
    output_path = (
        f"{tmp_path}/missing/out.jsonl" if fail_writer else f"{tmp_path}/both.jsonl"
    )
    source_override = (
        "sources=[semantic_scholar]"
        if semantic_scholar_only
        else "sources=[arxiv,semantic_scholar]"
    )
    cfg = _hydra_cfg(
        tmp_path,
        dbfile,
        overrides=[
            source_override,
            "db_writer.enabled=true",
            "feeds.cs_lg.api_url=http://fake/rss",
            "semantic_scholar_newest.run_date=2026-01-31",
            "semantic_scholar_newest.publication_lag_days=1",
            "semantic_scholar_newest.backfill_days=0",
            "semantic_scholar_newest.regular_delay_seconds=0",
            "semantic_scholar_newest.backoff_seconds=0",
            "semantic_scholar_newest.queries=[machine_learning]",
            f"jsonl_output_path={output_path}",
        ],
    )
    triage_verdicts = (
        [
            {"id": "paper_0", "relevant": True},
            {"id": "paper_1", "relevant": True},
        ]
        if semantic_scholar_only
        else [
            {"id": "paper_0", "relevant": True},
            {"id": "paper_1", "relevant": False},
            {"id": "paper_2", "relevant": True},
        ]
    )
    scripted = [
        json.dumps({"verdicts": triage_verdicts}),
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
                        "score": 85,
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
    s2_payload = {
        "data": [
            {
                "paperId": "s2-shared",
                "title": "Semantic Scholar duplicate title",
                "url": "https://www.semanticscholar.org/paper/s2-shared",
                "abstract": "Secondary metadata must not replace arXiv",
                "publicationDate": "2026-01-30",
                "authors": [{"name": "J. Smith"}],
                "externalIds": {
                    "ArXiv": "2401.00001v2",
                    "DOI": "10.1000/shared",
                },
            },
            {
                "paperId": "s2-journal",
                "title": "Journal Only KV Cache",
                "url": "https://www.semanticscholar.org/paper/s2-journal",
                "abstract": "A journal paper about KV cache optimization",
                "publicationDate": "2026-01-30",
                "authors": [{"name": "J. Journal"}],
                "externalIds": {"DOI": "10.1000/journal"},
            },
            {
                "paperId": "s2-no-doi",
                "title": "No DOI means no candidate",
                "url": "https://www.semanticscholar.org/paper/s2-no-doi",
                "abstract": "Excluded before triage",
                "publicationDate": "2026-01-30",
                "authors": [],
                "externalIds": {},
            },
        ],
        "token": None,
    }
    fake_client = MagicMock()

    def get_side_effect(url, **kwargs):
        if url == "http://fake/rss":
            return _FakeResponse(SAMPLE_FEED)
        if url.endswith("/paper/search/bulk"):
            return _FakeResponse(json.dumps(s2_payload))
        return _FakeResponse(AFFIL_HTML)

    fake_client.get.side_effect = get_side_effect
    from types import SimpleNamespace

    import httpx as real_httpx

    monkeypatch.setattr(
        "mourat.scripts.collect_newest_papers.httpx",
        SimpleNamespace(Client=lambda **kw: fake_client, Timeout=real_httpx.Timeout),
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
    return dbfile, output_path, fake_client


class TestScriptWiring:
    def test_both_sources_deduplicate_output_and_commit_state(
        self, tmp_path, monkeypatch
    ):
        dbfile, output_path, _ = _run_newest_sources(tmp_path, monkeypatch)

        records = [
            json.loads(line) for line in Path(output_path).read_text().splitlines()
        ]
        assert [record["title"] for record in records] == [
            "KV Cache Trick",
            "Journal Only KV Cache",
        ]
        assert records[0]["provenance"] == ["arxiv_rss", "semantic_scholar"]
        assert records[1]["provenance"] == ["semantic_scholar"]

        from mourat.database import content_item as ci
        from mourat.database import init_db
        from mourat.database import semantic_scholar as s2db

        conn = init_db(dbfile)
        assert ci.get_content_item(conn, "paper_arxiv-2401.00001") is not None
        assert ci.get_content_item(conn, "paper_doi-10.1000/journal") is not None
        assert ci.get_content_item(conn, "paper_doi-10.1000/shared") is None
        assert s2db.get_seen_paper_ids(
            conn, ["s2-shared", "s2-journal", "s2-no-doi"]
        ) == {"s2-shared", "s2-journal", "s2-no-doi"}
        state = s2db.load_query_state(conn, "machine_learning")
        assert state["last_completed_end_date"] == "2026-01-30"
        conn.close()

    def test_semantic_scholar_only_never_requests_arxiv_feeds(
        self, tmp_path, monkeypatch
    ):
        _, output_path, fake_client = _run_newest_sources(
            tmp_path, monkeypatch, semantic_scholar_only=True
        )

        records = [
            json.loads(line) for line in Path(output_path).read_text().splitlines()
        ]
        assert [record["title"] for record in records] == [
            "Semantic Scholar duplicate title",
            "Journal Only KV Cache",
        ]
        assert all(record["provenance"] == ["semantic_scholar"] for record in records)
        assert all(
            call.args[0] != "http://fake/rss" for call in fake_client.get.call_args_list
        )

    def test_downstream_failure_does_not_commit_semantic_scholar_state(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(FileNotFoundError):
            _run_newest_sources(tmp_path, monkeypatch, fail_writer=True)

        from mourat.database import init_db
        from mourat.database import semantic_scholar as s2db

        dbfile = next(tmp_path.glob("*.db"))
        conn = init_db(dbfile)
        assert (
            s2db.get_seen_paper_ids(conn, ["s2-shared", "s2-journal", "s2-no-doi"])
            == set()
        )
        assert s2db.load_query_state(conn, "machine_learning") is None
        conn.close()

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
