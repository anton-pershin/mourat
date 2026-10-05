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
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>KV Cache Trick</title>
    <link href="http://arxiv.org/abs/2401.00001"/>
    <description>arXiv:2401.00001v1 [cs.LG] KV Cache Trick
Abstract: A method for KV cache optimization</description>
    <author><name>J. Smith</name></author>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
  <entry>
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
        fake_client.get.side_effect = lambda url: (
            _FakeResponse(SAMPLE_FEED)
            if url == "http://fake/rss"
            else _FakeResponse(AFFIL_HTML)
        )

        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx.Client",
            lambda **kw: fake_client,
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
            if target == "pydantic_ai.models.openai.OpenAIChatModel":
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
        fake_client.get.side_effect = lambda url: (
            _FakeResponse(SAMPLE_FEED)
            if url == "http://fake/rss"
            else _FakeResponse(AFFIL_HTML)
        )
        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx.Client",
            lambda **kw: fake_client,
        )

        import hydra as _hydra

        real_instantiate = _hydra.utils.instantiate

        def smart_instantiate(cfg_node, *args, **kwargs):
            target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
            if target == "pydantic_ai.models.openai.OpenAIChatModel":
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
        fake_client.get.side_effect = lambda url: (
            _FakeResponse(SAMPLE_FEED)
            if url == "http://fake/rss"
            else _FakeResponse(AFFIL_HTML)
        )
        monkeypatch.setattr(
            "mourat.scripts.collect_newest_papers.httpx.Client",
            lambda **kw: fake_client,
        )

        import hydra as _hydra

        real_instantiate = _hydra.utils.instantiate

        def smart_instantiate(cfg_node, *args, **kwargs):
            target = cfg_node.get("_target_", "") if hasattr(cfg_node, "get") else ""
            if target == "pydantic_ai.models.openai.OpenAIChatModel":
                return _llm_factory(scripted)
            return real_instantiate(cfg_node, *args, **kwargs)

        monkeypatch.setattr(_hydra.utils, "instantiate", smart_instantiate)

        from mourat.scripts.collect_newest_papers import collect_newest_papers_main

        collect_newest_papers_main(cfg)
        out = Path(cfg.jsonl_output_path)
        assert out.exists()
        assert len(out.read_text().strip().splitlines()) == 1
