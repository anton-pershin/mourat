"""Tests for arXiv client configuration and protected transport."""

from pathlib import Path

import pytest
from comourat.clients.arxiv import ArxivClient
from comourat.fetch import SafeHttpClient
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PAPER_CONFIGS = [
    "config_collect_influential_papers_from_scratch",
    "config_collect_influential_papers_from_seeds",
]
EXPLICIT_PROXY = "socks5h://user:pass@arxiv-proxy.example:1080"


def _compose(config_name: str):
    configs_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        return compose(config_name=config_name)


def _set_generic_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://generic-http.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://generic-https.example:8080")


@pytest.mark.parametrize("config_name", PAPER_CONFIGS)
def test_arxiv_config_uses_explicit_proxy_and_operation_deadline(
    config_name: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ARXIV_PROXY", EXPLICIT_PROXY)
    _set_generic_proxies(monkeypatch)

    cfg = _compose(config_name)
    client_config = cfg.arxiv_pdf_verifier.arxiv_client
    assert client_config.proxy == EXPLICIT_PROXY
    assert client_config.timeout_seconds == 120.0
    assert client_config.overall_timeout_seconds == 120.0

    client = instantiate(client_config)
    assert isinstance(client, ArxivClient)
    assert isinstance(client._safe_client, SafeHttpClient)
    assert client._safe_client.session.proxies == {
        "http": EXPLICIT_PROXY,
        "https": EXPLICIT_PROXY,
    }
    assert client._safe_client.policy.connection_timeout == 120.0
    assert client._safe_client.policy.read_timeout == 120.0
    assert client._safe_client.policy.overall_timeout == 120.0


@pytest.mark.parametrize("config_name", PAPER_CONFIGS)
def test_arxiv_config_does_not_use_generic_environment_proxies(
    config_name: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ARXIV_PROXY", raising=False)
    _set_generic_proxies(monkeypatch)

    cfg = _compose(config_name)
    assert cfg.arxiv_pdf_verifier.arxiv_client.proxy is None
