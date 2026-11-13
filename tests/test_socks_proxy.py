"""Regression tests for HTTPX SOCKS proxy support (spec 34)."""

import tomllib
from pathlib import Path

import httpx
import pytest


def test_runtime_dependencies_include_httpx_socks_extra():
    project_root = Path(__file__).resolve().parent.parent
    with (project_root / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    requirement_dependencies = [
        line.split("#", 1)[0].strip()
        for line in (project_root / "requirements.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]

    for dependencies in (project_dependencies, requirement_dependencies):
        assert "httpx[socks]" in dependencies
        assert "httpx" not in dependencies


@pytest.mark.parametrize("scheme", ["socks5", "socks5h"])
def test_newest_paper_client_constructs_with_socks_proxy(scheme):
    from mourat.scripts.collect_newest_papers import _make_http_client

    with _make_http_client(proxy=f"{scheme}://127.0.0.1:1080") as client:
        assert isinstance(client, httpx.Client)
