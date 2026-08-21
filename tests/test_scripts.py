"""Smoke tests for CLI scripts. Each script is invoked via subprocess with Hydra config override to use the local (offline) LLM model."""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_script(script_name: str, config_name: str) -> subprocess.CompletedProcess:
    """Run a Hydra script with minimal overrides."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            f"mourat.scripts.{script_name}",
            f"--config-name={config_name}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestStubScripts:
    """Stub scripts should exit cleanly with just a config load."""

    def test_retrieve_content(self):
        result = _run_script("retrieve_content", "config_retrieve_content")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_update_database(self):
        result = _run_script("update_database", "config_update_database")
        assert result.returncode == 0, f"stderr: {result.stderr}"
