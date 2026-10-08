"""Helpers for handling Hydra configuration values."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from omegaconf import DictConfig, OmegaConf


@contextmanager
def open_dict(cfg: DictConfig) -> Iterator[DictConfig]:
    """Temporarily disable struct mode to permit deleting configured keys."""
    was_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        yield cfg
    finally:
        OmegaConf.set_struct(cfg, was_struct)


def read_enabled(writer_cfg: DictConfig) -> bool:
    """Read the writer flag and remove it before Hydra instantiates the writer."""
    enabled = bool(writer_cfg.get("enabled", False))
    if enabled and "enabled" in writer_cfg:
        with open_dict(writer_cfg):
            del writer_cfg["enabled"]
    return enabled
