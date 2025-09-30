from typing import Any

from omegaconf import DictConfig

import hydra
from mourat.pipeline import Pipeline
from mourat.utils.common import get_config_path

CONFIG_NAME = "config_run_pipeline"


def run_pipeline(cfg: DictConfig) -> None:
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    data: dict[str, Any] = {}
    data = pipeline(data)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(run_pipeline)()
