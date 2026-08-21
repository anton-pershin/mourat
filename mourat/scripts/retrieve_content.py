import hydra
from omegaconf import DictConfig

from mourat.utils.common import get_config_path

CONFIG_NAME = "config_retrieve_content"


def retrieve_content(cfg: DictConfig) -> None:
    print("retrieve_content")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(retrieve_content)()
