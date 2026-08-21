import hydra
from omegaconf import DictConfig

from mourat.utils.common import get_config_path

CONFIG_NAME = "config_update_database"


def update_database(cfg: DictConfig) -> None:
    print("update_database")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(update_database)()
