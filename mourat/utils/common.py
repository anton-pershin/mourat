from pathlib import Path
from textwrap import dedent

from mourat.data_models import PaperInfo, ScoredPaperInfo


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    return get_project_path() / "config"


def normalize_author_name(s: str) -> str:
    names = s.split(" ")
    return " ".join([n for n in names if "." not in n])


def to_text_description(template: str, paper_info: PaperInfo | ScoredPaperInfo) -> str:
    return dedent(template).format(**(paper_info.model_dump())) + "\n"
