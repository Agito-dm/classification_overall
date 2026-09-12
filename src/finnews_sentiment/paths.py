from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find repository root by looking for pyproject.toml."""
    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise RuntimeError(
        "Could not find project root containing pyproject.toml."
    )


def data_dir() -> Path:
    return find_project_root() / "data"


def reports_dir() -> Path:
    return find_project_root() / "reports"