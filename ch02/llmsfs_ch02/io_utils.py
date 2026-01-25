from __future__ import annotations

from pathlib import Path


def chapter_root() -> Path:
    """
    Returns the chapter root folder (ch02/), regardless of current working directory.
    """
    return Path(__file__).resolve().parents[1]


def data_path(filename: str) -> Path:
    """
    Returns a path inside ch02/ (e.g., ch02/the-verdict.txt).
    """
    return chapter_root() / filename


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)