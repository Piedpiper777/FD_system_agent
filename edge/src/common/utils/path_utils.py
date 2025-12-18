"""Utility helpers for handling workspace-relative paths.

All persisted metadata should store paths relative to the Edge project root so
that files remain valid after moving to new hardware or directories.
"""
from pathlib import Path
from typing import Union

# Edge project root (directory that contains data/, models/, etc.)
EDGE_ROOT = Path(__file__).resolve().parents[3]
_EDGE_ROOT_RESOLVED = EDGE_ROOT.resolve()


def _as_path(value: Union[str, Path]) -> Path:
    if isinstance(value, Path):
        return value
    if value is None:
        return Path()
    return Path(str(value))


def to_relative_path(path: Union[str, Path]) -> str:
    """Convert a filesystem path to a project-relative POSIX string."""
    path_obj = _as_path(path)
    if not path_obj:
        return ''

    # Try strict relative conversions in order of preference.
    try:
        rel = path_obj.resolve().relative_to(_EDGE_ROOT_RESOLVED)
        return rel.as_posix()
    except Exception:
        try:
            rel = path_obj.relative_to(EDGE_ROOT)
            return rel.as_posix()
        except Exception:
            return path_obj.as_posix()


def resolve_edge_path(path_str: str) -> Path:
    """Resolve a potentially relative path to an absolute path under EDGE_ROOT."""
    if not path_str:
        raise ValueError('路径不能为空')

    path_obj = _as_path(path_str)
    if not path_obj.is_absolute():
        path_obj = EDGE_ROOT / path_obj

    resolved = path_obj.resolve()
    try:
        resolved.relative_to(_EDGE_ROOT_RESOLVED)
    except ValueError as exc:
        raise ValueError('路径超出工作区范围') from exc

    return resolved
