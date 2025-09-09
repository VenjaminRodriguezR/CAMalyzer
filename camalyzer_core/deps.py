"""Optional dependency checking for CAMalyzer."""

import importlib.util

OPTIONAL_DEPS = ["torch", "monai", "open3d"]


def check_optional_deps() -> list:
    """Return a list of missing optional dependency module names."""
    missing = [name for name in OPTIONAL_DEPS if importlib.util.find_spec(name) is None]
    return missing
