"""Utility functions to ensure third party dependencies are available."""

import importlib.util
import logging
import subprocess
import sys
from typing import Iterable


def install_package_if_missing(package_name: str) -> None:
    """Install *package_name* via ``pip`` if it cannot be imported."""
    if importlib.util.find_spec(package_name) is None:
        logging.info(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def check_and_install_dependencies(packages: Iterable[str]) -> None:
    """Ensure that all packages from *packages* are installed."""
    for package in packages:
        install_package_if_missing(package)
