import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

slicer = pytest.importorskip("slicer")


def test_module_present():
    assert hasattr(slicer, "modules")
