import pytest

slicer = pytest.importorskip("slicer")


def test_slicer_present():
    assert hasattr(slicer, "mrmlScene")
