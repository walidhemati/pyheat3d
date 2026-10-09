import pytest

from pyheat3d.registry import cuda_available


def pytest_collection_modifyitems(config, items):
    if cuda_available():
        return
    skip = pytest.mark.skip(reason="no CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
