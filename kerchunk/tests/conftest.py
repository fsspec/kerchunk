import fsspec
import pytest


@pytest.fixture()
def m():
    fs = fsspec.filesystem("memory")
    yield fs
    fs.store.clear()
    del fs.pseudo_dirs[1:]
