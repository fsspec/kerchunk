import pandas as pd
import pytest

fastparquet = pytest.importorskip("fastparquet")
import fsspec
import ujson

from kerchunk.df import refs_to_dataframe


def test_1():
    refs = {
        "a/0": ["memory://url1.file"],
        "a/1": ["memory://url1.file", 10, 100],
        "a/2": ["memory://url2.file"],
        "a/3": ["memory://url3.file"],
        "a/4": ["memory://url4.file"],
        "a/5": ["memory://url5.file"],
        "a/6": b"data",
        "a/.zarray": b"""{"shape": [7], "chunks":[1], "filters": [], "compression": null}""",
        ".zgroup": b'{"zarr_format": 2}',
    }
    refs_to_dataframe(refs, "memory://outdir", record_size=4)
    with fsspec.open("memory:///outdir/.zmetadata") as f:
        meta = ujson.load(f)
        assert list(meta) == ["metadata", "zarr_consolidated_format", "record_size"]
        assert meta["record_size"] == 4

    df0 = pd.read_parquet("memory:///outdir/a/refs.0.parq")

    # no raw
    assert df0.to_dict() == {
        "offset": {0: 0, 1: 10, 2: 0, 3: 0},
        "path": {
            0: "memory://url1.file",
            1: "memory://url1.file",
            2: "memory://url2.file",
            3: "memory://url3.file",
        },
        "size": {0: 0, 1: 100, 2: 0, 3: 0},
    }

    # with raw column
    df1 = pd.read_parquet("memory:///outdir/a/refs.1.parq")
    assert df1.to_dict() == {
        "offset": {0: 0, 1: 0, 2: 0},
        "path": {0: "memory://url4.file", 1: "memory://url5.file", 2: None},
        "raw": {0: None, 1: None, 2: b"data"},
        "size": {0: 0, 1: 0, 2: 0},
    }
