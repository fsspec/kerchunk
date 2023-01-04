import fastparquet
import fsspec

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
        ".zgroup": b"i exist",
    }
    m = fsspec.filesystem("memory")
    refs_to_dataframe(refs, "memory://outdir", partition=True, template_length=1)
    with fsspec.open("memory:///outdir/metadata.parq") as f:
        pf0 = fastparquet.ParquetFile(f)
        df0 = pf0.to_pandas()
    with fsspec.open("memory:///outdir/a.parq") as f:
        pf1 = fastparquet.ParquetFile(f)
        df1 = pf1.to_pandas()
    assert df0.to_dict() == {
        "key": {0: ".zgroup"},
        "path": {0: None},
        "offset": {0: 0},
        "size": {0: 0},
        "raw": {0: b"i exist"},
    }
    assert df1.to_dict() == {
        "key": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"},
        "path": {
            0: "{u}1.file",
            1: "{u}1.file",
            2: "{u}2.file",
            3: "{u}3.file",
            4: "{u}4.file",
            5: "{u}5.file",
            6: None,
        },
        "offset": {0: 0, 1: 10, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "size": {0: 0, 1: 100, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "raw": {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: b"data"},
    }
    assert pf1.key_value_metadata["u"] == "memory://url"
    assert "u" not in pf1.key_value_metadata
