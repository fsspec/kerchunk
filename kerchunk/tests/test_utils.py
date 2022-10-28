import fsspec
import kerchunk.utils
import zarr


def test_rename():

    old = {"version": 1, "refs": {"v0": ["oldpath", 0, 0], "bin": "data"}}
    new = kerchunk.utils.rename_target(old, {"oldpath": "newpath"})
    assert new == {"version": 1, "refs": {"v0": ["newpath", 0, 0], "bin": "data"}}


def test_rename_files(m):
    m.pipe(
        "ref.json", b'{"version": 1, "refs": {"v0": ["oldpath", 0, 0], "bin": "data"}}'
    )
    kerchunk.utils.rename_target_files("memory://ref.json", {"oldpath": "newpath"})
    out = m.cat("ref.json")
    assert out == b'{"version":1,"refs":{"v0":["newpath",0,0],"bin":"data"}}'
    kerchunk.utils.rename_target_files(
        "memory://ref.json", {"newpath": "newerpath"}, url_out="memory://ref2.json"
    )
    out = m.cat("ref.json")
    assert out == b'{"version":1,"refs":{"v0":["newpath",0,0],"bin":"data"}}'
    out = m.cat("ref2.json")
    assert out == b'{"version":1,"refs":{"v0":["newerpath",0,0],"bin":"data"}}'


def test_inline(m):
    m.pipe("data", b"stuff")
    refs = {
        "key0": b"00",
        "key1": ["memory://data"],
        "key2": ["memory://data", 1, 1],
        "key1": ["memory://data", 2, 4],
    }
    out = kerchunk.utils._do_inline(refs, 2)
    expected = {
        "key0": b"00",
        "key1": ["memory://data"],
        "key2": b"t",
        "key1": ["memory://data", 2, 4],
    }
    assert out == expected


def test_inline_array():
    refs = {
        ".zgroup": b'{"zarr_format": 2}',
        "data/.zarray": """
    {
    "chunks": [
        1
    ],
    "compressor": null,
    "dtype": "<i4",
    "fill_value": 0,
    "filters": null,
    "order": "C",
    "shape": [
        2
    ],
    "zarr_format": 2
}
""",
        "data/0": b"\x01\x00\x00\x00",
        "data/1": b"\x02\x00\x00\x00",
    }
    fs = fsspec.filesystem("reference", fo=refs)
    out1 = kerchunk.utils.inline_array(refs, threshold=1000)  # does nothing
    assert out1 == refs
    out2 = kerchunk.utils.inline_array(refs, threshold=1000, names=["data"])  # explicit
    assert "data/1" not in out2
    fs = fsspec.filesystem("reference", fo=out2)
    g = zarr.open(fs.get_mapper())
    assert g.data[:].tolist() == [1, 2]

    out3 = kerchunk.utils.inline_array(refs, threshold=1)  # inlines because of size
    assert "data/1" not in out3
    fs = fsspec.filesystem("reference", fo=out3)
    g = zarr.open(fs.get_mapper())
    assert g.data[:].tolist() == [1, 2]
