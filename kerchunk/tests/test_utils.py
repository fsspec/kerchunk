import io

import fsspec
import kerchunk.utils
import kerchunk.zarr
import numpy as np
import pytest
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
    out = kerchunk.utils.do_inline(refs, 2)
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
        "data/.zattrs": '{"foo": "bar"}',
    }
    fs = fsspec.filesystem("reference", fo=refs)
    out1 = kerchunk.utils.inline_array(refs, threshold=1000)  # does nothing
    assert out1 == refs
    out2 = kerchunk.utils.inline_array(refs, threshold=1000, names=["data"])  # explicit
    assert "data/1" not in out2
    assert out2["data/.zattrs"] == refs["data/.zattrs"]
    fs = fsspec.filesystem("reference", fo=out2)
    g = zarr.open(fs.get_mapper())
    assert g.data[:].tolist() == [1, 2]

    out3 = kerchunk.utils.inline_array(refs, threshold=1)  # inlines because of size
    assert "data/1" not in out3
    fs = fsspec.filesystem("reference", fo=out3)
    g = zarr.open(fs.get_mapper())
    assert g.data[:].tolist() == [1, 2]


def test_json():
    data = {"a": "a", "b": b"b", "c": [None, None, None], "d": '{"key": 0}'}
    out = kerchunk.utils._encode_for_JSON(data)
    expected = {"a": "a", "b": "b", "c": [None, None, None], "d": '{"key":0}'}
    assert out == expected


@pytest.mark.parametrize("chunks", [[10, 10], [5, 10]])
def test_subchunk_exact(m, chunks):
    store = m.get_mapper("test.zarr")
    g = zarr.open_group(store, mode="w")
    data = np.arange(100).reshape(10, 10)
    arr = g.create_dataset("data", data=data, chunks=chunks, compression=None)
    ref = kerchunk.zarr.single_zarr("memory://test.zarr")

    extra = [] if chunks[0] == 10 else ["data/1.0"]
    assert list(ref) == [".zgroup", "data/.zarray", "data/0.0"] + extra

    out = kerchunk.utils.subchunk(ref, "data", 5)
    nchunk = 10 // chunks[0] * 5
    assert list(out) == [".zgroup", "data/.zarray"] + [
        f"data/{_}.0" for _ in range(nchunk)
    ]

    g2 = zarr.open_group(
        "reference://", storage_options={"fo": out, "remote_protocol": "memory"}
    )
    assert (g2.data[:] == data).all()


@pytest.mark.parametrize("archive", ["zip", "tar"])
def test_archive(m, archive):
    import zipfile
    import tarfile

    data = b"piece of data"
    with fsspec.open("memory://archive", "wb") as f:
        if archive == "zip":
            arc = zipfile.ZipFile(file=f, mode="w")
            arc.writestr("data1", data)
            arc.close()
        else:
            arc = tarfile.TarFile(fileobj=f, mode="w")
            ti = tarfile.TarInfo("data1")
            ti.size = len(data)
            arc.addfile(ti, io.BytesIO(data))
            arc.close()
    refs = {
        "a": b"stuff",
        "b": [f"{archive}://data1::memory://archive"],
        "c": [f"{archive}://data1::memory://archive", 5, 2],
    }

    refs2 = kerchunk.utils.dereference_archives(refs)

    fs = fsspec.filesystem("reference", fo=refs2)
    assert fs.cat("a") == b"stuff"
    assert fs.cat("b") == data
    assert fs.cat("c") == data[5:7]


def test_deflate_zip_archive(m):
    import zipfile
    from kerchunk.codecs import DeflateCodec

    dec = DeflateCodec()

    data = b"piece of data"
    with fsspec.open("memory://archive", "wb") as f:
        arc = zipfile.ZipFile(file=f, mode="w", compression=zipfile.ZIP_DEFLATED)
        arc.writestr("data1", data)
        arc.close()
    refs = {
        "b": [f"zip://data1::memory://archive"],
    }

    with pytest.warns(UserWarning):
        refs2 = kerchunk.utils.dereference_archives(refs)

    fs = fsspec.filesystem("reference", fo=refs2)
    assert dec.decode(fs.cat("b")) == data
