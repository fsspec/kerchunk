import io

import fsspec
import json
import kerchunk.utils
import kerchunk.zarr
import numpy as np
import pytest
import zarr

from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper


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
    out1 = kerchunk.utils.inline_array(refs, threshold=1)  # does nothing
    assert out1 == refs
    out2 = kerchunk.utils.inline_array(refs, threshold=1, names=["data"])  # explicit
    assert json.loads(out2["data/.zattrs"]) == json.loads(refs["data/.zattrs"])

    localfs = fsspec.filesystem("file")
    store = kerchunk.utils.refs_as_store(out2, fs=localfs)
    g = zarr.open(store, mode="r", zarr_format=2)
    assert g["data"][:].tolist() == [1, 2]  # What is g.data???

    out3 = kerchunk.utils.inline_array(refs, threshold=1000)  # inlines because of size
    store = kerchunk.utils.refs_as_store(out3, localfs)
    g = zarr.open(store, mode="r", zarr_format=2)
    assert g["data"][:].tolist() == [1, 2]  # What is g.data???


def test_json():
    data = {"a": "a", "b": b"b", "c": [None, None, None], "d": '{"key": 0}'}
    out = kerchunk.utils._encode_for_JSON(data)
    expected = {"a": "a", "b": "b", "c": [None, None, None], "d": '{"key":0}'}
    assert out == expected


def _subchunk_refs(shape, chunks):
    g = zarr.open_group("memory://test.zarr", mode="w", zarr_format=2)
    data = np.arange(np.prod(shape)).reshape(shape)
    arr = g.create_array(
        "data", dtype=data.dtype, shape=data.shape, chunks=chunks, compressor=None
    )
    arr[:] = data
    ref = kerchunk.zarr.single_zarr("memory://test.zarr")["refs"]
    return ref, data


def _chunk_keys(ref):
    return sorted(
        k for k in ref if k.startswith("data/") and not k.startswith("data/.z")
    )


@pytest.mark.parametrize(
    "chunks, factor, expected",
    [
        ([10, 10], 5, [2, 10]),
        ([5, 10], 5, [1, 10]),
        # factor larger than the leading axis: it is split into ones and the
        # rest of the factor applies to the next axis
        ([2, 10], 4, [1, 5]),
        ([10, 10], 20, [1, 5]),
        ([5, 10], 25, [1, 2]),
        ([10, 10], 100, [1, 1]),
    ],
)
def test_subchunk_exact(m, chunks, factor, expected):
    ref, data = _subchunk_refs((10, 10), chunks)

    ref2 = list(_ for _ in ref if not _.endswith("zattrs"))  # ignore empty attrs
    assert ref2 == [".zgroup", "data/.zarray"] + [
        f"data/{_}.0" for _ in range(10 // chunks[0])
    ]

    out = kerchunk.utils.subchunk(ref, "data", factor)
    assert json.loads(out["data/.zarray"])["chunks"] == expected
    nchunk = [10 // _ for _ in expected]
    assert _chunk_keys(out) == sorted(
        f"data/{i}.{j}" for i in range(nchunk[0]) for j in range(nchunk[1])
    )
    assert len(_chunk_keys(out)) == factor * len(_chunk_keys(ref))

    store = kerchunk.utils.refs_as_store(out, remote_protocol="memory")
    g2 = zarr.open_group(store, mode="r", zarr_format=2)

    # g2 = zarr.open_group(
    #     "reference://", storage_options={"fo": out, "remote_protocol": "memory"}, zarr_format=2
    # )
    assert (g2["data"][:] == data).all()


@pytest.mark.parametrize(
    "factor, expected",
    [(2, [1, 3, 8]), (6, [1, 1, 8]), (12, [1, 1, 4]), (48, [1, 1, 1])],
)
def test_subchunk_multiple_axes(m, factor, expected):
    ref, data = _subchunk_refs((4, 3, 8), [2, 3, 8])
    assert len(_chunk_keys(ref)) == 2

    out = kerchunk.utils.subchunk(ref, "data", factor)
    assert json.loads(out["data/.zarray"])["chunks"] == expected
    assert len(_chunk_keys(out)) == 2 * factor

    store = kerchunk.utils.refs_as_store(out, remote_protocol="memory")
    g2 = zarr.open_group(store, mode="r", zarr_format=2)
    assert (g2["data"][:] == data).all()


@pytest.mark.parametrize(
    "chunks, factor",
    [
        # 2 does not divide the leading axis, even though it divides the next
        ([5, 10], 2),
        # 4 = 2 * 2, but 2 neither divides nor is a multiple of the next axis
        ([2, 3, 8], 4),
        # more sub-chunks than there are elements in a chunk
        ([2, 10], 40),
        ([1, 1], 2),
        ([10, 10], 0),
    ],
)
def test_subchunk_impossible(m, chunks, factor):
    ref, _ = _subchunk_refs((10, 10) if len(chunks) == 2 else (4, 3, 8), chunks)
    with pytest.raises(ValueError):
        kerchunk.utils.subchunk(ref, "data", factor)


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


def test_encode_fill_value():
    assert kerchunk.utils.encode_fill_value(np.array([9999]), np.dtype("int")) == 9999
    assert kerchunk.utils.encode_fill_value(np.array(9999), np.dtype("int")) == 9999
    assert kerchunk.utils.encode_fill_value([9999], np.dtype("int")) == 9999
    assert kerchunk.utils.encode_fill_value(9999, np.dtype("int")) == 9999
