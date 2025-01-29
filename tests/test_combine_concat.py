import pytest

import fsspec
import numpy as np
import zarr

import kerchunk.combine
import kerchunk.zarr
import kerchunk.df
from kerchunk.utils import fs_as_store, refs_as_store


@pytest.mark.parametrize(
    "arrays,chunks,axis",
    [
        (
            [np.arange(10), np.arange(10, 20)],
            (2,),
            0,
        ),
        (
            [np.arange(12), np.arange(12, 36), np.arange(36, 42)],
            (6,),
            0,
        ),
        (
            # Terminal chunk does not need to be filled
            [np.arange(5), np.arange(5, 10), np.arange(10, 17)],
            (5,),
            0,
        ),
        (
            [
                np.broadcast_to(np.arange(6), (10, 6)),
                np.broadcast_to(np.arange(7, 10), (10, 3)),
            ],
            (10, 3),
            1,
        ),
        (
            [
                np.broadcast_to(np.arange(6), (10, 6)).T,
                np.broadcast_to(np.arange(7, 10), (10, 3)).T,
            ],
            (3, 10),
            0,
        ),
    ],
)
def test_success(tmpdir, arrays, chunks, axis, m):
    fns = []
    refs = []
    for i, x in enumerate(arrays):
        fn = f"{tmpdir}/out{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array("x", shape=x.shape, dtype=x.dtype, chunks=chunks)
        arr[:] = x
        fns.append(fn)
        ref = kerchunk.zarr.single_zarr(fn, inline=0)
        refs.append(ref)

    out = kerchunk.combine.concatenate_arrays(
        refs, axis=axis, path="x", check_arrays=True
    )

    store = refs_as_store(out)
    g = zarr.open(store, zarr_format=2)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()

    try:
        import fastparquet
    except ImportError:
        return
    kerchunk.df.refs_to_dataframe(out, "memory://out.parq")
    storage_options = dict(
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open("reference://", zarr_format=2, storage_options=storage_options)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()

    kerchunk.df.refs_to_dataframe(out, "memory://out.parq", record_size=1)
    storage_options = dict(
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open("reference://", zarr_format=2, storage_options=storage_options)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()


def test_fail_chunks(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 20)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype, chunks=(2,))
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype, chunks=(3,))
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array chunks at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_shape(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(12).reshape(6, 2)
    x2 = np.arange(12, 24)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype)
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype)
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array shape at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_irregular_chunk_boundaries(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 24)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype, chunks=(4,))
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype, chunks=(4,))
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Array at index 0 has irregular chunking.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)
