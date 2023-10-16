import pytest

import fsspec
import numpy as np
import zarr

import kerchunk.combine
import kerchunk.zarr
import kerchunk.df


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
        g = zarr.open(fn)
        g.create_dataset("x", data=x, chunks=chunks)
        fns.append(fn)
        ref = kerchunk.zarr.single_zarr(fn, inline=0)
        refs.append(ref)

    out = kerchunk.combine.concatenate_arrays(
        refs, axis=axis, path="x", check_arrays=True
    )

    mapper = fsspec.get_mapper("reference://", fo=out)
    g = zarr.open(mapper)
    assert (g.x[:] == np.concatenate(arrays, axis=axis)).all()

    try:
        import fastparquet
    except ImportError:
        return
    kerchunk.df.refs_to_dataframe(out, "memory://out.parq")
    mapper = fsspec.get_mapper(
        "reference://",
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open(mapper)
    assert (g.x[:] == np.concatenate(arrays, axis=axis)).all()

    kerchunk.df.refs_to_dataframe(out, "memory://out.parq", record_size=1)
    mapper = fsspec.get_mapper(
        "reference://",
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open(mapper)
    assert (g.x[:] == np.concatenate(arrays, axis=axis)).all()


def test_fail_chunks(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 20)
    g = zarr.open(fn1)
    g.create_dataset("x", data=x1, chunks=(2,))
    g = zarr.open(fn2)
    g.create_dataset("x", data=x2, chunks=(3,))

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array chunks at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_shape(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(12).reshape(6, 2)
    x2 = np.arange(12, 24)
    g = zarr.open(fn1)
    g.create_dataset("x", data=x1, chunks=(2,))
    g = zarr.open(fn2)
    g.create_dataset("x", data=x2, chunks=(2,))

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array shape at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_irregular_chunk_boundaries(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 24)
    g = zarr.open(fn1)
    g.create_dataset("x", data=x1, chunks=(4,))
    g = zarr.open(fn2)
    g.create_dataset("x", data=x2, chunks=(4,))

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Array at index 0 has irregular chunking.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


@pytest.mark.parametrize(
    "arrays,axis,expected_chunks",
    [
        (
            [
                zarr.array(np.arange(10), chunks=([2, 2, 2, 2, 2],)),
                zarr.array(np.arange(10, 20), chunks=([3, 3, 3, 1],)),
            ],
            0,
            ([2, 2, 2, 2, 2, 3, 3, 3, 1],),
        ),
        (
            [
                zarr.array(
                    np.broadcast_to(np.arange(6), (10, 6)), chunks=([5, 5], [6])
                ),
                zarr.array(
                    np.broadcast_to(np.arange(7, 10), (10, 3)), chunks=([5, 5], [3])
                ),
            ],
            1,
            ([5, 5], [6, 3]),
        ),
    ],
)
def test_variable_length_chunks_success(tmpdir, arrays, axis, expected_chunks):
    fns = []
    refs = []
    for i, x in enumerate(arrays):
        fn = f"{tmpdir}/out{i}.zarr"
        g = zarr.open(fn)
        g.create_dataset("x", data=x, chunks=x.chunks)
        fns.append(fn)
        ref = kerchunk.zarr.single_zarr(fn, inline=0)
        refs.append(ref)

    out = kerchunk.combine.concatenate_arrays(
        refs, axis=axis, path="x", check_arrays=True
    )

    mapper = fsspec.get_mapper("reference://", fo=out)
    g_result = zarr.open(mapper)

    assert g_result["x"].chunks == expected_chunks
    np.testing.assert_array_equal(
        g_result["x"][...], np.concatenate([a[...] for a in arrays], axis=axis)
    )


def test_variable_length_chunks_mismatch_chunk_failure(tmpdir):
    arrays = [
        zarr.array(np.arange(12).reshape(4, 3), chunks=([2, 2], [1, 2])),
        zarr.array(np.arange(12, 24).reshape(4, 3), chunks=([4], [2, 1])),
    ]
    axis = 0

    fns = []
    refs = []
    for i, x in enumerate(arrays):
        fn = f"{tmpdir}/out{i}.zarr"
        g = zarr.open(fn)
        g.create_dataset("x", data=x, chunks=x.chunks)
        fns.append(fn)
        ref = kerchunk.zarr.single_zarr(fn, inline=0)
        refs.append(ref)

    with pytest.raises(ValueError):
        kerchunk.combine.concatenate_arrays(
            refs, axis=axis, path="x", check_arrays=True
        )
