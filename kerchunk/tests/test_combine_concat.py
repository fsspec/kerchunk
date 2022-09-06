import pytest

import fsspec
import numpy as np
import zarr

import kerchunk.combine
import kerchunk.zarr


def test_success(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 20)
    g = zarr.open(fn1)
    g.create_dataset("x", data=x1, chunks=(2,))
    g = zarr.open(fn2)
    g.create_dataset("x", data=x2, chunks=(2,))

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    out = kerchunk.combine.concatenate_arrays([ref1, ref2], path="x")

    mapper = fsspec.get_mapper("reference://", fo=out)
    g = zarr.open(mapper)
    assert (g.x[:] == np.concatenate([x1, x2])).all()


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

    with pytest.raises(ValueError):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_shape(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10).reshape(5, 2)
    x2 = np.arange(10, 20)
    g = zarr.open(fn1)
    g.create_dataset("x", data=x1, chunks=(2,))
    g = zarr.open(fn2)
    g.create_dataset("x", data=x2, chunks=(2,))

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)
