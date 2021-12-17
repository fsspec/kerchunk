import numpy as np
import pytest
import xarray as xr

from kerchunk.zarr import single_zarr
from kerchunk.combine import MultiZarrToZarr

arr = np.random.rand(1, 10, 10)
t0 = np.array([1])
t1 = np.array([2])
data1 = xr.DataArray(
    data=arr,
    coords={"time": t0},
    dims=["time", "x", "y"],
    name="data",
    attrs={"attr0": 0}
)
data1_1 = xr.Dataset({"data": data1})


@pytest.fixture()
def onezarr(m):
    data1_1.to_zarr("memory://zarr.zarr")
    refs = single_zarr("memory://zarr.zarr")
    return refs


@pytest.fixture()
def twozarrs(m):
    data1_1.to_zarr("memory://zarr1.zarr")
    refs1 = single_zarr("memory://zarr1.zarr")
    data2_1 = data1_1.assign_coords(time=t1)
    data2_1.to_zarr("memory://zarr2.zarr")
    refs2 = single_zarr("memory://zarr2.zarr")
    return refs1, refs2


def test_fixture(onezarr):
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={"storage_options": {"fo": onezarr}, "consolidated": False},
        engine="zarr"
    )
    assert (z.data.values == arr).all()


def test_write_remote(twozarrs, m):
    mzz = MultiZarrToZarr(twozarrs, remote_protocol="memory",
                          xarray_concat_args={"dim": "time"})
    mzz.translate("memory://combined.json")
    z = xr.open_dataset(
        "reference://",
        backend_kwargs={"storage_options": {"fo": "memory://combined.json"},
                        "consolidated": False},
        engine="zarr"
    )
    # TODO: make some assert_eq style function
    assert z.time.values.tolist() == [t0, t1]
    assert z.data.shape == (2, 10, 10)
    assert (z.data[0].values == arr).all()
    assert (z.data[1].values == arr).all()
