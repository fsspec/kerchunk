import os

import fsspec
import numpy as np
import pytest
from kerchunk import netCDF3

xr = pytest.importorskip("xarray")

arr = np.random.rand(1, 10, 10)
static = xr.DataArray(data=np.random.rand(10, 10), dims=["x", "y"], name="static")
data = xr.DataArray(
    data=arr.squeeze(),
    dims=["x", "y"],
    name="data",
)
bdata = xr.Dataset({"data": data}, attrs={"attr0": 3}).to_netcdf(
    format="NETCDF3_CLASSIC"
)

m = fsspec.filesystem("memory")
m.pipe("data.nc3", bdata)


def test_one():
    h = netCDF3.netcdf_recording_file("memory://data.nc3")
    out = h.translate()
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {"fo": out, "remote_protocol": "memory"},
        },
    )
    assert (ds.data == data).all()


@pytest.fixture()
def unlimited_dataset(tmpdir):
    # https://unidata.github.io/netcdf4-python/#creatingopeningclosing-a-netcdf-file
    from netCDF4 import Dataset

    fn = os.path.join(tmpdir, "test.nc")
    rootgrp = Dataset(fn, "w", format="NETCDF3_CLASSIC")
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("lat", 10)
    rootgrp.createDimension("lon", 5)
    rootgrp.createVariable("time", "f8", ("time",))
    rootgrp.title = "testing"
    latitudes = rootgrp.createVariable("lat", "f4", ("lat",))
    longitudes = rootgrp.createVariable("lon", "f4", ("lon",))
    temp = rootgrp.createVariable(
        "temp",
        "f4",
        (
            "time",
            "lat",
            "lon",
        ),
    )
    temp.units = "K"
    latitudes[:] = np.arange(-0.5, 0.5, 0.1)
    longitudes[:] = np.arange(0, 0.5, 0.1)
    for i in range(8):
        temp[i] = np.random.uniform(size=(1, 10, 5))
    rootgrp.close()
    return fn


def test_unlimited(unlimited_dataset):
    fn = unlimited_dataset
    expected = xr.open_dataset(fn, engine="scipy")
    h = netCDF3.NetCDF3ToZarr(fn)
    out = h.translate()
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {"fo": out},
        },
    )
    assert ds.attrs["title"] == "testing"
    assert ds.temp.attrs["units"] == "K"
    assert (ds.lat.values == expected.lat.values).all()
    assert (ds.lon.values == expected.lon.values).all()
    assert (ds.temp.values == expected.temp.values).all()
