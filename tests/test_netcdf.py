import os


import fsspec
import numpy as np
from packaging.version import Version
import pytest
from kerchunk import netCDF3

xr = pytest.importorskip("xarray")


has_xarray_2023_8_0 = Version(xr.__version__) >= Version("2023.8.0")


arr = np.random.rand(1, 10, 10)
data = xr.DataArray(
    data=arr.squeeze(),
    dims=["x", "y"],
    name="data",
)
bdata = xr.Dataset({"data": data}, attrs={"attr0": 3}).to_netcdf(
    format="NETCDF3_CLASSIC"
)


def test_one(m):
    m.pipe("data.nc3", bdata)
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
    # reference time is an unbounded dimension that is a half byte long, so it
    # has padding to line up to take up exactly one byte. It is here to test that
    # kerchunk can handle the padding correctly and read following variables
    # correctly.
    rootgrp.createVariable("reference_time", "h", ("time",))
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


@pytest.fixture()
def matching_coordinate_dimension_dataset(tmpdir):
    """Create a dataset with a coordinate dimension that matches the name of a
    variable dimension."""
    # https://unidata.github.io/netcdf4-python/#creatingopeningclosing-a-netcdf-file
    from netCDF4 import Dataset

    fn = os.path.join(tmpdir, "test.nc")
    rootgrp = Dataset(fn, "w", format="NETCDF3_64BIT")
    rootgrp.createDimension("node", 2)
    rootgrp.createDimension("sigma", 2)

    node = rootgrp.createVariable("node", "i4", ("node",))
    sigma = rootgrp.createVariable("sigma", "f8", ("sigma", "node"))

    node[:] = [0, 1]
    for i in range(2):
        sigma[i] = np.random.uniform(size=(2,))

    rootgrp.close()
    return fn


@pytest.mark.skipif(
    not has_xarray_2023_8_0, reason="XArray 2023.08.0 is required for this behavior."
)
def test_matching_coordinate_dimension(matching_coordinate_dimension_dataset):
    fn = matching_coordinate_dimension_dataset
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
    assert (ds.node.values == expected.node.values).all()
    assert (ds.sigma.values == expected.sigma.values).all()
