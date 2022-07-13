import fsspec
import io
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
