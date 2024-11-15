import xarray as xr
import numpy as np
from kerchunk import netCDF3

arr = np.random.rand(1, 10, 10)
data = xr.DataArray(
    data=arr.squeeze(),
    dims=["x", "y"],
    name="data",
)
bdata = xr.Dataset({"data": data}, attrs={"attr0": 3}).to_netcdf(
    format="NETCDF3_CLASSIC"
)

# Add parameterize test for storage formats (.json, .parquet)
def test_reference_netcdf(m):
    m.pipe("data.nc3", bdata)
    h = netCDF3.netcdf_recording_file("memory://data.nc3")
    out = h.translate()
    ds = xr.open_dataset(
        out, engine="kerchunk", storage_options={"remote_protocol": "memory"}
    )
    assert (ds.data == data).all()
