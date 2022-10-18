import os.path

import numpy as np
import pytest
import xarray as xr

from kerchunk.grib2 import scan_grib

pytest.importorskip("cfgrib")
here = os.path.dirname(__file__)


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_one(zarr_version):
    # from https://dd.weather.gc.ca/model_gem_regional/10km/grib2/00/000
    fn = os.path.join(here, "CMC_reg_DEPR_ISBL_10_ps10km_2022072000_P000.grib2")
    out = scan_grib(fn, zarr_version=zarr_version)
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "zarr_version": zarr_version,
            "storage_options": {"fo": out[0]},
        },
    )

    assert ds.attrs["centre"] == "cwao"
    ds2 = xr.open_dataset(fn, engine="cfgrib", backend_kwargs={"indexpath": ""})

    for var in ["latitude", "longitude", "unknown", "isobaricInhPa", "time"]:
        d1 = ds[var].values
        d2 = ds2[var].values
        assert (np.isnan(d1) == np.isnan(d2)).all()
        assert (d1[~np.isnan(d1)] == d2[~np.isnan(d2)]).all()
