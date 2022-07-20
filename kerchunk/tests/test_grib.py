import os.path

import numpy as np
import pytest
import xarray as xr

from kerchunk.grib2 import scan_grib

pytest.importorskip("cfgrib")
here = os.path.dirname(__file__)


@pytest.fixture()
def remove_temp():
    # because cfgrib puts .idx file next to original when we read
    import glob

    yield
    index_files = glob.glob(os.path.join(here, "CMC*.idx"))
    for ind in index_files:
        os.remove(ind)


def test_one(remove_temp):
    # from https://dd.weather.gc.ca/model_gem_regional/10km/grib2/00/000
    fn = os.path.join(here, "CMC_reg_DEPR_ISBL_10_ps10km_2022072000_P000.grib2")
    out = scan_grib(fn)
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={"consolidated": False, "storage_options": {"fo": out[0]}},
    )

    assert ds.attrs["centre"] == "cwao"
    ds2 = xr.open_dataset(fn)
    for var in ["latitude", "longitude", "unknown", "isobaricInhPa", "time"]:
        d1 = ds[var].values
        d2 = ds2[var].values
        assert (np.isnan(d1) == np.isnan(d2)).all()
        assert (d1[~np.isnan(d1)] == d2[~np.isnan(d2)]).all()
