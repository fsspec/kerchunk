import os.path

import fsspec
import numpy as np
import pytest
import xarray as xr

from kerchunk.grib2 import scan_grib, _split_file, GribToZarr

cfgrib = pytest.importorskip("cfgrib")
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


def _fetch_first(url):
    fs = fsspec.filesystem("s3", anon=True)
    with fs.open(url, "rb") as f:
        for _, _, data in _split_file(f, skip=1):
            return data


@pytest.mark.parametrize(
    "url",
    [
        "s3://noaa-hrrr-bdp-pds/hrrr.20140730/conus/hrrr.t23z.wrfsubhf1430.grib2",
        "s3://noaa-gefs-pds/gefs.20221011/00/atmos/pgrb2ap5/gep01.t00z.pgrb2a.0p50.f570",
    ],
)
def test_archives(tmpdir, url):
    grib = GribToZarr(url, storage_options={"anon": True}, skip=1)
    out = grib.translate()[0]
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": out,
                "remote_protocol": "s3",
                "remote_options": {"anon": True},
            },
        },
    )

    data = _fetch_first(url)
    fn = os.path.join(tmpdir, "grib.grib2")
    with open(fn, "wb") as f:
        f.write(data)

    ds2 = cfgrib.open_dataset(fn)
    dims = list(ds.dims)
    if "hrrr" in url:
        assert (ds.refc == ds2.refc).all()
        assert dims.index("y") < dims.index("x")
    else:
        assert np.allclose(ds.gh, ds2.gh)
        assert dims[0] == "latitude"
