import os.path

import fsspec
import numpy as np
import pytest
import xarray as xr

from kerchunk.grib2 import scan_grib, _split_file, GribToZarr

cfgrib = pytest.importorskip("cfgrib")
here = os.path.dirname(__file__)


def test_one():
    # from https://dd.weather.gc.ca/model_gem_regional/10km/grib2/00/000
    fn = os.path.join(here, "CMC_reg_DEPR_ISBL_10_ps10km_2022072000_P000.grib2")
    out = scan_grib(fn)
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={"consolidated": False, "storage_options": {"fo": out[0]}},
    )

    assert ds.attrs["GRIB_centre"] == "cwao"
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
        "s3://noaa-hrrr-bdp-pds/hrrr.20140730/conus/hrrr.t23z.wrfsubhf08.grib2",
        "s3://noaa-gefs-pds/gefs.20221011/00/atmos/pgrb2ap5/gep01.t00z.pgrb2a.0p50.f570",
        "s3://noaa-gefs-retrospective/GEFSv12/reforecast/2000/2000010100/c00/Days:10-16/acpcp_sfc_2000010100_c00.grib2",
    ],
)
def test_archives(tmpdir, url):
    grib = GribToZarr(url, storage_options={"anon": True}, skip=1)
    out = grib.translate()[0]
    ours = xr.open_dataset(
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

    theirs = cfgrib.open_dataset(fn)
    if "hrrr" in url:
        # for some reason, cfgrib reads `step` as 7.25 hours
        # while grib_ls and kerchunk reads `step` as 425 hours.
        ours = ours.drop_vars("step")
        theirs = theirs.drop_vars("step")

    xr.testing.assert_allclose(ours, theirs)


def test_subhourly():
    # two messages extracted from a hrrr output including one with an eccodes
    # non-compliant endstep type which raises WrongStepUnitError
    fpath = os.path.join(here, "hrrr.wrfsubhf.sample.grib2")
    result = scan_grib(fpath)
    assert len(result) == 2, "Expected two grib messages"
