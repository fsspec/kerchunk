import os.path

import fsspec
import numpy as np
import pytest
import xarray as xr
import datatree
import zarr
import ujson
from kerchunk.grib2 import (
    scan_grib,
    _split_file,
    GribToZarr,
    grib_tree,
    correct_hrrr_subhf_step,
)

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


def test_grib_tree():
    """
    End-to-end test from grib file to zarr hierarchy
    """
    fpath = os.path.join(here, "hrrr.wrfsubhf.sample.grib2")
    scanned_msg_groups = scan_grib(fpath)
    corrected_msg_groups = [correct_hrrr_subhf_step(msg) for msg in scanned_msg_groups]
    result = grib_tree(corrected_msg_groups)
    fs = fsspec.filesystem("reference", fo=result)
    zg = zarr.open_group(fs.get_mapper(""))
    assert isinstance(zg["refc/instant/atmosphere/refc"], zarr.Array)
    assert isinstance(zg["vbdsf/avg/surface/vbdsf"], zarr.Array)
    assert (
        zg["vbdsf/avg/surface"].attrs["coordinates"]
        == "surface latitude longitude time valid_time step"
    )
    assert (
        zg["refc/instant/atmosphere"].attrs["coordinates"]
        == "atmosphere latitude longitude step time valid_time"
    )
    # Assert that the fill value is set correctly
    assert zg.refc.instant.atmosphere.step.fill_value is np.NaN


# The following two tests use json fixture data generated from calling scan grib
#   scan_grib("testdata/hrrr.t01z.wrfsubhf00.grib2")
#   scan_grib("testdata/hrrr.t01z.wrfsubhf01.grib2")
# and filtering the results message groups for keys starting with "dswrf" or "u"
# The original files are:
# gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsubhf00.grib2"
# gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsubhf01.grib2"


def test_correct_hrrr_subhf_group_step():
    fpath = os.path.join(here, "hrrr.wrfsubhf.subset.json")
    with open(fpath, "rb") as fobj:
        scanned_msgs = ujson.load(fobj)

    original_zg = [
        zarr.open_group(fsspec.filesystem("reference", fo=val).get_mapper(""))
        for val in scanned_msgs
    ]

    corrected_msgs = [correct_hrrr_subhf_step(msg) for msg in scanned_msgs]

    corrected_zg = [
        zarr.open_group(fsspec.filesystem("reference", fo=val).get_mapper(""))
        for val in corrected_msgs
    ]

    # The groups that were missing a step variable got fixed
    assert all(["step" in zg.array_keys() for zg in corrected_zg])
    assert not all(["step" in zg.array_keys() for zg in original_zg])

    # The step values are corrected to floating point hour
    assert all([zg.step[()] <= 1.0 for zg in corrected_zg])
    # The original seems to have values in minutes for some step variables!
    assert not all(
        [zg.step[()] <= 1.0 for zg in original_zg if "step" in zg.array_keys()]
    )


def test_hrrr_subhf_corrected_grib_tree():
    fpath = os.path.join(here, "hrrr.wrfsubhf.subset.json")
    with open(fpath, "rb") as fobj:
        scanned_msgs = ujson.load(fobj)

    corrected_msgs = [correct_hrrr_subhf_step(msg) for msg in scanned_msgs]
    merged = grib_tree(corrected_msgs)
    zg = zarr.open_group(fsspec.filesystem("reference", fo=merged).get_mapper(""))
    # Check the values and shape of the time coordinates
    assert zg.u.instant.heightAboveGround.step[:].tolist() == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert zg.u.instant.heightAboveGround.step.shape == (5,)

    assert zg.u.instant.heightAboveGround.valid_time[:].tolist() == [
        [1695862800, 1695863700, 1695864600, 1695865500, 1695866400]
    ]
    assert zg.u.instant.heightAboveGround.valid_time.shape == (1, 5)

    assert zg.u.instant.heightAboveGround.time[:].tolist() == [1695862800]
    assert zg.u.instant.heightAboveGround.time.shape == (1,)

    assert zg.dswrf.avg.surface.step[:].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert zg.dswrf.avg.surface.step.shape == (5,)

    assert zg.dswrf.avg.surface.valid_time[:].tolist() == [
        [1695862800, 1695863700, 1695864600, 1695865500, 1695866400]
    ]
    assert zg.dswrf.avg.surface.valid_time.shape == (1, 5)

    assert zg.dswrf.avg.surface.time[:].tolist() == [1695862800]
    assert zg.dswrf.avg.surface.time.shape == (1,)


# The following two test use json fixture data generated from calling scan grib
#   scan_grib("testdata/hrrr.t01z.wrfsfcf00.grib2")
#   scan_grib("testdata/hrrr.t01z.wrfsfcf01.grib2")
# and filtering the results for keys starting with "dswrf" or "u"
# The original files are:
# gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsfcf00.grib2"
# gs://high-resolution-rapid-refresh/hrrr.20210928/conus/hrrr.t01z.wrfsfcf01.grib2"
def test_hrrr_sfcf_grib_tree():
    fpath = os.path.join(here, "hrrr.wrfsfcf.subset.json")
    with open(fpath, "rb") as fobj:
        scanned_msgs = ujson.load(fobj)
    merged = grib_tree(scanned_msgs)
    zg = zarr.open_group(fsspec.filesystem("reference", fo=merged).get_mapper(""))
    # Check the heightAboveGround level shape of the time coordinates
    assert zg.u.instant.heightAboveGround.heightAboveGround[()] == 80.0
    assert zg.u.instant.heightAboveGround.heightAboveGround.shape == ()

    assert zg.u.instant.heightAboveGround.step[:].tolist() == [0.0, 1.0]
    assert zg.u.instant.heightAboveGround.step.shape == (2,)

    assert zg.u.instant.heightAboveGround.valid_time[:].tolist() == [
        [1695862800, 1695866400]
    ]
    assert zg.u.instant.heightAboveGround.valid_time.shape == (1, 2)

    assert zg.u.instant.heightAboveGround.time[:].tolist() == [1695862800]
    assert zg.u.instant.heightAboveGround.time.shape == (1,)

    # Check the isobaricInhPa level shape and time coordinates
    assert zg.u.instant.isobaricInhPa.isobaricInhPa[:].tolist() == [
        250.0,
        300.0,
        500.0,
        700.0,
        850.0,
        925.0,
        1000.0,
    ]
    assert zg.u.instant.isobaricInhPa.isobaricInhPa.shape == (7,)

    assert zg.u.instant.isobaricInhPa.step[:].tolist() == [0.0, 1.0]
    assert zg.u.instant.isobaricInhPa.step.shape == (2,)

    # Valid time values get exploded by isobaricInhPa aggregation
    # Is this a feature or a bug?
    expected_valid_times = [
        [
            [1695862800 for _ in range(7)],
            [1695866400 for _ in range(7)],
        ]
    ]
    assert zg.u.instant.isobaricInhPa.valid_time[:].tolist() == expected_valid_times
    assert zg.u.instant.isobaricInhPa.valid_time.shape == (1, 2, 7)

    assert zg.u.instant.isobaricInhPa.time[:].tolist() == [1695862800]
    assert zg.u.instant.isobaricInhPa.time.shape == (1,)


def test_hrrr_sfcf_grib_datatree():
    fpath = os.path.join(here, "hrrr.wrfsfcf.subset.json")
    with open(fpath, "rb") as fobj:
        scanned_msgs = ujson.load(fobj)
    merged = grib_tree(scanned_msgs)
    dt = datatree.open_datatree(
        fsspec.filesystem("reference", fo=merged).get_mapper(""),
        engine="zarr",
        consolidated=False,
    )
    # Assert a few things... but if it loads we are mostly done.
    np.testing.assert_array_equal(
        dt.u.instant.heightAboveGround.step.values[:],
        np.array([0, 3600 * 10**9], dtype="timedelta64[ns]"),
    )
    assert dt.u.attrs == dict(name="U component of wind")
