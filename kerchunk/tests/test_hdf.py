import fsspec
import os.path as osp

import kerchunk.hdf
import numpy as np
import pytest
import xarray as xr
import zarr

from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr, drop

here = osp.dirname(__file__)


def test_single():
    """Test creating references for a single HDF file"""
    url = "s3://noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp"
    so = dict(anon=True, default_fill_cache=False, default_cache_type="none")
    with fsspec.open(url, **so) as f:
        h5chunks = SingleHdf5ToZarr(f, url)
        test_dict = h5chunks.translate()

    m = fsspec.get_mapper(
        "reference://", fo=test_dict, remote_protocol="s3", remote_options=so
    )
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))

    with fsspec.open(url, **so) as f:
        expected = xr.open_dataset(f, engine="h5netcdf")

        xr.testing.assert_equal(ds.drop_vars("crs"), expected.drop_vars("crs"))


urls = [
    "s3://" + p
    for p in [
        "noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp",
        "noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010100.CHRTOUT_DOMAIN1.comp",
        "noaa-nwm-retro-v2.0-pds/full_physics/2017/201704010200.CHRTOUT_DOMAIN1.comp",
    ]
]
so = dict(anon=True, default_fill_cache=False, default_cache_type="first")


def test_multizarr(generate_mzz):
    """Test creating a combined reference file with MultiZarrToZarr"""
    mzz = generate_mzz
    test_dict = mzz.translate()

    m = fsspec.get_mapper(
        "reference://", fo=test_dict, remote_protocol="s3", remote_options=so
    )
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))

    with fsspec.open_files(urls, **so) as fs:
        expts = [xr.open_dataset(f, engine="h5netcdf") for f in fs]
        expected = xr.concat(expts, dim="time")

        assert set(ds) == set(expected)
        for name in ds:
            exp = {
                k: (v.tolist() if v.size > 1 else v[0])
                if isinstance(v, np.ndarray)
                else v
                for k, v in expected[name].attrs.items()
            }
            assert dict(ds[name].attrs) == dict(exp)
        for coo in ds.coords:
            assert (ds[coo].values == expected[coo].values).all()


@pytest.fixture(scope="module")
def generate_mzz():
    """This function generates a MultiZarrToZarr class for tests"""

    dict_list = []

    for u in urls:
        with fsspec.open(u, **so) as inf:
            h5chunks = SingleHdf5ToZarr(inf, u, inline_threshold=100)
            dict_list.append(h5chunks.translate())

    mzz = MultiZarrToZarr(
        dict_list,
        remote_protocol="s3",
        remote_options={"anon": True},
        concat_dims=["time"],
        preprocess=drop("reference_time"),
    )
    return mzz


@pytest.fixture()
def times_data(tmpdir):
    lat = xr.DataArray(np.linspace(-90, 90, 10), dims=["lat"], name="lat")
    lon = xr.DataArray(np.linspace(-90, 90, 10), dims=["lon"], name="lon")
    time_attrs = {"axis": "T", "long_name": "time", "standard_name": "time"}
    time1 = xr.DataArray(
        np.arange(-631108800000000000, -630158390000000000, 86400000000000).view(
            "datetime64[ns]"
        ),
        dims=["time"],
        name="time",
        attrs=time_attrs,
    )

    x1 = xr.DataArray(
        np.zeros((12, 10, 10)),
        dims=["time", "lat", "lon"],
        coords={"time": time1, "lat": lat, "lon": lon},
        name="prcp",
    )
    url = str(tmpdir.join("x1.nc"))
    x1.to_netcdf(url, engine="h5netcdf")
    return x1, url


def test_times(times_data):
    x1, url = times_data
    # Test taken from https://github.com/fsspec/kerchunk/issues/115#issue-1091163872
    with fsspec.open(url) as f:
        h5chunks = SingleHdf5ToZarr(f, url)
        test_dict = h5chunks.translate()

    m = fsspec.get_mapper(
        "reference://",
        fo=test_dict,
    )
    result = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))
    expected = x1.to_dataset()
    xr.testing.assert_equal(result, expected)


def test_times_str(times_data):
    # as test above, but using str input for SingleHdf5ToZarr file
    x1, url = times_data
    # Test taken from https://github.com/fsspec/kerchunk/issues/115#issue-1091163872
    h5chunks = SingleHdf5ToZarr(url)
    test_dict = h5chunks.translate()

    m = fsspec.get_mapper(
        "reference://",
        fo=test_dict,
    )
    result = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))
    expected = x1.to_dataset()
    xr.testing.assert_equal(result, expected)


# https://stackoverflow.com/a/43935389/3821154
txt = "the change of water into water vapour"


def test_string_embed():
    fn = osp.join(here, "vlen.h5")
    h = kerchunk.hdf.SingleHdf5ToZarr(fn, fn, vlen_encode="embed")
    out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    assert txt in fs.references["vlen_str/0"]
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str.dtype == "O"
    assert z.vlen_str[0] == txt
    assert (z.vlen_str[1:] == "").all()


def test_string_null():
    fn = osp.join(here, "vlen.h5")
    h = kerchunk.hdf.SingleHdf5ToZarr(fn, fn, vlen_encode="null", inline_threshold=0)
    out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str.dtype == "O"
    assert (z.vlen_str[:] == None).all()


def test_string_leave():
    fn = osp.join(here, "vlen.h5")
    with open(fn, "rb") as f:
        h = kerchunk.hdf.SingleHdf5ToZarr(
            f, fn, vlen_encode="leave", inline_threshold=0
        )
        out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str.dtype == "S16"
    assert z.vlen_str[0]  # some obscured ID
    assert (z.vlen_str[1:] == b"").all()


def test_string_decode():
    fn = osp.join(here, "vlen.h5")
    with open(fn, "rb") as f:
        h = kerchunk.hdf.SingleHdf5ToZarr(
            f, fn, vlen_encode="encode", inline_threshold=0
        )
        out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    assert txt in fs.cat("vlen_str/.zarray").decode()  # stored in filter def
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str[0] == txt
    assert (z.vlen_str[1:] == "").all()


def test_compound_string_null():
    fn = osp.join(here, "vlen2.h5")
    with open(fn, "rb") as f:
        h = kerchunk.hdf.SingleHdf5ToZarr(f, fn, vlen_encode="null", inline_threshold=0)
        out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str[0].tolist() == (10, None)
    assert (z.vlen_str["ints"][1:] == 0).all()
    assert (z.vlen_str["strs"][1:] == None).all()


def test_compound_string_leave():
    fn = osp.join(here, "vlen2.h5")
    with open(fn, "rb") as f:
        h = kerchunk.hdf.SingleHdf5ToZarr(
            f, fn, vlen_encode="leave", inline_threshold=0
        )
        out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str["ints"][0] == 10
    assert z.vlen_str["strs"][0]  # random ID
    assert (z.vlen_str["ints"][1:] == 0).all()
    assert (z.vlen_str["strs"][1:] == b"").all()


def test_compound_string_encode():
    fn = osp.join(here, "vlen2.h5")
    with open(fn, "rb") as f:
        h = kerchunk.hdf.SingleHdf5ToZarr(
            f, fn, vlen_encode="encode", inline_threshold=0
        )
        out = h.translate()
    fs = fsspec.filesystem("reference", fo=out)
    z = zarr.open(fs.get_mapper())
    assert z.vlen_str["ints"][0] == 10
    assert z.vlen_str["strs"][0] == "water"
    assert (z.vlen_str["ints"][1:] == 0).all()
    assert (z.vlen_str["strs"][1:] == "").all()
