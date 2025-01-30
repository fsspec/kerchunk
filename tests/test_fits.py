import os.path
import fsspec
import pytest

from kerchunk.utils import refs_as_store


fits = pytest.importorskip("astropy.io.fits")
import kerchunk.fits
import zarr

testdir = os.path.dirname(fits.util.get_testdata_filepath("btable.fits"))
btable = os.path.join(testdir, "btable.fits")
range_im = os.path.join(testdir, "arange.fits")
var = os.path.join(testdir, "variable_length_table.fits")


def test_image():
    # this one directly hits a remote server - should cache?
    url = "https://fits.gsfc.nasa.gov/samples/WFPC2ASSNu5780205bx.fits"
    out = kerchunk.fits.process_file(url)
    g = zarr.open(
        "reference://",
        storage_options=dict(
            fo=out,
            remote_protocol="https",
            asynchronous=True,
            remote_options={"asynchronous": True},
        ),
    )

    # store = refs_as_store(out, remote_protocol="https")
    # g = zarr.open(store)
    arr = g["PRIMARY"][:]
    with fsspec.open(
        "https://fits.gsfc.nasa.gov/samples/WFPC2ASSNu5780205bx.fits"
    ) as f:
        hdu = fits.getdata(f)
        assert (hdu == arr).all()


def test_ascii_table():
    # this one directly hits a remote server - should cache?
    url = "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"
    out = kerchunk.fits.process_file(url, extension=1)
    store = refs_as_store(out, remote_protocol="https")
    g = zarr.open(store, zarr_format=2)
    arr = g["u5780205r_cvt.c0h.tab"][:]
    with fsspec.open(
        "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"
    ) as f:
        hdul = fits.open(f)
        hdu = hdul[1]
        assert list(hdu.data.astype(arr.dtype) == arr) == [True, True, True, True]


def test_binary_table():
    out = kerchunk.fits.process_file(btable, extension=1)
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    arr = z["1"]
    with open(btable, "rb") as f:
        hdul = fits.open(f)
        attr2 = dict(arr.attrs)
        assert attr2.pop("_ARRAY_DIMENSIONS") == ["x"]
        assert attr2 == dict(hdul[1].header)
        assert (arr[:]["order"] == hdul[1].data["order"]).all()
        assert (arr[:]["mag"] == hdul[1].data["mag"]).all()
        assert (
            arr[:]["name"].astype("U") == hdul[1].data["name"]
        ).all()  # string come out as bytes


def test_cube():
    out = kerchunk.fits.process_file(range_im)
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    arr = z["PRIMARY"]
    with open(range_im, "rb") as f:
        hdul = fits.open(f)
        expected = hdul[0].data
    assert (arr[:] == expected).all()


def test_with_class():
    ftz = kerchunk.fits.FitsToZarr(range_im)
    out = ftz.translate()
    assert "fits" in repr(ftz)
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    arr = z["PRIMARY"]
    with open(range_im, "rb") as f:
        hdul = fits.open(f)
        expected = hdul[0].data
    assert (arr[:] == expected).all()


def test_var():
    data = fits.open(var)[1].data
    expected = [_.tolist() for _ in data["var"]]

    ftz = kerchunk.fits.FitsToZarr(var)
    out = ftz.translate()
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    arr = z["1"]
    vars = [_.tolist() for _ in arr[:]["var"]]

    assert vars == expected
    assert (z["1"][:]["xyz"] == data["xyz"]).all()
