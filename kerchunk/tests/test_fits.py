import os.path
import fsspec
import pytest


fits = pytest.importorskip("astropy.io.fits")
import kerchunk.fits
import zarr

testdir = os.path.dirname(fits.util.get_testdata_filepath("btable.fits"))
btable = os.path.join(testdir, "btable.fits")
range_im = os.path.join(testdir, "arange.fits")
var = os.path.join(testdir, "variable_length_table.fits")


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_ascii_table(zarr_version):
    # this one directly hits a remote server - should cache?
    url = "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"
    out = kerchunk.fits.process_file(url, extension=1, zarr_version=zarr_version)
    m = fsspec.get_mapper("reference://", fo=out, remote_protocol="https")
    g = zarr.open(m, zarr_version=zarr_version)
    arr = g["u5780205r_cvt.c0h.tab"][:]
    with fsspec.open(
        "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"
    ) as f:
        hdul = fits.open(f)
        hdu = hdul[1]
        assert list(hdu.data.astype(arr.dtype) == arr) == [True, True, True, True]


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_binary_table(zarr_version):
    out = kerchunk.fits.process_file(btable, extension=1, zarr_version=zarr_version)
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m, zarr_version=zarr_version)
    arr = z["1"]
    with open(btable, "rb") as f:
        hdul = fits.open(f)
        attr2 = dict(arr.attrs)
        assert attr2.pop("_ARRAY_DIMENSIONS") == ["x"]
        assert attr2 == dict(hdul[1].header)
        assert (arr["order"] == hdul[1].data["order"]).all()
        assert (arr["mag"] == hdul[1].data["mag"]).all()
        assert (
            arr["name"].astype("U") == hdul[1].data["name"]
        ).all()  # string come out as bytes


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_cube(zarr_version):
    out = kerchunk.fits.process_file(range_im, zarr_version=zarr_version)
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m, zarr_version=zarr_version)
    arr = z["PRIMARY"]
    with open(range_im, "rb") as f:
        hdul = fits.open(f)
        expected = hdul[0].data
    assert (arr[:] == expected).all()


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_with_class(zarr_version):
    ftz = kerchunk.fits.FitsToZarr(range_im, zarr_version=zarr_version)
    out = ftz.translate()
    assert "fits" in repr(ftz)
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m, zarr_version=zarr_version)
    arr = z["PRIMARY"]
    with open(range_im, "rb") as f:
        hdul = fits.open(f)
        expected = hdul[0].data
    assert (arr[:] == expected).all()


@pytest.mark.parametrize("zarr_version", [2, 3])
def test_var(zarr_version):
    data = fits.open(var)[1].data
    expected = [_.tolist() for _ in data["var"]]

    ftz = kerchunk.fits.FitsToZarr(var, zarr_version=zarr_version)
    out = ftz.translate()
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m, zarr_version=zarr_version)
    arr = z["1"]
    vars = [_.tolist() for _ in arr["var"]]

    assert vars == expected
    assert (z["1"]["xyz"] == data["xyz"]).all()
