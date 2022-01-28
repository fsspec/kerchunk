import os.path
import fsspec
import pytest


fits = pytest.importorskip("astropy.io.fits")
import kerchunk.fits
import zarr
testdir = os.path.dirname(fits.util.get_testdata_filepath('btable.fits'))


def test_fits_ascii():
    # this one directly hits a remote server - should cache?
    url = "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"
    out = kerchunk.fits.process_file(url, extension=1)
    m = fsspec.get_mapper("reference://", fo=out, remote_protocol="http")
    g = zarr.open(m)
    arr = g['u5780205r_cvt.c0h.tab']
    with fsspec.open("https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits") as f:
        hdul = fits.open(f)
        hdu = hdul[1]
        assert list(hdu.data.astype(arr.dtype) == arr) == [True, True, True, True]
