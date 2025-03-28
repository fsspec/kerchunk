import glob
import os.path
import zarr
import pytest
import xarray as xr

from kerchunk.utils import refs_as_store

pytest.importorskip("tifffile")
pytest.importorskip("rioxarray")
import kerchunk.tiff

files = glob.glob(os.path.join(os.path.dirname(__file__), "lc*tif"))


def test_one():
    fn = files[0]
    out = kerchunk.tiff.tiff_to_zarr(fn)
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    assert list(z) == ["0", "1", "2"]
    assert z.attrs["multiscales"] == [
        {
            "datasets": [{"path": "0"}, {"path": "1"}, {"path": "2"}],
            "metadata": {},
            "name": "",
            "version": "0.1",
        }
    ]
    assert z["0"].shape == (2048, 2048)
    assert z["0"][:].max() == 8


def test_one_class():
    from collections import Counter

    fn = files[0]
    out = kerchunk.tiff.TiffToZarr(fn, inline_threshold=1000).translate()

    # test inlineing
    c = Counter(type(_) for k, _ in out.items() if ".z" not in k)
    assert c[list] == 72
    assert c[bytes] == 24

    # test zarr output
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)
    assert list(z) == ["0", "1", "2"]
    assert z.attrs["multiscales"] == [
        {
            "datasets": [{"path": "0"}, {"path": "1"}, {"path": "2"}],
            "metadata": {},
            "name": "",
            "version": "0.1",
        }
    ]
    assert z["0"].shape == (2048, 2048)
    assert z["0"][:].max() == 8


def test_coord():
    fn = files[0]
    out = kerchunk.tiff.tiff_to_zarr(fn)
    store = refs_as_store(out)
    z = zarr.open(store, zarr_format=2)  # highest res is the one xarray picks
    out = kerchunk.tiff.generate_coords(z.attrs, z["0"].shape)

    ds = xr.open_dataset(fn)
    assert (ds.x == out["x"]).all()
    assert (ds.y == out["y"]).all()
