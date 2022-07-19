import glob
import fsspec
import os.path
import zarr
import pytest
import xarray as xr

pytest.importorskip("tifffile")
pytest.importorskip("rioxarray")
import kerchunk.tiff

files = glob.glob(os.path.join(os.path.dirname(__file__), "lc*tif"))


def test_one():
    fn = files[0]
    out = kerchunk.tiff.tiff_to_zarr(fn)
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m)
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
    m = fsspec.get_mapper("reference://", fo=out)
    z = zarr.open(m)  # highest res is the one xarray picks
    out = kerchunk.tiff.generate_coords(z.attrs, z[0].shape)

    ds = xr.open_dataset(fn)
    assert (ds.x == out["x"]).all()
    assert (ds.y == out["y"]).all()
