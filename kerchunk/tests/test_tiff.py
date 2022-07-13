import glob
import fsspec
import os.path
import zarr
import pytest

pytest.importorskip("tifffile")
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
