import os.path

import zarr

import kerchunk.hdf4
from kerchunk.utils import refs_as_store


def test1():
    here = os.path.dirname(__file__)
    fn = os.path.join(here, "MOD14.hdf4")

    out = kerchunk.hdf4.HDF4ToZarr(fn).translate()
    store = refs_as_store(out)
    g = zarr.open(store, zarr_format=2)
    assert g["fire mask"][:].max() == 5
