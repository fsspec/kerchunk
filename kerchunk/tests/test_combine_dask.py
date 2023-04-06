import pytest
import re

import fsspec
import xarray as xr

from kerchunk.combine import auto_dask
from kerchunk.zarr import ZarrToZarr

dask = pytest.importorskip("dask")


@pytest.mark.parametrize("n_batches", [1, 2, 3])
def test_simplest(m, n_batches):
    for i in range(4):
        m.pipe(
            {
                f"data{i}/.zgroup": b'{"zarr_format":2}',
                f"data{i}/data/.zarray": b'{"chunks":[3],"compressor": null,"dtype": "<i1",'
                b'"fill_value": 0,"filters": null,"order": "C",'
                b'"shape": [3],"zarr_format": 2}',
                f"data{i}/data/0": f"{i}{i}{i}".encode(),
            }
        )
    out = auto_dask(
        [f"memory:///data{i}" for i in range(4)],
        single_driver=ZarrToZarr,
        single_kwargs={"inline": 0},
        n_batches=n_batches,
        mzz_kwargs={
            "coo_map": {"count": re.compile(r".*(\d)")},
            "inline_threshold": 0,
            "coo_dtypes": {"count": "i4"},
        },
    )
    fs = fsspec.filesystem("reference", fo=out)
    ds = xr.open_dataset(
        fs.get_mapper(), engine="zarr", backend_kwargs={"consolidated": False}
    )
    assert ds["count"].values.tolist() == [0, 1, 2, 3]
    assert ds.data.shape == (4, 3)
    assert (ds.data.values.T == [48, 49, 50, 51]).all()
