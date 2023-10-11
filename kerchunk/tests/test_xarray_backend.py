import pytest
import fsspec
import xarray as xr

# Add parameterize test for storage formats (.json, .parquet)
def test_open_reference_xarray():

    ds = xr.open_dataset("tests/single_file_kerchunk.json", engine="kerchunk")
