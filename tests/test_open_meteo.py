import os
import json
import pytest
import fsspec
import numpy as np
import zarr
import numcodecs

import omfiles
from omfiles.omfiles_numcodecs import PyPforDelta2dSerializer, PyPforDelta2d

from kerchunk.utils import fs_as_store, refs_as_store
from kerchunk.df import refs_to_dataframe
from kerchunk.open_meteo import SingleOmToZarr, Delta2D

# Register codecs needed by our pipeline
numcodecs.register_codec(PyPforDelta2d, "pfor")
numcodecs.register_codec(PyPforDelta2dSerializer, "pfor_serializer")
numcodecs.register_codec(Delta2D)

# Path to test file - adjust as needed
test_file = "tests/rh_icon_chunk1914.om"
# Skip test if file doesn't exist
pytestmark = pytest.mark.skipif(
    not os.path.exists(test_file), reason=f"Test file {test_file} not found"
)

def test_single_om_to_zarr():
    """Test creating references for a single OM file and reading via Zarr"""
    # Create references using SingleOmToZarr
    om_to_zarr = SingleOmToZarr(test_file, inline_threshold=0)  # No inlining for testing
    references = om_to_zarr.translate()
    om_to_zarr.close()

    print("type(references)", type(references))
    print("references.keys", references.keys())

    # Optionally save references to a file for inspection
    with open("om_refs.json", "w") as f:
        json.dump(references, f)

    output_path = "om_refs_parquet/"  # This will be a directory
    refs_to_dataframe(
        fo="om_refs.json",              # References dict
        url=output_path,            # Output URL
        record_size=100000,         # Records per file, adjust as needed
        categorical_threshold=10    # URL encoding efficiency
    )

    # Create a filesystem from the references
    fs = fsspec.filesystem("reference", fo=references)
    store = fs_as_store(fs)

    # Open with zarr
    group = zarr.open(store, zarr_format=2)
    z = group["data"] # Here we just use a dummy data name we have hardcoded in SingleOmToZarr

    # Verify basic metadata matches original file
    reader = omfiles.OmFilePyReader(test_file)
    assert list(z.shape) == reader.shape, f"Shape mismatch: {z.shape} vs {reader.shape}"
    assert str(z.dtype) == str(reader.dtype), f"Dtype mismatch: {z.dtype} vs {reader.dtype}"
    assert list(z.chunks) == reader.chunk_dimensions, f"Chunks mismatch: {z.chunks} vs {reader.chunk_dimensions}"

    # Test retrieving a specific chunk (same chunk as in your example)
    chunk_index = (5, 5, ...)

    # Get direct chunk data
    direct_data = reader[chunk_index]

    # Now get the same chunk via kerchunk/zarr
    # Get the data via zarr
    try:
        print("z", z)
        zarr_data = z[chunk_index]
        print("zarr_data", zarr_data)

        # Compare the data
        print(f"Direct data shape: {direct_data.shape}, Zarr data shape: {zarr_data.shape}")
        print(f"Direct data min/max: {np.min(direct_data)}/{np.max(direct_data)}")
        print(f"Zarr data min/max: {np.min(zarr_data)}/{np.max(zarr_data)}")

        # Assert data is equivalent
        np.testing.assert_allclose(zarr_data, direct_data, rtol=1e-5)

        print("✓ Data from kerchunk matches direct access!")
    except Exception as e:
        pytest.fail(f"Failed to read data through kerchunk: {e}")

    # Clean up
    reader.close()


# def test_om_to_xarray():
#     """Test opening kerchunked OM file with xarray"""
#     import xarray as xr

#     # Create references using SingleOmToZarr
#     om_to_zarr = SingleOmToZarr(test_file)
#     references = om_to_zarr.translate()
#     om_to_zarr.close()

#     # Open with xarray
#     store = refs_as_store(references)

#     try:
#         # Open dataset with xarray using zarr engine
#         ds = xr.open_zarr(store, consolidated=False)

#         # Basic validation
#         reader = omfiles.OmFilePyReader(test_file)
#         assert ds.dims == dict(zip(["time", "y", "x"], reader.shape))

#         # Get some data to verify decompression pipeline
#         data_sample = ds.isel(time=0, y=0, x=slice(0, 5)).data
#         assert data_sample.shape == (5,)

#         print(f"Successfully opened dataset with xarray: {ds}")
#         print(f"Sample data: {data_sample}")

#         reader.close()
#     except Exception as e:
#         pytest.fail(f"Failed to open with xarray: {e}")


if __name__ == "__main__":
    # Run tests directly if file is executed
    test_single_om_to_zarr()
    # test_om_to_xarray()
