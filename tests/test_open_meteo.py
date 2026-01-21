import os

import fsspec
import numpy as np
import omfiles
import pytest
import zarr

from kerchunk.combine import MultiZarrToZarr
from kerchunk.df import refs_to_dataframe
from kerchunk.open_meteo import SingleOmToZarr, SupportedDomain
from kerchunk.utils import fs_as_store

zarr.config.config["async"]["concurrency"] = 1

def test_single_om_to_zarr():
    # Path to test file - adjust as needed
    chunk_no=1914
    test_file = f"tests/rh_icon_chunk{chunk_no}.om"
    """Test creating references for a single OM file and reading via Zarr"""
    # Create references using SingleOmToZarr
    om_to_zarr = SingleOmToZarr(test_file, inline_threshold=0, domain=SupportedDomain.dwd_icon, chunk_no=chunk_no)  # No inlining for testing
    references = om_to_zarr.translate()
    om_to_zarr.close()

    # Save references to json if desired. These are veryyy big...
    # with open("om_refs.json", "w") as f:
    #     json.dump(references, f)

    # Save references to parquet
    refs_to_dataframe(
        fo=references,              # References dict
        url="om_refs_parquet/",     # Output directory
        record_size=100000,         # Records per file, adjust as needed
        categorical_threshold=10    # URL encoding efficiency
    )

    # Create a filesystem from the references
    fs = fsspec.filesystem("reference", fo=references)
    store = fs_as_store(fs)

    # Open with zarr
    group = zarr.open(store, zarr_format=2)
    print("group['time']", group["time"][:])
    z = group["data"] # Here we just use a dummy data name we have hardcoded in SingleOmToZarr

    print("z.shape", z.shape)
    print("z.chunks", z.chunks)

    # Verify basic metadata matches original file
    reader = omfiles.OmFileReader(test_file)
    assert z.shape == reader.shape, f"Shape mismatch: {z.shape} vs {reader.shape}"
    assert str(z.dtype) == str(reader.dtype), f"Dtype mismatch: {z.dtype} vs {reader.dtype}"
    assert z.chunks == reader.chunks, f"Chunks mismatch: {z.chunks} vs {reader.chunks}"

    # TODO: Using the following chunk_index leads to a double free / corruption error!
    # Even with a concurrency of 1: `zarr.config.config["async"]["concurrency"] = 1`
    # Most likely, because zarr and open-meteo treat partial chunks differently:
    # om-files encode partial chunks with a reduced dimension, while zarr most likely expects a full block of data?
    # chunk_index = (slice(90, 100), 2878, ...)

    # Test retrieving a specific chunk
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
        pytest.fail(f"Failed to read kerchunked data through zarr: {e}")

    # Clean up
    reader.close()

def test_multizarr_to_zarr():
    """Test combining two OM files into a single kerchunked Zarr reference"""
    chunk_no1=1914
    chunk_no2=1915
    file1 = f"tests/rh_icon_chunk{chunk_no1}.om"
    file2 = f"tests/rh_icon_chunk{chunk_no2}.om"
    assert os.path.exists(file1), f"{file1} not found"
    assert os.path.exists(file2), f"{file2} not found"
    refs1 = SingleOmToZarr(file1, inline_threshold=0, domain=SupportedDomain.dwd_icon, chunk_no=chunk_no1).translate()
    refs2 = SingleOmToZarr(file2, inline_threshold=0, domain=SupportedDomain.dwd_icon, chunk_no=chunk_no2).translate()

    # Combine using MultiZarrToZarr
    mzz = MultiZarrToZarr(
        [refs1, refs2],
        concat_dims=["time"],
        coo_map={"time": "data:time"},
        identical_dims=["y", "x"],
        remote_protocol=None,
        remote_options=None,
        target_options=None,
        inline_threshold=0,
        out=None,
    )
    combined_refs = mzz.translate()

    # Save references to parquet
    refs_to_dataframe(
        fo=combined_refs,               # References dict
        url="om_refs_mzz_parquet/",     # Output directory
        record_size=100000,             # Records per file, adjust as needed
        categorical_threshold=10        # URL encoding efficiency
    )

    # Open with zarr
    fs = fsspec.filesystem("reference", fo=combined_refs)
    store = fs_as_store(fs)
    group = zarr.open(store, zarr_format=2)
    z = group["data"]

    # Open both original files for comparison
    reader1 = omfiles.OmFileReader(file1)
    reader2 = omfiles.OmFileReader(file2)

    # Check that the combined shape is the sum along the time axis
    expected_shape = list(reader1.shape)
    expected_shape[2] += reader2.shape[2]
    assert list(z.shape) == expected_shape, f"Combined shape mismatch: {z.shape} vs {expected_shape}"

    # Check that the first part matches file1 and the second part matches file2
    # (Assume 3D: time, y, x)
    slc1 = (5, 5, slice(0, reader1.shape[2]))
    slc2 = (5, 5, slice(reader1.shape[2], expected_shape[2]))
    np.testing.assert_allclose(z[slc1], reader1[slc1], rtol=1e-5)
    np.testing.assert_allclose(z[slc2], reader2[slc1], rtol=1e-5)

    print("✓ MultiZarrToZarr combined data matches originals!")

    reader1.close()
    reader2.close()


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
#         reader = omfiles.OmFileReader(test_file)
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
    test_multizarr_to_zarr()
    # test_om_to_xarray()
