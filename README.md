# fsspec-reference-maker
Functions to make reference descriptions for ReferenceFileSystem


Proposed spec for the structure required by ReferenceFileSystem:

```
{
  "key0": "data",
  "key1": {
    ["protocol://target_url", 10000, 100]
  }
}
```
where:
* `key0` includes data as-is (stored as text)
* `key1` refers to a data file URL, the offset within the file (in bytes), and the length of the data item (in bytes).

For example, Zarr data in this proposed spec might be represented as:

```
{
  ".zgroup": "{\n    \"zarr_format\": 2\n"},
  ".zattrs": "{\n    \"Conventions\": \"UGRID-0.9.0\n\"},
  "x/.zattrs": "{\n    \"_ARRAY_DIMENSIONS\": [\n        \"node\"\n ...",
  "x/.zarray": "{\n    \"chunks\": [\n        9228245\n    ],\n    \"compressor\": null,\n    \"dtype\": \"<f8\",\n  ...",
  "x/0": ["s3://bucket/path/file.nc", 294094376, 73825960]
},
```


Click here to run a notebook comparing reading HDF5 with Zarr, using this approach:
[![Binder](https://aws-uswest2-binder.pangeo.io/badge_logo.svg)](https://aws-uswest2-binder.pangeo.io/v2/gh/rsignell-usgs/fsspec-reference-maker/example?filepath=examples%2Fike_intake.ipynb)
