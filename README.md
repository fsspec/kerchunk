# fsspec-reference-maker
Functions to make reference descriptions for ReferenceFileSystem


Spec for the structure required by ReferenceFileSystem:

```json
{
  "key0": "data",
  "key1": {
    ("protocol://target_url", 100, 200)
  }
}
```

where "key0" includes data as-is (stored as text), and "key1" refers to the given remote URL, the set of bytes
between the given start-stop values (start is inclusive, end is not, as is typical for python).

For example, zarr data might be stored like

```json
{
  ".zgroup": "{\n    "zarr_format": 2\n}",
  ".zattrs": "{\n    "Conventions": "UGRID-0.9.0\n}",
  "x/.zattrs": "{\n    "_ARRAY_DIMENSIONS": [\n        "node"\n    ],\n    "_FillValue": "NaN",\n    "long_name": "longitude",\n    "positive": "east",\n    "standard_name": "longitude",\n    "units": "degrees_east"\n}",
  "x/.zarray": "{\n    "chunks": [\n        9228245\n    ],\n    "compressor": null,\n    "dtype": "<f8",\n    "fill_value": "NaN",\n    "filters": null,\n    "order": "C",\n    "shape": [\n        9228245\n    ],\n    "zarr_format": 2\n}",
  "x/0": ('s3://bucket/path/file.nc', 294094376, 367920336)
},
```
