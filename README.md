# fsspec-reference-maker

Functions to make reference descriptions for ReferenceFileSystem


### Version 0

Prototype spec for the structure required by ReferenceFileSystem:

```json
{
  "key0": "data",
  "key1": ["protocol://target_url", 10000, 100]
}
```
where:
* `key0` includes data as-is (stored as text)
* `key1` refers to a data file URL, the offset within the file (in bytes), and the length of the data item (in bytes).

For example, Zarr data in this proposed spec might be represented as:

```json
{
  ".zgroup": "{\n    \"zarr_format\": 2\n"},
  ".zattrs": "{\n    \"Conventions\": \"UGRID-0.9.0\n\"},
  "x/.zattrs": "{\n    \"_ARRAY_DIMENSIONS\": [\n        \"node\"\n ...",
  "x/.zarray": "{\n    \"chunks\": [\n        9228245\n    ],\n    \"compressor\": null,\n    \"dtype\": \"<f8\",\n  ...",
  "x/0": ["s3://bucket/path/file.nc", 294094376, 73825960]
},
```

### Version 1

Metadata structure in JSON. We note, for future possible binary storage, that "version", "gen" and "templates" should
be considered attributes, and "refs" as the data that ought to dominate the storage size. The previous definition,
Version 0, is compatible with the "refs" entry, but here we add features. It will also be possible to *expand*
this new enhanced spec into Version 0 format.

```json
{
    "version": 1,
    "templates": {
        "u": "long_text_template",
        "f": "{c}"
    },
    "gen": [
        {
            "key": "gen_key{i}",
            "url": "protocol://{u}_{i}",
            "offset": "{(i + 1) * 1000}",
            "length": "1000",
            "i": "range(9)"
        }   
    ],
    "refs": {
      "key0": "data",
      "key1": ["protocol://target_url", 10000, 100],
      "key2": ["protocol://{u}", 10000, 100],
      "key3": ["protocol://{f(c='text')}", 10000, 100]
    }
}
```

Explanation of fields follows. Only "version" and "refs" are required:

- version: set to 1 for this spec.
- templates: set of named string templates. These can be plain strings, to be included verbatim, or format strings
  (anything containing "{" and "}" characters) which will be called with parameters. The format specifiers for each
  variable follows the python string formatting spec.
- gen: programmatically generated key/value pairs. Each entry adds one or more items to "refs"; in practice, in the
  implementation, we may choose to populate these or create them on-demand. Any of the fields can contain
  templated parameters.
    - key, url: generated key names and target URLs
    - offset, length: to define the bytes range, will be converted to int
    - additional named parameters: for each iterable found (i.e., returns successfully from `iter()`), creates a 
      dimension of generated keys
- refs: keys with either data or [url, offset, length]. The URL will be treated as a template if it contains 
  "{" and "}".

In the example, "key2" becomes ["protocol://long_text_template", ..] and "key3" becomes ["protocol://text", ..].
Also contained will be keys "gen_ref0": ["protocol://long_text_template_0", 1000, 1000] to "gen_ref8":
["protocol://long_text_template_9", 9000, 1000].


## Examples

Run a notebook example comparing reading HDF5 using this approach vs. native Zarr format: <br> 
[![Binder](https://aws-uswest2-binder.pangeo.io/badge_logo.svg)](https://aws-uswest2-binder.pangeo.io/v2/gh/intake/fsspec-reference-maker/main?urlpath=lab%2Ftree%2Fexamples%2Fike_intake.ipynb)

