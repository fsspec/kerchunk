References specification
========================

The content of a reference set should match the given description here.
``fsspec``'s ``ReferenceFileSystem`` expects this kind of input.

Version 0
---------

Prototype spec for the structure required by ReferenceFileSystem:

.. code-block:: json

    {
      "key0": "data",
      "key1": ["protocol://target_url", 10000, 100]
    }

where:

* ``key0`` includes data as-is (stored as text)
* ``key1`` refers to a data file URL, the offset within the file (in bytes), and the length of the data item (in bytes).

For example, Zarr data in this proposed spec might be represented as:

.. code-block:: json

    {
      ".zgroup": "{\n    \"zarr_format\": 2\n}",
      ".zattrs": "{\n    \"Conventions\": \"UGRID-0.9.0\n\"}",
      "x/.zattrs": "{\n    \"_ARRAY_DIMENSIONS\": [\n        \"node\"\n ...",
      "x/.zarray": "{\n    \"chunks\": [\n        9228245\n    ],\n    \"compressor\": null,\n    \"dtype\": \"<f8\",\n  ...",
      "x/0": ["s3://bucket/path/file.nc", 294094376, 73825960]
    }

Version 1
---------

Metadata structure in JSON. We note, for future possible binary storage, that "version", "gen" and "templates" should
be considered attributes, and "refs" as the data that ought to dominate the storage size. The previous definition,
Version 0, is compatible with the "refs" entry, but here we add features. It will also be possible to *expand*
this new enhanced spec into Version 0 format.


.. code-block::

    {
      "version": (required, must be equal to) 1,
      "templates": (optional, zero or more arbitrary keys) {
        "template_name": jinja-str
      },
      "gen": (optional, zero or more items) [
        "key": (required) jinja-str,
        "url": (required) jinja-str,
        "offset": (optional, required with "length") jinja-str,
        "length": (optional, required with "offset") jinja-str,
        "dimensions": (required, one or more arbitrary keys) {
          "variable_name": (required)
            {"start": (optional) int, "stop": (required) int, "step": (optional) int}
            OR
            [int, ...]
        }
      ],
      "refs": (optional, zero or more arbitrary keys) {
        "key_name": (required) str OR [url(jinja-str)] OR [url(jinja-str), offset(int), length(int)]
      }
    }

Where:

- ``jinja-str`` is a string which will be rendered by jinja2 or its non-python equivalent; i.e., it may be
  a literal string, or may include "{{..}}" annotations, where:

  - for the values associated with a template_name, the variables are to be passed in reference URL strings that use this template
  - for the values within a "gen" object, variables come from the "dimensions" and "templates"

- the ``str`` format of a reference value may be:

  - a string starting "base64:", which will be decoded to binary
  - any other string, interpreted as ascii data

- the str version of ref values indicates data, the one-element array a whole url, and the three-element version
  a binary section of a url

Here is an example

.. code-block:: json

    {
        "version": 1,
        "templates": {
            "u": "server.domain/path",
            "f": "{{c}}"
        },
        "gen": [
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "offset": "{{(i + 1) * 1000}}",
                "length": "1000",
                "dimensions":
                  {
                    "i": {"stop":  5}
                  }
            }
        ],
        "refs": {
          "key0": "data",
          "key1": ["http://target_url", 10000, 100],
          "key2": ["http://{{u}}", 10000, 100],
          "key3": ["http://{{f(c='text')}}", 10000, 100]
        }
    }

Here the variable ``i`` takes the values ``[0, 1, 2, 3, 4]``, which could have been provided in array form. Where there
is more than one variable, a cartesian product is formed.

This example evaluates to the Version 0 equivalent

.. code-block:: json

    {
      "key0": "data",
      "key1": ["http://target_url", 10000, 100],
      "key2": ["http://server.domain/path", 10000, 100],
      "key3": ["http://text", 10000, 100],
      "gen_key0": ["http://server.domain/path_0", 1000, 1000],
      "gen_key1": ["http://server.domain/path_1", 2000, 1000],
      "gen_key2": ["http://server.domain/path_2", 3000, 1000],
      "gen_key3": ["http://server.domain/path_3", 4000, 1000],
      "gen_key4": ["http://server.domain/path_4", 5000, 1000]
    }

such that accessing, for instance, "key0" returns ``b"data"`` and accessing "gen_key0" returns 1000 bytes
from the given URL, at an offset of 1000.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>

Parquet references
------------------

Since JSON is rather verbose, it is easy with enough chunks to make a references file
that is too big: slow to load and heavy on memory. Although the former can be
alleviated by compression (I recommend Zstd), the latter cannot. This can
become particularly apparent during the combine phase when loading many reference sets.

The class `fsspec.implementations.reference.LazyReferenceMapper`_ provides an
alternative *implementation*, and its on-disk layout effectively is a new reference
spec, and we describe it here. The class itself has a dict mapper interface, just
like the rendered references from JSON files; except that it assumes that it is
working on a zarr dataset. This is because the references are split into files, and
an array's shape/chunk information is used to figure out which reference file
to load.

.. _fsspec.implementations.reference.LazyReferenceMapper: https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=lazyreference#fsspec.implementations.reference.LazyReferenceMapper

The following code

.. code-block:: python

    lz = fsspec.implementations.reference.LazyReferenceMapper.create("ref.parquet")
    z = zarr.open_group(lz, mode="w")
    d = z.create_dataset("name", shape=(1,))
    d[:] = 1
    g2 = z.create_group("deep")
    d = g2.create_dataset("name", shape=(1,))
    d[:] = 1

produces files

.. code-block:: text

    ref.parquet/deep/name/refs.0.parq
    ref.parquet/name/refs.0.parq
    ref.parquet/.zmetadata

Here, .zmetadata is all of the metadata of all of all subgroups/arrays (similar to
zarr "consolidated metadata"), with two top-level fields: "metadata" (dict[str, str]
all of the
zarr metadata key/values) and "record_size", an integer set during ``.create()``.

Each parquet file contains references within the corresponding path to where it is.
For example, key "name/0" will be the zeroth reference in "./name/refs.0.parq". If
there are multiple dimensions, normal C indexing is used to find the Nth reference,
and there are up to "record_size" references (default 10000) in the first file;
reference >10000,<=20000 would be in "./name/refs.2.parquet". Each file is (for now)
padded to record_size, but they compress really well.

Each row of the parquet data contains fields

.. code-block::

    path: optional str/categorical, remote location URL
    offset: int, start location of block
    size: int, number of bytes in block
    raw: optional bytes, binary data

If ``raw`` is populated, this is the data of the key. If ``path`` is
populated but size is 0, it is the whole file indicated (like a JSON [url] reference).
Otherwise, it is a byte block in the indicated file (like a JSON [url, offset, size] reference).
If both ``raw`` and ``path`` are NULL, the key does not exist.

We reserve the possibility to store small array data in .zmetadata instead
of creating a small/mostly empty parquet file for each.
