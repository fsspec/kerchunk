Beyond Python
=============

The reference files are generally in JSON format and can be readily interpreted
in any language, as documented in detail in the :doc:`spec`.
Essentially, the contents of each key
is either encoded binary data, or a URL/offset/size set. It can be loaded so long
as the language in question can access that particular URL type, which will very likely
be yes for local files and HTTP, but maybe harder for more obscure URLs.

To interpret the blocks as parts of a zarr dataset, the language should, of course,
have an `implementation of zarr`_, as well as whichever binary codecs the target
requires (maybe nothing for plain binary, or common compressors like gzip, but might
be more specific). You would need to write some code to expose the reference set
as a storage object that ``zarr`` can use.

.. _implementation of zarr: https://github.com/zarr-developers/zarr_implementations

One example of a reference dataset used via a JS implementation, applied to multi-scale
TIFF microscopy, can be found
at `https://observablehq.com/@manzt/ome-tiff-as-filesystemreference`__.
