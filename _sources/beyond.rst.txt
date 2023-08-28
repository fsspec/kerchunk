Beyond Python
=============

The reference files currently stored in JSON format and can be readily interpreted
in any language, as documented in detail in the :doc:`spec`.
The contents of each key
is either encoded binary data or a URL/offset/size set. It can be loaded as long
as the language being used can access the particular URL type.

To interpret the blocks as parts of a zarr dataset, the language should
have an `implementation of zarr`_, as well as whichever binary codecs the target
requires (maybe nothing for plain binary, or common compressors like gzip, but might
be more specific). You would need to write some code to expose the reference set
as a storage object that ``zarr`` can use.

.. _implementation of zarr: https://github.com/zarr-developers/zarr_implementations

One example of a reference dataset used via a JS implementation, applied to multi-scale
TIFF microscopy, can be found
at https://observablehq.com/@manzt/ome-tiff-as-filesystemreference.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
