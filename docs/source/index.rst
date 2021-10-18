
kerchunk
============

This library allows you to create "references" to binary blocks of data in other files.
This means that you can create a virtual aggregate dataset over potentially many source
files, for efficient, parallel and cloud-friendly *in-situ* access without having to copy or
translate the originals. It is a gateway to in-the-cloud massive data processing while
the data providers still insist on using legacy formats for archival storage.

Introduction
------------

The scale of data available for processing is far greater than individual machines or
download speeds can handle. Thus, in the cloud era, you need to move compute to the data, process in
parallel, and access the data *in situ*, and only as you need to solve a problem.

Data are stored in many different arrangements of files in many different file formats.
Unfortunately, many of these formats are designed for a previous generation and hard
to use seamlessly in the cloud. That is particularly true of archival data, which, by
design, uses old, established formats.

For binary storage of array data, essentially all formats involve taking blocks of in-memory
C buffers and encoding/compressing them to disc, with some additional metadata describing
the details of that buffer plus any other attributes. This description can be applied to a
very wide variety of data formats.

The primary purpose of ``kerchunk`` is to find where these binary blocks are, and how to decode them,
so that blocks from one or more files can be arranged into aggregate datasets accessed via the
`zarr`_ library and the power of `fsspec`_. To understand how this works, please read
:doc:`detail`. A full worked example for multiple HDF5 files is presented in :doc:`test_example`.

.. _zarr: https://zarr.readthedocs.io/en/stable/
.. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   test_example
   detail
   cases
   spec
   beyond
   nonzarr
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
