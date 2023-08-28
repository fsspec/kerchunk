
kerchunk
========

Kerchunk is a library that provides a unified way to represent a variety of chunked, compressed data formats
(e.g. NetCDF/HDF5, GRIB2, TIFF, ...),
allowing efficient access to the data from traditional file systems or cloud object storage.
It also provides a flexible way to create
virtual datasets from multiple files.  It does this by extracting the byte ranges,
compression information and other information about the
data and storing this metadata in a new, separate object.  This means that you can
create a virtual aggregate dataset over potentially many source
files, for efficient, parallel and cloud-friendly *in-situ* access without having to copy or
translate the originals. It is a gateway to in-the-cloud massive data processing while
the data providers still insist on using legacy formats for archival storage.

*Why Kerchunk*:

We provide the following things:

- completely serverless architecture
- metadata consolidation, so you can understand a many-file dataset (metadata plus physical storage) in a single read
- read from all of the storage backends supported by fsspec, including object storage (s3, gcs, abfs, alibaba), http,
  cloud user storage (dropbox, gdrive) and network protocols (ftp, ssh, hdfs, smb...)
- loading of various file types (currently netcdf4/HDF, grib2, tiff, fits, zarr), potentially heterogeneous within a
  single dataset, without a need to go via the specific driver (e.g., no need for h5py)
- asynchronous concurrent fetch of many data chunks in one go, amortizing the cost of latency
- parallel access with a library like zarr without any locks
- logical datasets viewing many (>~millions) data files, and direct access/subselection to them via coordinate
  indexing across an arbitrary number of dimensions


Introduction
------------

This library was created to solve the problem of reading existing scientific datatypes as efficiently as possible in the cloud. The amount of observed and simulated data
is now too large to be handled effectively via download and local processing.  In the cloud era, the answer is to move compute to the data in the Cloud, process in
parallel, and access the data *in situ*, and only as much as needed to solve a problem or use case.

Datasets are stored in many different file formats, and often as collections of files.
Many of these formats are designed for on-premises filesystems and are hard or inefficient
to use seamlessly in the cloud.

For binary storage of array data, essentially all formats involve taking blocks of in-memory
C buffers and encoding/compressing them to disc, with some additional metadata describing
the details of that buffer plus any other attributes. This description can be applied to a
very wide variety of data formats.

The primary purpose of ``kerchunk`` is to find where these binary blocks are, and how to decode them,
so that blocks from one or more files can be arranged into aggregate datasets accessed via the
`zarr`_ library and the power of `fsspec`_. To understand how this works, please read
:doc:`detail`. Or consider the PyData talk: `All You Need Is Zarr`_.


.. _zarr: https://zarr.readthedocs.io/en/stable/
.. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
.. _All You Need Is Zarr: https://www.youtube.com/watch?v=0bqpxX3Nn_A

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   test_example
   tutorial
   detail
   cases
   spec
   beyond
   nonzarr
   reference
   contributing
   advanced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
