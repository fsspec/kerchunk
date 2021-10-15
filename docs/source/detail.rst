Detailed description
====================

We will demonstrate how this library can lead to parallel, cloud-friendly access of data with
specific reference to netCDF4/HDF5 files, which were the motivating first case we attempted.

.. image:: images/binary_buffer.png
  :width: 400
  :alt: HDF5 files contain C buffers

.. image:: images/multi_refs.png
  :width: 400
  :alt: Zarr can view many files as a single dataset
