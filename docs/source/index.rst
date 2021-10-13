
fsspec-reference-maker
======================

This library allows you to create "references" to binary blocks of data in other files.
This means that you can create a virtual aggregate dataset over potentially many source
files, for efficient, parallel and cloud-friendly *in-situ* access without having to copy or
translate the originals. It is a gateway to in-the-cloud massive data processing while
the data providers still insist on using legacy formats for archival storage.

Introduction
------------

There is a lot of data available on the Internet. A lot. You won't believe how much there is.
You may think a 1TB disk is big, but that's peanuts to the Internet.

Data are stored in many different
arrangements of files in many different file formats.

For binary storage of array data, essentially all formats involve taking blocks of in-memory
C buffers and encoding/compressing them to disc, with some additional metadata describing
the details of that buffer plus any other attributes. This description can be applied to a
very wide variety of data formats.


Beyond python
-------------

Other uses
----------


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   test_example
   detail
   cases
   spec
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
