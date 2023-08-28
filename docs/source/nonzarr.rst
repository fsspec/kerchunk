Non-zarr uses
=============

The majority of this documentation focuses on reading arrays via ``zarr``. However,
there are other applications where accessing files in a particular directory structure
or reading binary pieces of large files at known offsets is very useful. Here descriptions
of some use cases, yet to be developed.

.tar.zstd
---------

Often, files are stored for archival by first concatenating and then compressing them.
This is done to save space and data transfer upon access. However, it makes direct
access of the contents hard, and often means that access can only happen by downloading
whole files and unpacking them locally.

The TAR format is simple enough to be able to index at which offsets which files start,
and Zstd is a modern enough compressor that it can be read block-wise, so it should
be possible to get the best of both worlds: to only pull the parts of the data that
are actually needed (in parallel!), but still achieve good compression for long-term
cost saving.

.csv/.json
----------

Binary formats are much more efficient at data storage, both in terms of space and
performance. However, a whole lot of data is still sored in text formats like CSV, JSON
and others. For these two, subsequent rows are indicated by a terminating newline character,
which means that parallel/random access is possible by searching for them.

However, both formats allow for newline characters that are *not* record terminators,
because they can be included in quoted strings. In this case, if reading from some
random offset in the file, it's not possible to know whether a given newline is indeed
a record terminator, and any misidentification will lead to parsing failure.

If we can scan the file just once, we can determine where the rows end, and which offsets
are safe to use, allowing parallel/distributed processing of files with embedded
newlines.

parquet/orc/feather
-------------------

Parquet is an efficient tabular data format. Sometimes, however, it is useful to
*partition* the data on low-cardinality rows, so save on storage and make it easier
to exclude unneeded files from processing. For instance, if one file in a dataset
has the name "month=may/part.001.parquet", then all the values of the column "month"
are "may: for this part of the dataset, and there is no need to store the column
values.

Since we have the ability to name files as we wish in a virtual file system, we can
use this feature of parquet to be able to assign values to partitions even when the
files were not written as part of the same dataset. We can, in effect, perform smart
concatenation without having to move/copy files around.

Feather is a more raw buffer-wise serialisation of Arrow data, and we may be able to
construct logical feather files out of buffer pieces (and serialised feather metadata)
to present to the pyarrow API.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
