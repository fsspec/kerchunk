Advanced Topics
===============

Using Dask
----------

Scanning and combining datasets can be computationally intensive and may
require a lot of bandwidth for some data formats. Where the target data
contains many input files, it makes sense to parallelise the job with
dask and maybe distribute the workload on a cluster to get additional
CPUs and network performance.

Simple parallel
~~~~~~~~~~~~~~~

The simplest case is for processing many individual files in parallel.
Let's say you have a list of input files; you will need to encapsulate
the processing on each in a single function. In this mode, it is typical
to save the single-file outputs to files, although returning them is OK
too (especially if you mean to combine them immediately).

Here is an example for HDF5 files. The caller should make sure the
storage options and any parameters needed for the transformer are in place.

.. code-block:: python

    import ujson, fsspec, dask

    def process(url, outputfile, storage_options_in={}, storage_options_out={}):
        transformer = kerchunk.hdf.SingleHdf5ToZarr(url, **storage_options_in)
        refs = transformer.translate()
        with fsspec.open(outputfile, mode="wt", **storage_options_out) as f:
            ujson.dump(refs, f)

    tasks = [dask.delayed(process)(u, o)
             for u, o in zip(infilenames, outfilenames)]
    dask.compute(tasks)

Tree reduction
~~~~~~~~~~~~~~

In some cases, the combine process can itself be slow or memory hungry.
In such cases, it is useful to combine the single-file reference sets in
batches (which reduce a lot of redundancy between them) and then
combine the results of the batches. This technique is known as tree
reduction. An example of doing this by hand can be seen `here`_.

.. _here: https://gist.github.com/peterm790/5f901453ed7ac75ac28ed21a7138dcf8

We also provide :func:`kerchunk.combine.auto_dask` as a convenience. This
function is a one-stop call to process the individual inputs, combine
them in batches, and then combine the results of those batches into a
final combined references set.

The ``auto_dask`` function takes a number of dicts as arguments, and users
should consult the docstrings of the specific class which decodes the
input files, and also of :class:`kerchunk.combine.MultiZarrToZarr`. Note that
any "preprocessing" for ``MultiZarrToZarr`` will be performed *before* the
batch stage, and any "postprocessing" only *after* the final combine.

Archive Files
-------------

It is often convenient to distribute datasets by wrapping multiple files
into an archive, ZIP or TAR. If those files are of formats supported by
``kerchunk``, they can be directly scanned with something like

.. code-block:: python

    transformer = kerchunk.netCDF3.NetCDF3ToZarr(
        "tar://myfile.nc::file://archive.tar",
        inline_threshold=0
    )
    out = transformer.translate()


where "myfile.nc" is a member file of the local archive.

.. note::

    We have turned off inlining here (it can be done
    later with :func:`kerchunk.utils.do_inline`; support for this
    will come later.

At this point, the
generated references will contain URLs "tar://myfile.nc::file://archive.tar",
which are problematic for loading, so we can transform them to point to
ranges in the original tar file instead, and then transform back to
nominal form, ready to use. We may automate these steps in the future.

.. code-block:: python

    out2 = kerchunk.utils.dereference_archives(out)
    # optional out2 = kerchunk.utils.do_inline(out2, 100)
    final = kerchunk.utils.consolidate(out2)

Now the references are all "file://archive.tar", and the reference set
can be used directly or in combining.

.. warning::

   For ZIP archives, only uncompressed members can be accessed this way

Parquet Storage
---------------

JSON is very convenient as a storage format for references, because it is
simple, human-readable and ubiquitously supported. However, it is not the most
efficient in terms of storage size of parsing speed. For python, in particular,
it comes with the added downside of repeated strings becoming separate python
string instances, greatly inflating memory footprint at load time.

To overcome these problems, and in particular keep down the memory use for the
end-user of kerchunked data, we can convert references to be stored in parquet,
and use them with ``fsspec.implementations.reference.ReferenceFileSystem``,
an alternative new implementation designed to work only with parquet input.

The principle benefits of the parquet path are:

- much more compact storage, typically 2x smaller than compressed JSON or 10x
  smaller than uncompressed

- correspondingly faster instantiation of a filesystem, since much of that time
  is taken by loading in the bytes of the references

- smaller in-memory size (e.g., a python int requires 28 bytes, but an int in
  an array needs 4 or 8.

- optional lazy loading, by partitioning the references into files by key; only
  the variables you actually access need to have their references loaded

- optional dictionary encoding of URLs in the case that there are may repeated
  URLs (many references per target file). In this format, each unique URL is only
  stored in memory once.

The only access point to the new parquet storage is
:func:`kerchunk.df.refs_to_dataframe`, which transforms an existing kerchunk
reference set (in memory as dicts or in a JSON file) to parquet. Careful reading
of the docstring is recommended, to understand the options. More options may
be added.

.. note::

   For now, :class:`kerchunk.combine.MultiZarrToZarr` only operates on JSON/dict
   input. Therefore, ``refs_to_dataframe`` can only be used on the final output
   reference set. For a very large merge of many/large inputs, this may mean
   that the combine step requires a lot of memory, as will converting the
   output to parquet. However, the end-user should be able to access data via
   these references with much smaller  memory requirements.

A concrete workflow may be something like the following. Note that
:func:`kerchunk.combine.auto_dask` can execute the first three stages in
one go and may be faster, if you have a Dask cluster available.

.. code-block:: python

   from kerchunk import hdf, combine, df
   import fsspec.implementations.reference
   from fsspec.implementations.reference import LazyReferenceMapper
   from tempfile import TemporaryDirectory

   import xarray as xr

   files = fsspec.open(location_of_data)

   # Create LazyReferenceMapper to pass to MultiZarrToZarr
   fs = fsspec.filesystem("file")

   os.makedirs("combined.parq")
   out = LazyReferenceMapper.create(record_size=1000, root="combined.parq", fs=fs)

   # Create references from input files
   single_ref_sets = [hdf.SingleHdf5ToZarr(_).translate() for _ in files]

   out_dict = MultiZarrToZarr(
    single_ref_sets,
    remote_protocol="s3",
    concat_dims=["time"],
    remote_options={"anon": True},
    out=out
    ).translate()

   out.flush()

   df.refs_to_dataframe(out_dict, "combined.parq")

   fs = fsspec.implementations.reference.ReferenceFileSystem(
       "combined.parq", remote_protocol="s3", target_protocol="file", lazy=True)
   ds = xr.open_dataset(
       fs.get_mapper(), engine="zarr",
       backend_kwargs={"consolidated": False}
   )


At this point, xarray has loaded the metadata and coordinates only, so the
main reference files corresponding to the data variables have not been touched.
Even for a very large reference set, the memory use at this point should be <500MB.

As you access the variables of ``ds``, they will be loaded on demand and cached.
If using ``dask``, workers will also only load those references they need.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
