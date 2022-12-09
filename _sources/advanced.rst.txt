Advanced Topics
===============

Using Dask
----------

Scanning and combining datasets can be computationally intensive and may
require a lot of bandwidth for some data formats. Where the target data
contains many input files, it makes sense to parallelise the job with
dask and maybe disrtibuted the workload on a cluster to get additional
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
batches (which reducec a lot of redundancy between them) and then
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
