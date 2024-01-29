API Reference
*************

File format backends
--------------------

.. autosummary::
   kerchunk.hdf.SingleHdf5ToZarr
   kerchunk.grib2.scan_grib
   kerchunk.fits.process_file
   kerchunk.tiff.tiff_to_zarr
   kerchunk.netCDF3.NetCDF3ToZarr

.. autoclass:: kerchunk.hdf.SingleHdf5ToZarr
    :members:

.. autofunction:: kerchunk.grib2.scan_grib


.. autofunction:: kerchunk.fits.process_file

.. autofunction:: kerchunk.tiff.tiff_to_zarr

.. autoclass:: kerchunk.netCDF3.NetCDF3ToZarr
    :members: __init__, translate

Codecs
------

.. autosummary::
   kerchunk.codecs.GRIBCodec
   kerchunk.codecs.AsciiTableCodec
   kerchunk.codecs.FillStringsCodec
   kerchunk.codecs.VarArrCodec
   kerchunk.codecs.RecordArrayMember


.. autoclass:: kerchunk.codecs.GRIBCodec
    :members: __init__

.. autoclass:: kerchunk.codecs.AsciiTableCodec
    :members: __init__

.. autoclass:: kerchunk.codecs.FillStringsCodec
    :members: __init__

.. autoclass:: kerchunk.codecs.VarArrCodec
    :members: __init__

.. autoclass:: kerchunk.codecs.RecordArrayMember
    :members: __init__

Combining
---------

.. autosummary::
   kerchunk.combine.MultiZarrToZarr
   kerchunk.combine.merge_vars
   kerchunk.combine.concatenate_arrays
   kerchunk.combine.auto_dask
   kerchunk.combine.drop

.. autoclass:: kerchunk.combine.MultiZarrToZarr
    :members: __init__, append, translate

.. autofunction:: kerchunk.combine.merge_vars

.. autofunction:: kerchunk.combine.concatenate_arrays

.. autofunction:: kerchunk.combine.auto_dask

.. autofunction:: kerchunk.combine.drop

Utilities
---------

.. autosummary::
    kerchunk.utils.rename_target
    kerchunk.utils.rename_target_files
    kerchunk.utils.subchunk
    kerchunk.utils.dereference_archives
    kerchunk.utils.consolidate
    kerchunk.utils.do_inline
    kerchunk.utils.inline_array
    kerchunk.df.refs_to_dataframe

.. autofunction:: kerchunk.utils.rename_target

.. autofunction:: kerchunk.utils.rename_target_files

.. autofunction:: kerchunk.tiff.generate_coords

.. autofunction:: kerchunk.utils.subchunk

.. autofunction:: kerchunk.utils.dereference_archives

.. autofunction:: kerchunk.utils.consolidate

.. autofunction:: kerchunk.utils.do_inline

.. autofunction:: kerchunk.utils.inline_array

.. autofunction:: kerchunk.df.refs_to_dataframe

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
