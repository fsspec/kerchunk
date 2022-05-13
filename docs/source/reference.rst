API Reference
*************

.. currentmodule:: kerchunk

.. autosummary::
   kerchunk.hdf.SingleHdf5ToZarr
   kerchunk.grib2.scan_grib
   kerchunk.grib2.GRIBCodec
   kerchunk.combine.MultiZarrToZarr
   kerchunk.fits.process_file
   kerchunk.tiff.tiff_to_zarr
   kerchunk.netCDF3.netcdf_recording_file
   kerchunk.netCDF3.RecordArrayMember

.. autoclass:: kerchunk.hdf.SingleHdf5ToZarr
    :members:

.. autofunction:: kerchunk.grib2.scan_grib

.. autoclass:: kerchunk.grib2.GRIBCodec

.. autoclass:: kerchunk.combine.MultiZarrToZarr
    :members:

.. autofunction:: kerchunk.fits.process_file

.. autofunction:: kerchunk.tiff.tiff_to_zarr

.. autoclass:: kerchunk.netCDF3.netcdf_recording_file
    :members: __init__, translate

.. autoclass:: kerchunk.netCDF3.RecordArrayMember
    :members: __init__
